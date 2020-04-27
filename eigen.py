# ====================================================================================== #
# Module for solving dynamical equations corresponding to passive and active adaptation
# using recursive, eigenfunction method.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from numba import types
from warnings import warn
import multiprocess as mp
from threadpoolctl import threadpool_limits



class VisionBinary():
    """Eigenvalue formulation for binary (h0, -h0) environment state.
    """
    def __init__(self, tau, h0, beta, nBatch,
                 L=1, dx=1e-3, **kwargs):
        """
        Parameters
        ----------
        tau : float
            Time scale for flipping external field.
        h0 : float
            Magnitude of external field.
        beta : float
            Weight on memory.
        nBatch : int
            Number of samples from which to learn. This determines Gaussian noise profile.
        L : float, 1
            Max positive value of lattice. Domain spans [-L, L].
        dx : float, 1e-3
            Spacing of points on lattice spreading out from x=0.
        **kwargs
        """
        
        # check input args
        assert tau>0
        assert 0 <= h0 < L
        assert 0<=beta<=1
        if nBatch<1000:
            warn("If nBatch is too small, Gaussian assumption is broken.")
        assert dx>=1e-4, "dx is too small for fast computation"
        
        # set up
        self.tau = tau
        self.h0 = h0
        self.beta = beta
        self.nBatch = nBatch
        self.L = L
        self.dx = dx
        x = np.arange(L//dx+1) * dx
        x = np.concatenate((-x[1:][::-1], x))
        self.x = x
        self.cache_phatpos = {}  # cache for different beta
        self.cache_phatneg = {}
        self.cache_phat = {}
        
        s = np.sqrt(pplus(h0) * pminus(h0) / nBatch)
        self.eps_gaussian = lambda x, mu=0., sigma=s: (np.exp(-(x-mu)**2 / 2 / sigma**2) /
                                                       np.sqrt(2 * np.pi) / sigma)
        self.M = np.ones(x.size) * dx  # integration operator
        self.M[0] /= 2
        self.M[-1] /= 2
        self.trapz_row = lambda mat, M=self.M : mat.dot(M)
        
        dhhat = (x[:,None] - x[None,:]*beta) / (1-beta)  # what would delta hhat be given the change to hhat
        eps = (np.tanh(h0) - np.tanh(dhhat)) / 2  # error in measured field relative to h0
        self.staycoeff = self.eps_gaussian(eps) * .5 * (1-np.tanh(dhhat)**2)  # term that goes into integral
        eps = (-np.tanh(h0) - np.tanh(dhhat)) / 2  # error in measured field relative to -h0
        self.leavecoeff = self.eps_gaussian(eps) * .5 * (1-np.tanh(dhhat)**2)  # term that goes into integral
        
        # basic precision checks (for one's sanity)
        assert np.isclose(self.eps_gaussian(x).dot(self.M), 1)
        assert np.isclose(np.linalg.norm(x+x[::-1]), 0)

        self._init_addon(**kwargs)

    def _init_addon(self):
        return
        
    def stay(self, phat):
        """Transformation of density given that external field stays the same."""
        return self.trapz_row(phat[None,:] * self.staycoeff) / (1-self.beta)

    def leave(self, phat):
        """Transformation of density given that external field flips in sign."""
        return self.trapz_row(phat[None,:] * self.leavecoeff) / (1-self.beta)

    def apply_transform(self,
                        phat,
                        recurse=False,
                        recurse_skip=0,
                        run_checks=False):
        """One iteration of probability density transformation.
        
        Parameters
        ----------
        phat : ndarray
        recurse : int, False
        recurse_skip : int, 0
        run_checks : bool, False

        Returns
        -------
        ndarray
        """

        assert recurse>=0
        assert recurse%(recurse_skip+1)==0

        # action of first term, h0 -> h0
        # don't forget that when we calculate Gaussian distribution, we must account for scaling
        # transformation of hhat in order to calculate a normalized distribution
        newphat = np.zeros_like(self.x)

        # starting with h0 and staying at h0
        if recurse_skip:
            d = self.stay(phat)
            for i in range(recurse_skip-1):
                d = self.stay(d)
            d *= 1 - recurse_skip/self.tau
        else:
            d = (1 - 1/self.tau) * self.stay(phat)
        if recurse:
            d = self.apply_transform(d, recurse-1-recurse_skip)
        newphat += d

        # starting with h0 and switching to -h0
        if recurse_skip:
            d = self.leave(phat)
            for i in range(recurse_skip-1):
                d = self.leave(d)
            d *= recurse_skip/self.tau
        else:
            d = 1/self.tau * self.leave(phat)
        if recurse:
            d = self.apply_transform(d[::-1], recurse-1-recurse_skip)[::-1]
        newphat += d

        if run_checks:
            assert np.isclose(newphat.dot(self.M), 1), newphat.dot(self.M)
        return newphat
 
    def solve(self,
              recursion_depth,
              no_of_one_degree_steps=3,
              tol=1e-5,
              tmax=30,
              phat0=None,
              iprint=True,
              symmetrize=True,
              transform_kw={}):
        """For a given recursion depth, find the solution that satisfies the given tolerance
        or before max steps is reached.
        
        Parameters
        ----------
        recursion_depth : int
        no_of_one_degree_steps : int, 3
            Number of first order approximations to run to get an approximate starting form.
        tol : float, 1e-3
        tmax : int, 20
        phat0 : ndarray, None
            Starting solution. Otherwise, it is approximated using the tree to depth 1.
        iprint : bool, True
        symmetrize : bool, True
        transform_kw : dict, {}
        
        Returns
        -------
        ndarray
            Solution.
        int
            Error flag.
        """

        newphat = np.ones_like(self.x)
        newphat /= newphat.dot(self.M)
        # sanity check for normalization
        assert np.isclose(newphat.dot(self.M), 1), newphat.dot(self.M)

        # setup while loop
        counter = 0
        phat = np.zeros_like(newphat)
        err = np.linalg.norm(newphat-phat)

        # first get approximate first order solution
        if phat0 is None:
            for i in range(no_of_one_degree_steps):
                phat = newphat
                newphat = self.apply_transform(phat)
                err = np.sqrt(((newphat-phat)**2).dot(self.M))
        # or use given starting approximation
        else:
            assert phat0.size==newphat.size
            newphat = phat0

        while counter<=tmax and err>tol:
            phat = newphat
            newphat = self.apply_transform(phat, recursion_depth, **transform_kw)

            err = np.sqrt(((newphat-phat)**2).dot(self.M))
            counter += 1

        phat = newphat

        # symmetrize
        if symmetrize:
            phat += phat[::-1]
            phat /= 2
        
        # set error flag
        if counter>=tmax and err>tol:
            errflag = 1
        else:
            errflag = 0
        # poor normalization supersedes other error flags
        if not np.isclose(phat.dot(self.M), 1):
            errflag = 2
            warn("Solution is unstable and not normalized.")
        
        if iprint:
            print("Done in %d steps."%counter)
            print("Error of %E."%err)

        self.cache_phat[self.beta] = phat.copy()
        return phat, errflag

    def iter_cond_tree(self,
                       phat,
                       recurse=False):
        """
        Parameters
        ----------
        recurse : int, False

        Returns
        -------
        ndarray
        ndarray
        """

        assert recurse>=0
        
        leaves = [phat]
        sign = [1]

        for i in range(recurse+1):
            newleaves = []
            newsign = []
            for s, leaf in zip(sign, leaves):
                # starting with + and staying
                newleaves.append( (1 - 1/self.tau) * self.stay(leaf) )
                newsign.append(s)

                # starting with + and switching to -
                newleaves.append( 1/self.tau * self.leave(leaf[::-1]) )
                newsign.append(-s)

            leaves = newleaves
            sign = newsign
        
        phat_pos = np.zeros_like(self.x)
        phat_neg = np.zeros_like(self.x)
        parity = 0
        for s, leaf in zip(sign, leaves):
            if parity==0:
                if s==1:
                    phat_pos += leaf
                else:
                    phat_neg += leaf
            else:
                if s==-1:
                    phat_pos += leaf[::-1]
                else:
                    phat_neg += leaf[::-1]
            parity = (parity+1)%2

        return phat_pos, phat_neg

    def apply_transform_cond_external(self,
                                      phat,
                                      sign,
                                      recurse=False,
                                      phat_pos=None,
                                      run_checks=False):
        """Imagine starting a system with equal probability on h0 and -h0. Then, one
        iteration of transformation will maintain probability density with weight
        (1-1/tau) conditional on starting at h0.  At the same time, probability with
        weight 1/tau will flow in from -h0. Recurse.

        One iteration of probability density transformation while keeping track of
        density separately conditional on external field. 

        This is the eigenvalue problem except that we are keeping track of the
        conditional probability distributions separately. The recursion number tells us
        the order of the expansion.

        Parameters
        ----------
        phat_pos : ndarray
        sign : int, +/-1
            Sign indicates if we're starting from  (+) or - (-). This is necessary to
            determine which newphat we add the contribution to since exploit the symmetry
            of this function to call this exact same function for the negative operation.
        recurse : int, False
        run_checks : bool, False

        Returns
        -------
        ndarray
        ndarray
        """
        
        if run_checks:
            assert recurse>=0
            assert sign==1 or sign==-1
        
        newphat = np.zeros_like(self.x)
        if phat_pos is None:
            phat_pos = np.zeros_like(self.x)

        # starting with + and staying
        d = (1 - 1/self.tau) * self.stay(phat)
        if recurse:
            d = self.apply_transform_cond_external(d, sign, recurse-1,
                                                   phat_pos=phat_pos)[0]
        else:  # if we've reached leaves of the recursion tree
            phat_pos += d
        newphat += d

        # probability density flowing in from mirrored config
        d = 1/self.tau * self.leave(phat)
        if recurse:
            d = self.apply_transform_cond_external(d[::-1], -sign, recurse-1,
                                                   phat_pos=phat_pos)[0]
        else:  # if we've reached leaves of the recursion tree
            phat_pos += d[::-1]
        newphat += d[::-1]  # must be inverted by self.leave() is defined rel to h=h0
        
        return newphat, phat_pos

    def solve_external_cond(self,
                            recursion_depth,
                            no_of_one_degree_steps=3,
                            tol=1e-5,
                            tmax=20,
                            phat0=None,
                            iprint=True,
                            cache=True,
                            recursion_check=True):
        """For a given recursion depth, find the solution that satisfies the given tolerance
        or before max steps is reached.
        
        Parameters
        ----------
        recursion_depth : int
        no_of_one_degree_steps : int, 3
            Number of first order approximations to run to get an approximate starting form.
        tol : float, 1e-3
        tmax : int, 20
        phat0 : ndarray, None
            Starting solution. Otherwise, it is approximated using the tree to depth 1.
        iprint : bool, True
        cache : bool, True
            If True, cache results.
        recursion_check : bool, True
            If True, reduce recusion by one to have a check for convergence.  This should
            only be used internally by the function.
        
        Returns
        -------
        ndarray
            Solution.
        int
            Error flag.
        tuple of float
            Eigenvalue solution error (change in functional form after repeated
            iteration).
            Difference from one step smaller recursion.
        tuple of ndarrays
            Separate solutions conditional on postive and neg external field.
        """
        
        newphat = np.ones_like(self.x)
        newphat /= newphat.dot(self.M)

        # setup while loop
        counter = 0
        err = tol * 10

        # first get approximate first order solution
        if phat0 is None:
            for i in range(no_of_one_degree_steps):
                newphat /= newphat.dot(self.M)
                phatpos = self.apply_transform_cond_external(newphat, 1, 0)[1]
                newphat = phatpos
        # or use given starting approximation
        else:
            assert phat0.size==self.x.size
            newphat = phat0

        # solve on recursion_depth-1 to check for error
        if recursion_check and recursion_depth>1:
            phatToCheck = self.solve_external_cond(recursion_depth-1,
                                                   tol=tol,
                                                   tmax=tmax,
                                                   phat0=newphat,
                                                   iprint=False,
                                                   cache=False,
                                                   recursion_check=False)[0]
        else:
            phatToCheck = newphat
        
        # use recursion_depth-1 as starting point
        oldphat = newphat = phatToCheck.copy()
        while counter<=tmax and err>tol:
            newphat /= newphat.dot(self.M)
            newphat = self.apply_transform_cond_external(newphat, 1, recursion_depth)[1]

            err = np.sqrt( ((newphat-oldphat)**2).dot(self.M) )
            oldphat = newphat
            counter += 1

        # set error flag
        if counter>=tmax and err>tol:
            errflag = 1
        else:
            errflag = 0
        if err>tol:
            errflag = 2
        
        if iprint:
            print("Done in %d steps."%counter)
        
        if cache:
            self.cache_phatpos[self.beta] = newphat.copy()

        if recursion_check:
            recursionErr = np.sqrt( ((newphat - phatToCheck)**2).dot(self.M) )
            return  newphat, errflag, (err, recursionErr)
        return  newphat, errflag, (err,)
    
    def increase_depth(self, phat, o_recurse, delta_recurse,
                       external_cond=False, 
                       **solve_kw):
        """Given solution to certain recursion depth, increase recursion depth.
        
        Parameters
        ----------
        phat : ndarray
            Initial guess.
        o_recurse : ndarray
        delta_recurse: ndarray
        external_cond : bool, False
        **solve_kw
        
        Returns
        -------
        ndarray
            Probability density given positive external field.
        ndarray
        int
            Error flag.
        float
            Norm change in density.
        """
        
        # averaged over both pos and neg ext fields
        if not external_cond:
            return self.solve(o_recurse + delta_recurse,
                              phat0=phat, **solve_kw)

        # conditional on positive external week
        output = self.solve_external_cond(o_recurse + delta_recurse,
                                          phat0=phat,
                                          recursion_check=False,
                                          **solve_kw)
        newphatavg, errflag, errs, (newphatpos, newphatneg) = output
        errs = errs[0], np.sqrt( ((newphatavg - phat)**2).dot(self.M) )
        return newphatavg, errflag, errs, (newphatpos, newphatneg)

    def dkl(self, beta_range, recurse,
            n_cpus=None,
            **kwargs):
        """Calculate Kullback-Leibler divergence across a range of beta.

        Parameters
        ----------
        beta_range : ndarray
        recurse : list or int
            If list, specifies recursion depth to use for each beta specified. As a good
            rule of thumb, this should be several times the time scale implied by beta.
        n_cpus : int, None
        **kwargs

        Returns
        -------
        ndarray
            Array of averaged Kullback-Leibler divergence.
        """
        
        if not hasattr(recurse, '__len__'):
            recurse = [recurse]*beta_range.size
        n_cpus = n_cpus or (mp.cpu_count()-1)
        solvedDkl = np.zeros_like(beta_range)
        # dkl as a function of x to be average by density later
        dkl = (pplus(self.x) * ( np.log(pplus(self.x)) - np.log(pplus(self.h0)) ) +
               pminus(self.x) * ( np.log(pminus(self.x)) - np.log(pminus(self.h0)) ))
        #dkl = (pplus(self.h0) * ( np.log(pplus(self.h0)) - np.log(pplus(self.x)) ) +
        #       pminus(self.h0) * ( np.log(pminus(self.h0)) - np.log(pminus(self.x)) ))

        def loop_wrapper(args):
            i, beta, recurse = args
            
            if beta in self.cache_phatpos.keys():
                return self.cache_phatpos[beta]
            
            if 'v' in self.__dict__.keys():
                solver = self.__class__(self.tau, self.h0, beta, self.nBatch,
                                        dx=self.dx, L=self.L, v=self.v, weight=self.weight)
            else:
                solver = self.__class__(self.tau, self.h0, beta, self.nBatch,
                                        dx=self.dx, L=self.L)
            phatpos, errflag, errs = solver.solve_external_cond(recurse, **kwargs)
            if errs[0]>1:
                print("Large iteration error %f for beta = %f."%(errs[0],beta))
            if errs[1]>1:
                print("Large recursion error %f for beta = %f."%(errs[1],beta))
            return phatpos
        
        with threadpool_limits(limits=1, user_api='blas'):
            with mp.Pool(n_cpus) as pool:
                args = zip(range(beta_range.size), beta_range, recurse)
                phatpos = list(pool.map(loop_wrapper, args))
        
        for i, beta in enumerate(beta_range):
            if not beta in self.cache_phatpos.keys():
                self.cache_phatpos[beta] = phatpos[i]
            
            # properly weighted average of DKL
            solvedDkl[i] = ( dkl * phatpos[i] ).dot(self.M)

        return solvedDkl
#end VisionBinary



class StigmergyBinary(VisionBinary):
    """Eigenvalue formulation for binary (h0, -h0) environment state.
    """
    def _init_addon(self, v=1, weight=1):
        self.v = v
        self.weight = weight
        
        dh = self.h0 - self.x[None,:]
        if v>=0:
            stayprob = binary_env_stay_rate(dh, self.tau, v, weight)
            
            # term will be multiplied by 1-1/tau
            self.staycoeff *= stayprob / (1 - 1/self.tau)
            # term will be multiplied by 1/tau
            self.leavecoeff *= (1 - stayprob) * self.tau
        else:
            raise NotImplementedError
#end StigmergyBinary
