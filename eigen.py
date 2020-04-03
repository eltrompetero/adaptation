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
        """Transformation of density for external fields that stay the same."""
        return self.trapz_row(phat[None,:] * self.staycoeff) / (1-self.beta)

    def leave(self, phat):
        """Transformation of density for external fields that leave for the other side."""
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
              tol=1e-3,
              tmax=20,
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

    def apply_transform_cond_external(self,
                                      phat,
                                      sign,
                                      recurse=False,
                                      phat_pos=None,
                                      phat_neg=None,
                                      run_checks=False):
        """One iteration of probability density transformation while keeping track of
        density separately conditional on external field. 

        This is the eigenvalue problem except that we are keeping track of the
        conditional probability distributions separately. The recursion number tells us
        the order of the expansion.

        Since h0 and -h0 are symmetric, we actually can compare the two distributions to
        see how well we've converged (errors are +/- of each other), i.e., we can take
        average.
        
        Parameters
        ----------
        phat_pos : ndarray
        phat_neg : ndarray
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

        assert recurse>=0
        assert sign==1 or sign==-1
        
        newphat = np.zeros_like(self.x)
        if phat_pos is None:
            phat_pos = np.zeros_like(self.x)
            phat_neg = np.zeros_like(self.x)

        # starting with  and staying at 
        d = (1 - 1/self.tau) * self.stay(phat)
        if recurse:
            d = self.apply_transform_cond_external(d, sign, recurse-1,
                                                   phat_pos=phat_pos,
                                                   phat_neg=phat_neg)[0]
        else:  # if we've reached leaves of the recursion tree
            if sign==1:
                phat_pos += d
            else:
                phat_neg += d
        newphat += d

        # starting with  and switching to -
        d = 1/self.tau * self.leave(phat)
        if recurse:
            d = self.apply_transform_cond_external(d[::-1], -sign, recurse-1,
                                                   phat_pos=phat_pos,
                                                   phat_neg=phat_neg)[0]
        else:  # if we've reached leaves of the recursion tree
            if sign==-1:
                phat_pos += d[::-1]
            else:
                phat_neg += d[::-1]
        newphat += d
        
        return phat, phat_pos, phat_neg

    def solve_external_cond(self,
                            recursion_depth,
                            no_of_one_degree_steps=3,
                            tol=1e-3,
                            tmax=20,
                            phat0=None,
                            iprint=True):
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
        
        Returns
        -------
        ndarray
            Solution.
        ndarray
            Symmetric solution. Good for checking convergence with recursion depth.
        int
            Error flag.
        """
        
        newphat = np.ones_like(self.x)
        newphat /= newphat.dot(self.M)
        phatavg = np.zeros_like(self.x)
        newphatavg = np.zeros_like(self.x)

        # setup while loop
        counter = 0
        err = tol * 10

        # first get approximate first order solution
        if phat0 is None:
            for i in range(no_of_one_degree_steps):
                newphat, phatpos, phatneg = self.apply_transform_cond_external(newphat, 1, False)
            newphatavg = phatpos + phatneg
            newphatavg /= newphatavg.dot(self.M)
        # or use given starting approximation
        else:
            assert phat0.size==self.x.size
            newphat = phat0
        
        while counter<=tmax and err>tol:
            phatavg = newphatavg
            newphat, phatpos, phatneg = self.apply_transform_cond_external(newphat, 1, recursion_depth)
            newphatavg = phatpos + phatneg
            newphatavg /= newphatavg.dot(self.M)

            err = np.sqrt( ((phatavg-newphatavg)**2).dot(self.M) )
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
            print("Error of %E."%err)

        self.cache_phatpos[self.beta] = phatpos.copy()
        self.cache_phatneg[self.beta] = phatneg.copy()
        return  phatavg, errflag, (phatpos, phatneg)
    
    def increase_depth(self, phat, o_recurse, delta_recurse,
                       external_cond=False):
        """Given solution to certain recursion depth, increase recursion depth.
        
        Parameters
        ----------
        phat : ndarray
        o_recurse : ndarray
        delta_recurse: ndarray
        external_cond : bool, False
        
        Returns
        -------
        ndarray
        ndarray (optional)
        int
            Error flag.
        float
            Norm change in density.
        """
        
        if not external_cond:
            newphat, errflag = self.solve(o_recurse + delta_recurse,
                                          phat0=phat)
            err = np.sqrt( ((phat-newphat)**2).dot(self.M) )
            return newphat, errflag, err

        newphatpos, newphatneg, errflag = self.solve_external_cond(o_recurse + delta_recurse,
                                                                   phat0=phat)
        err = np.sqrt( ((phat-newphatpos)**2).dot(self.M) )
        return newphatpos, newphatneg, errflag, err

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
        dkl = (pplus(self.h0) * ( np.log(pplus(self.h0)) - np.log(pplus(self.x)) ) +
               pminus(self.h0) * ( np.log(pminus(self.h0)) - np.log(pminus(self.x)) ))

        def loop_wrapper(args):
            i, beta, recurse = args
            
            if beta in self.cache_phatneg.keys():
                return ((self.cache_phatneg[beta] + self.cache_phatpos[beta])/2,
                        None,
                        None)

            solver = self.__class__(self.tau, self.h0, beta, self.nBatch, dx=self.dx, L=self.L)
            phatavg, errflag, (phatpos, phatneg) = solver.solve_external_cond(recurse, **kwargs)
            return phatavg, phatpos, phatneg
        
        with threadpool_limits(limits=1, user_api='blas'):
            with mp.Pool(n_cpus) as pool:
                args = zip(range(beta_range.size), beta_range, recurse)
                phatavg, phatpos, phatneg = list(zip(*pool.map(loop_wrapper, args)))
        
        for i, beta in enumerate(beta_range):
            if not beta in self.cache_phatneg.keys():
                self.cache_phatpos[beta] = phatpos[i]
                self.cache_phatneg[beta] = phatneg[i]
            
            # properly weighted average of DKL
            solvedDkl[i] = ( dkl * phatavg[i] ).dot(self.M)

        return solvedDkl
#end VisionBinary



class StigmergyBinary(VisionBinary):
    """Eigenvalue formulation for binary (h0, -h0) environment state.
    """
    def _init_addon(self, v=1, weight=1):
        self.v = v
        
        if v>=0:
            r = 1 + weight * v / (1 - 1/self.tau) / self.tau / ((self.h0-self.x[None,:])**2 + v)
            # term will be multiplied by 1-1/tau
            self.staycoeff *= r
            # term will be multiplied by 1/tau
            self.leavecoeff *= (1 - r * (1 - 1/self.tau)) * self.tau
    #return 1 - 1/tau + weight * v / tau / (dh*dh + v)
#end StigmergyBinary
