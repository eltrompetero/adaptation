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



class Vision():
    """Eigenvalue formulation for binary (h0, -h0) environment state.

    For divergence landscape, use .dkl() to calculate divergence as a function of memory
    weighting term beta.
    """
    def __init__(self, tau, h0, beta, nBatch,
                 L=.5, dx=None, **kwargs):
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
        L : float, .5
            Max positive value of lattice. Domain spans [-L, L].
        dx : float, None
            Spacing of points on lattice spreading out from x=0.
        **kwargs
        """
        
        # check input args
        assert tau>=1
        assert 0 <= h0 < L
        assert 0<=beta<=1
        if nBatch<1000:
            warn("If nBatch is too small, Gaussian assumption is broken.")
        if dx is None:
            dx = default_x_spacing(beta, h0, nBatch)
        
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
        if self.x.size>1e5:
            raise Exception("x vector is too large.")
        
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
        assert np.isclose(self.eps_gaussian(x).dot(self.M), 1), (h0, beta, nBatch, L, dx)
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
        tol : float, 1e-5
        tmax : int, 30
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
                                      phat):
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
        phat : ndarray

        Returns
        -------
        ndarray
        ndarray
        """
        
        phat_pos = np.zeros_like(self.x)

        # starting with + and staying
        if self.tau>1:
            d = (1 - 1/self.tau) * self.stay(phat)
            phat_pos += d

        # probability density flowing in from mirrored config
        d = 1/self.tau * self.leave(phat)
        phat_pos += d[::-1]
        
        return phat_pos

    def solve_external_cond(self,
                            tol=1e-5,
                            phat0=None,
                            iprint=True,
                            recursion_check=True,
                            **kwargs):
        """For a given recursion depth, find distribution over agent prediction
        of the external field conditional on the environment bias fixed. Returns
        solution that satisfies the given tolerance or before max steps is
        reached.

        Wrapper for _solve_external_cond().
        
        Parameters
        ----------
        tmax : int, 200
        phat0 : ndarray, None
            Starting solution. Otherwise, it is approximated using the tree to depth 1.
        iprint : bool, True
        tmax : int, 500
        cache : bool, True
            If True, cache results.
        recursion_check : bool, True
            If True, reduce recusion by one to have a check for convergence.
            This should only be used internally by the function.
        
        Returns
        -------
        ndarray
            Solution.
        int
            Error flag.
        tuple of float
            Eigenvalue solution error (change in functional form after repeated
            iteration).
        """
 
        newphat = np.ones_like(self.x)
        newphat /= newphat.dot(self.M)

        # first get approximate first order solution
        if not phat0 is None:
            newphat = phat0

        errs = (0, tol+1)
        newphat, errflag, errs, steps = self._solve_external_cond(newphat, 1,
                                                                  tol=tol,
                                                                  recursion_check=False,
                                                                  **kwargs)

        if iprint:
            print("Done in %d steps."%steps)
        
        return  newphat, errflag, errs

    def _solve_external_cond(self,
                             phat0,
                             recursion_depth,
                             tol=1e-5,
                             mx_tol=1e6,
                             tmax=1000,
                             cache=True,
                             recursion_check=True):
        """For a given recursion depth, find the solution that satisfies the given tolerance
        or before max steps is reached.
        
        Parameters
        ----------
        phat0 : ndarray
            Starting solution. Otherwise, it is approximated using the tree to depth 1.
        recursion_depth : int
        tol : float, 1e-5
        mx_tol : float, 1e6
        tmax : int, 1000
            Max number of iterations to run to convergence. If it has not converged by
            this number, seems like it will not converge for precision reasons (domain
            resolution dx needs to be finer).
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
        
        assert phat0.size==self.x.size
        newphat = phat0
        err = tol * 10

        counter = 0
        oldphat = newphat
        while counter<=tmax and mx_tol>err>tol:
            newphat /= newphat.dot(self.M)
            newphat = self.apply_transform_cond_external(newphat)

            err = np.sqrt( ((newphat-oldphat)**2).dot(self.M) )
            oldphat = newphat
            counter += 1
        counter -= 1

        # set error flag
        if counter>=tmax and err>tol:
            errflag = 1
        else:
            errflag = 0
        if err>tol:
            errflag = 2
        
        return  newphat, errflag, (err,), counter
    
    def dkl(self, beta_range,
            recurse=None,
            n_cpus=None,
            dx_res_factor=None,
            **kwargs):
        """Calculate Kullback-Leibler divergence across a range of beta. Automatically
        adjusts resolution along domain to best calculate DKL landscape.

        Parameters
        ----------
        beta_range : ndarray
        recurse : list or int
            If list, specifies recursion depth to use for each beta specified. As a good
            rule of thumb, this should be several times the time scale implied by beta.
        n_cpus : int, None
        dx_res_factor : float, None
            In case you need more fine-grained control over dx interval.
        **kwargs

        Returns
        -------
        ndarray
            Array of averaged Kullback-Leibler divergence.
        ndarray
            Array of errors for each point. First col is iteration error, second col is
            recursion error.
        """
        
        if not hasattr(recurse, '__len__'):
            recurse = [recurse]*beta_range.size
        n_cpus = n_cpus or (mp.cpu_count()-1)

        def loop_wrapper(args):
            i, beta, recurse = args
            
            # return cached copy
            if beta in self.cache_phatpos.keys():
                return self.cache_phatpos[beta]
            
            # apply resolution factor
            if dx_res_factor is None:
                dx = default_x_spacing(beta, self.h0, self.nBatch)
            else:
                dx = default_x_spacing(beta, self.h0, self.nBatch) / dx_res_factor
            
            # create a copy of the class for this beta
            solver = self.__class__(self.tau, self.h0, beta, self.nBatch,
                                    dx=dx, L=self.L)
            phatpos, errflag, errs = solver.solve_external_cond(**kwargs)
            x = solver.x
            if errs[0]>1:
                print("Large iteration error %f for beta = %f."%(errs[0], beta))
            return phatpos, errs, x
        
        args = zip(range(beta_range.size), beta_range, recurse)
        if n_cpus>1:
            with threadpool_limits(limits=1, user_api='blas'):
                with mp.Pool(n_cpus) as pool:
                    phatpos, errs, x = list(zip(*pool.map(loop_wrapper, args)))
        else:
            phatpos = []
            errs = []
            x = []
            for arg in args:
                output = loop_wrapper(arg)
                phatpos.append(output[0])
                errs.append(output[1])
                x.append(output[2])
        
        return self._dkl_end(beta_range, phatpos, errs, x), np.vstack(errs)

    def _dkl_end(self, beta_range, phatpos, errs, x):
        """Cache results and calculate divergence."""
        solvedDkl = np.zeros_like(beta_range)
        
        # use outputs to calculate typical unfitness
        for i, beta in enumerate(beta_range):
            if not beta in self.cache_phatpos.keys():
                self.cache_phatpos[beta] = phatpos[i], errs[i], x[i]
            
            # properly weighted average of DKL
            # dkl as a function of x
            dkl = (pplus(self.h0) * ( np.log(pplus(self.h0)) - np.log(pplus(x[i])) ) +
                   pminus(self.h0) * ( np.log(pminus(self.h0)) - np.log(pminus(x[i])) ))
            M = np.ones(x[i].size) * (x[i][1]-x[i][0])
            M[0] /= 2
            M[-1] /= 2

            solvedDkl[i] = ( dkl * phatpos[i] ).dot(M)
        return solvedDkl 
Passive = Vision  # alias
#end Vision



class Stigmergy(Vision):
    """Eigenvalue formulation for binary (h0, -h0) environment state. See Vision class for
    more details.

    Note that parameter v in the code is given as v^2 in the paper and that it carries the
    sign (destabilizer vs. stabilizer) in the simulation and not the variable "weight"
    which corresponds to the absolute value of the alpha parameter in the paper.
    """
    def _init_addon(self, v=1, weight=1):
        """For destabilizers v<0 and for stabilizers v>0 (which is more intuitive but the
        negative of the specification in the paper).
        """

        self.v = v
        self.weight = weight
        
        dh = self.h0 - self.x[None,:]
        stayprob = self.binary_env_stay_p(dh)
            
        # term will be multiplied by 1-1/tau
        if self.tau==1:  # for dissipators tau<1 means the environment changes for sure
            self.staycoeff *= stayprob
        else:
            self.staycoeff *= stayprob / (1 - 1/self.tau)
        # term will be multiplied by 1/tau
        self.leavecoeff *= (1 - stayprob) * self.tau

    def apply_transform_cond_external(self, phat):
        """Imagine starting a system with equal probability on h0 and -h0. Then, one
        iteration of transformation will maintain probability density with weight
        (1-1/tau) conditional on starting at h0.  At the same time, probability with
        weight 1/tau will flow in from -h0. Recurse.

        One iteration of probability density transformation while keeping track of density
        separately conditional on external field. 

        This is the eigenvalue problem except that we are keeping track of the conditional
        probability distributions separately. The recursion number tells us the order of
        the expansion.

        Parameters
        ----------
        phat : ndarray

        Returns
        -------
        ndarray
        ndarray
        """
        
        phat_pos = np.zeros_like(self.x)

        # starting with + and staying
        if self.tau==1:
            d = self.stay(phat)
        else:
            d = (1 - 1/self.tau) * self.stay(phat)
        phat_pos += d

        # probability density flowing in from mirrored config
        d = 1/self.tau * self.leave(phat)
        phat_pos += d[::-1]
        
        return phat_pos

    def stability_cost(self, phatpos):
        """Calculate averaged stability cost.

        Parameters
        ----------
        phatpos : ndarray

        Returns
        -------
        ndarray
        """
        
        stay = self.binary_env_stay_p(self.x-self.h0)
        
        term1 = np.log(1/(1-stay) / self.tau) / self.tau 
        term2 = (1-1/self.tau) * np.log((1-1/self.tau) / stay)
        
        d = np.zeros_like(term1)

        notnanix = ~np.isnan(term1)
        d[notnanix] += term1[notnanix]
        notnanix = ~np.isnan(term2)
        d[notnanix] += term2[notnanix]

        return (d * phatpos).dot(self.M)

    def dkl(self, beta_range,
            recurse=None,
            n_cpus=None,
            dx_res_factor=None,
            **kwargs):
        """Calculate Kullback-Leibler divergence across a range of beta.

        Parameters
        ----------
        beta_range : ndarray
        recurse : list or int
            If list, specifies recursion depth to use for each beta specified. As a good
            rule of thumb, this should be several times the time scale implied by beta.
        n_cpus : int, None
        dx_res_factor : float, None
            In case you need more fine-grained control over dx interval.
        **kwargs

        Returns
        -------
        ndarray
            Time-averaged Kullback-Leibler divergence.
        ndarray
            Array of errors for each point. First col is iteration error, second col is
            recursion error.
        ndarray
            Time-averaged stability cost.
        """
        
        if not hasattr(recurse, '__len__'):
            recurse = [recurse]*beta_range.size
        n_cpus = n_cpus or (mp.cpu_count()-1)
        solvedDkl = np.zeros_like(beta_range)
        # dkl as a function of x to be average by density later
        dkl = (pplus(self.h0) * ( np.log(pplus(self.h0)) - np.log(pplus(self.x)) ) +
               pminus(self.h0) * ( np.log(pminus(self.h0)) - np.log(pminus(self.x)) ))

        def loop_wrapper(args):
            i, beta, recurse = args
            
            if beta in self.cache_phatpos.keys():
                return self.cache_phatpos[beta]
            
            if dx_res_factor:
                dx = default_x_spacing(beta, self.h0, self.nBatch, dx_res_factor)
            else:
                dx = default_x_spacing(beta, self.h0, self.nBatch)
            if self.v>0:  # higher density of points for stabilizers
                dx /= 1.25
            solver = self.__class__(self.tau, self.h0, beta, self.nBatch,
                                    dx=dx, L=self.L, v=self.v, weight=self.weight)
            phatpos, errflag, errs = solver.solve_external_cond(**kwargs)
            if errs[0]>1:
                print("Large iteration error %f for beta = %f."%(errs[0],beta))

            scost = solver.stability_cost(phatpos)
            return phatpos, errs, scost, solver.x
        
        args = zip(range(beta_range.size), beta_range, recurse)
        if n_cpus>1:
            with threadpool_limits(limits=1, user_api='blas'):
                with mp.Pool(n_cpus) as pool:
                    phatpos, errs, scost, x = list(zip(*pool.map(loop_wrapper, args)))
        else:
            phatpos = []
            errs = []
            scost = []
            x = []
            for arg in args:
                output = loop_wrapper(arg)
                phatpos.append(output[0])
                errs.append(output[1])
                scost.append(output[2])
                x.append(output[3])

        return self._dkl_end(beta_range, phatpos, errs, scost, x), np.vstack(errs), np.array(scost)

    def _dkl_end(self, beta_range, phatpos, errs, scost, x):
        """Cache results and calculate divergence."""
        solvedDkl = np.zeros_like(beta_range)
        
        # use outputs to calculate typical unfitness
        for i, beta in enumerate(beta_range):
            if not beta in self.cache_phatpos.keys():
                self.cache_phatpos[beta] = phatpos[i], errs[i], scost[i], x[i]
            
            # properly weighted average of DKL
            # dkl as a function of x
            dkl = (pplus(self.h0) * ( np.log(pplus(self.h0)) - np.log(pplus(x[i])) ) +
                   pminus(self.h0) * ( np.log(pminus(self.h0)) - np.log(pminus(x[i])) ))
            M = np.ones(x[i].size) * (x[i][1]-x[i][0])
            M[0] /= 2
            M[-1] /= 2

            solvedDkl[i] = ( dkl * phatpos[i] ).dot(M)
        return solvedDkl 

    def binary_env_stay_p(self, dh):
        """Probability at each time step that the environment remains fixed and the
        probability that it switches. This is for a stabilizer. For a dissipator, one should
        simply switch the two probabilities as is done in this function.
        
        Parameters
        ----------
        dh : float

        Returns
        -------
        float or ndarray
            Stay probability. Decay probability is 1 minus this.
        """
        
        if self.v<0:
            v = -self.v
            weight = -self.weight
        else:
            v = self.v
            weight = self.weight
            
        if isinstance(dh, np.ndarray):
            return (1 - 1/self.tau + weight * v / self.tau / (dh*dh + v)).clip(0)
        else:
            return max(1 - 1/self.tau + weight * v / self.tau / (dh*dh + v), 0)

    def tau_e_moments(self, order, phat, x=None):
        """Calculate moments of environmental change timescale using
        transformation of variables relating tau_e to h.

        Parameters
        ----------
        order : int
            Order of moment to calculate where order=1 would be the mean.
        phat : ndarray
            Probability density of agent bias h.
        x : ndarray, None
            Domain.

        Returns
        -------
        ndarray
            Desired moment of tilde taue.
        """
        
        assert order>=1
        order = float(order)

        # convert sim parameters to those in eq given in pg. 8 of Learning II
        v2 = abs(self.v)
        alpha = -np.sign(self.v) * self.weight  # remember it's the neg of the code
        h0 = self.h0
        if x is None:
            h = self.x
        else:
            h = x
        taue0 = self.tau

        # transform h to tau_e
        # distance term is symmetric about h0 so must consider both branches of quadratic soln
        selectix = h<=h0
        nselectix = h>=h0
        taue = 1 / ( 1/taue0 + alpha * v2 / taue0 / (v2 + (h[selectix]-h0)**2) )
        ntaue = 1 / ( 1/taue0 + alpha * v2 / taue0 / (v2 + (h[nselectix]-h0)**2) )
        termToAvg = taue**order
        ntermToAvg = ntaue**order

        jac = (alpha * taue0 * np.sqrt(v2) / 
               2 / (taue - taue0)**1.5 / np.sqrt(taue0 - (alpha+1) * taue))
        njac = (alpha * taue0 * np.sqrt(v2) / 
                2 / (ntaue - taue0)**1.5 / np.sqrt(taue0 - (alpha+1) * ntaue))
        
        # sanity checks
        #assert np.isclose(np.trapz(phat, h), 1)
        part1 = np.trapz(phat[selectix] * jac, taue) * np.sign(alpha)
        part2 = np.trapz(phat[nselectix] * njac, ntaue) * -np.sign(alpha)
        if np.abs((part1+part2)-1)>1e-2:
            warn(f"Substantial error in transformed distribution p(tau_e)={part1+part2}.")
        
        # integrate over two solutions separately
        #part1 = np.trapz(phat[selectix] * jac * termToAvg, taue) * np.sign(alpha)
        #part2 = np.trapz(phat[nselectix] * njac * ntermToAvg, ntaue) * -np.sign(alpha)
        #return part1 + part2

        return (phat[selectix].dot(termToAvg) + phat[nselectix].dot(ntermToAvg)) / phat.sum()
#end Stigmergy


    
class Landscape():
    def __init__(self, env_prop, agent_prop, beta_range, scale_range, nbatch_range):
        """
        Parameters
        ----------
        env_prop : dict
        agent_prop : dict
        beta_range : ndarray
        scale_range : ndarray
        nbatch_range : ndarray
        """

        self.scaleRange = scale_range
        self.betaRange = beta_range
        self.nBatchRange = nbatch_range

        assert 'tau' in env_prop.keys()
        self.envProp = env_prop

        assert 'weight' in agent_prop.keys()
        assert 'v' in agent_prop.keys()
        self.agentProp = agent_prop

    def run(self, n_cpus=None):
        """Put every combination of scale, nbatch, and beta on a separate thread. Though
        this will be expensive in terms of the time to start up each thread, it will not
        have to wait for long recursions to finish before starting on a new instance of
        Stigmergy.

        Parameters
        ----------
        n_cpus : int, None

        Returns
        -------
        dict
            Arrays of measured unfitness D.
        dict
            Arrays of errors from eigen calculation.
        """

        from itertools import product
        nCpus = mp.cpu_count()-1 if n_cpus is None else n_cpus

        tau = self.envProp['tau']
        weight = self.agentProp['weight']
        v = self.agentProp['v']

        def loop_wrapper(args):
            scale, nbatch, beta = args
            # must be careful to maintain small enough spacing for accurate computation
            # note that we do not go beyond h0=1 for standard sims and following
            # parameters suffice
            solver = Stigmergy(tau, scale, 0, nbatch,
                               L=max(.5,scale*2),
                               weight=weight, v=v)
            return solver.dkl(np.array([beta]), n_cpus=1, iprint=False)

        with threadpool_limits(limits=1, user_api='blas'):
            with mp.Pool(nCpus) as pool:
                args = product(self.scaleRange, self.nBatchRange, self.betaRange)
                dkl_, errs_, scost_ = list(zip(*pool.map(loop_wrapper, args)))

        # group by landscape axes
        n = self.betaRange.size
        dkl_ = [np.concatenate(dkl_[i*n:(i+1)*n]) for i in range(len(dkl_)//n)]
        errs_ = [np.vstack(errs_[i*n:(i+1)*n]) for i in range(len(errs_)//n)]
        scost_ = [np.concatenate(scost_[i*n:(i+1)*n]) for i in range(len(scost_)//n)]

        errs, dkl, scost = {}, {}, {}
        for i, (scale, nbatch) in enumerate(product(self.scaleRange, self.nBatchRange)):
            dkl[(scale, nbatch)] = dkl_[i]
            errs[(scale, nbatch)] = errs_[i]
            scost[(scale, nbatch)] = scost_[i]
        
        self.dkl = dkl
        self.errs = errs
        self.scost = scost
        return dkl, errs, scost
#end Landscape



class AgentLandscape():
    """Show 2D landscape of agent properties with a fixed environment with varying agent
    properties.
    """
    def __init__(self, env_prop, agent_prop, beta_range, nbatch_range):
        """
        Parameters
        ----------
        env_prop : dict
        agent_prop : dict
        beta_range : ndarray
        nbatch_range : ndarray
        """

        self.betaRange = beta_range
        self.nBatchRange = nbatch_range

        assert 'tau' in env_prop.keys()
        assert 'h0' in env_prop.keys()
        self.envProp = env_prop

        assert 'weight' in agent_prop.keys()
        assert 'v' in agent_prop.keys()
        self.agentProp = agent_prop

    def run(self, n_cpus=None):
        """Put every combination of nbatch and beta on a separate thread. 

        Parameters
        ----------
        n_cpus : int, None

        Returns
        -------
        dict
            Arrays of measured unfitness D.
        dict
            Arrays of errors from eigen calculation.
        """

        from itertools import product
        nCpus = mp.cpu_count()-1 if n_cpus is None else n_cpus

        tau = self.envProp['tau']
        scale = self.envProp['h0']
        weight = self.agentProp['weight']
        v = self.agentProp['v']

        def loop_wrapper(args):
            nbatch, beta = args
            # must be careful to maintain small enough spacing for accurate computation
            # note that we do not go beyond h0=1 for standard sims and following
            # parameters suffice
            solver = Stigmergy(tau, scale, 0, nbatch,
                               L=max(.5,scale*2.5),
                               weight=weight, v=v)
            return solver.dkl(np.array([beta]), n_cpus=1, iprint=False)

        with threadpool_limits(limits=1, user_api='blas'):
            with mp.Pool(nCpus) as pool:
                args = product(self.nBatchRange, self.betaRange)
                dkl_, errs_, scost_ = list(zip(*pool.map(loop_wrapper, args)))

        # group by landscape axes
        n = self.betaRange.size
        dkl = np.vstack([np.concatenate(dkl_[i*n:(i+1)*n]) for i in range(len(dkl_)//n)])
        errs = np.vstack([np.vstack(errs_[i*n:(i+1)*n]).ravel() for i in range(len(errs_)//n)])
        scost = np.vstack([np.concatenate(scost_[i*n:(i+1)*n]) for i in range(len(scost_)//n)])
        
        self.dkl = dkl
        self.errs = errs
        self.scost = scost
        return dkl, errs, scost
#end AgentLandscape
