# ====================================================================================== #
# Module for learning algorithms comparing passive and active adaptation.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from copy import deepcopy
from numba.typed import Dict
from numba import types
from warnings import warn
import multiprocess as mp
from threadpoolctl import threadpool_limits



class Vision():
    """Learn the probability distribution of a sequence of coin flips that evolves
    according to Brownian spring (or Ornstein-Ulhenbeck process).
    
    One instance has a fixed noise trajectory that is decided once.
    """
    def __init__(self,
                 noise,
                 nBatch = 20,
                 T = 10_000,
                 beta = .1,
                 rng=None):
        """
        Parameters
        ----------
        noise : dict
            Noise to consider.
            Need to specify 'type' which can be 'OU' for Ornstein-Uhlenbeck or 'binary'.
            For OU:
                dragh : float, .01
                    Coeff on linear force pulling h back to 0, inverse time.
                sigmah : float, 1 
                    Std of normal distribution modifying h, specifying environmental noise per
                    batch step, or rate fluctuations. This also determines the absolute timescale
                    where unit steps in time are when sigmah=1.
            For binary:
                tau : float
                    Decorrelation time for exponential distribution.
                scale : float
                    Magnitude for binary values of fields.
        nBatch : int, 20
            Number of samples on which the cells learn. Effectively, the time scale for
            cell learning.
        T : int, 10_000
            Number of iterations in terms of cell time (i.e., total number of samples is
            nBatch x T).
        beta : float, 0.1
            Weight on history for learning. This is related to the decorrelation time.
        rng : np.random.RandomState, None
        """
        
        assert nBatch>=1 and T>0
        self.nBatch = nBatch
        self.T = T
        self.rng = rng or np.random.RandomState()
        self.update_beta(beta)
        h = np.zeros(T)  # underlying signal
        
        # generate time trajectory of fields
        if noise['type']=='OU' or noise['type']=='ou':
            dragh = noise.get('dragh', .01)
            sigmah = noise.get('sigmah', 1)
            assert 1>=dragh>=0 and sigmah>0

            for t in range(1, T):
                # one could explicitly put the dependence on time for the std, but because the time
                # step is fixed, it only modulates sigmah by a constant
                #h[t] = h[t-1] * (1-dragh) + self.rng.normal(scale=sigmah * np.sqrt(2*dragh))

                # discrete version of OH dynamics
                h[t] = h[t-1] * (1-dragh) + self.rng.normal(scale=sigmah)

            self.dragh = dragh
            self.sigmah = sigmah

        elif noise['type']=='binary':
            tau = noise['tau']  # decorrelation time
            hscale = noise['scale']  # magnitude of biasing fields

            h[0] = hscale
            for t in range(1, T):
                if self.rng.rand()<(1/tau):
                    h[t] = -h[t-1]
                else:
                    h[t] = h[t-1]
            #t = 0
            #while t<T:
            #    dt = int(self.rng.exponential(tau)) + 1
            #    hscale *= -1
            #    h[t:t+dt] = hscale
            #    t += dt
            self.hscale = hscale
        else:
            raise Exception("Unrecognized noise type.")
        
        self.h = h
        
    def update_beta(self, beta):
        assert 0<=beta<=1
        self.beta = beta
        self.alpha = 1 - beta

    def learn(self, beta_range, **kwargs):
        """Learn distribution of environment as it evolves for a range of beta values.

        Parameters
        ----------
        beta_range : ndarray
            Learning weight for aggregator, effectively setting the time scale for the
            feedback loop.
        use_other_dkl : bool, False
            Default is to measure probability distribution of h instead of hhat. Note that
            using hhat as probability distribution can lead to infinities from numerical
            precision errors.
        """

        if beta_range is None:
            beta_range = [self.beta]
        assert np.unique(beta_range).size==len(beta_range)
        assert all([0<=b<=1 for b in beta_range])

        hhat, H, dkl = {}, {}, {}
        rng = deepcopy(self.rng)
        
        for beta in beta_range:
            self.update_beta(beta)
            # reset rng every loop so that random trajectory remains the same
            self.rng = deepcopy(rng)
            #hhat[beta], H[beta], dkl[beta] = self._learn(**kwargs)
            out = jit_learn_vision(self.rng.randint(2**32-1),
                                   self.T,
                                   self.h,
                                   self.nBatch,
                                   self.beta,
                                   self.alpha)
            hhat[beta], H[beta], dkl[beta] = out

        self.hhat, self.H, self.dkl = hhat, H, dkl

    def _learn(self, use_other_dkl=False):
        """Learn distribution of environment as it evolves.
        Keep track of measurement error.
        """
        
        hhat = np.zeros(self.T)
        H = np.zeros(self.T)
        dkl = np.zeros(self.T)
        
        for t in range(self.T):    
            # simulate coin flip using environmental field
            p = [pminus(self.h[t]), pplus(self.h[t])]
            X = self.rng.choice([-1,1], size=self.nBatch, p=p)

            # regularize estimate to avoid infinities (set to the point right where rounding error to 0
            # occurs)
            Xmu = X.mean()
            # floating point precision limits
            #if Xmu<(-1+1e-16):
            #    Xmu = -1+1e-16
            #if Xmu>(1-1e-16):
            #    Xmu = 1-1e-16
            # Laplace counting limits
            if Xmu<(-1+2/self.nBatch):
                Xmu = -1+2/self.nBatch
            if Xmu>(1-2/self.nBatch):
                Xmu = 1-2/self.nBatch

            if t==0:
                hhat[t] = np.arctanh(Xmu)
                H[t] = hhat[t]
            else:
                # weighted combination of memory and current measurement
                hhat[t] = self.beta * H[t-1] + self.alpha * np.arctanh(Xmu)
                H[t] = hhat[t]
            
            if use_other_dkl:
                dkl[t] = np.nansum([pplus(hhat[t]) * (np.log(pplus(hhat[t])) - np.log(pplus(self.h[t]))),
                                    pminus(hhat[t]) * (np.log(pminus(hhat[t])) - np.log(pminus(self.h[t])))])
            else:
                dkl[t] = np.nansum([pplus(self.h[t]) * (np.log(pplus(self.h[t])) - np.log(pplus(hhat[t]))),
                                    pminus(self.h[t]) * (np.log(pminus(self.h[t])) - np.log(pminus(hhat[t])))])

        return hhat, H, dkl
#end Vision

@njit
def jit_learn_vision(seed, T, h, nBatch, beta, alpha):
    """Jit version of Vision._learn().
    """

    if seed!=-1:
        np.random.seed(seed)
    
    hhat = np.zeros(T)
    H = np.zeros(T)
    dkl = np.zeros(T)

    for t in range(T):
        # simulate coin flip using environmental field
        p = pplus(h[t])
        X = np.zeros(nBatch)
        for i in range(nBatch):
            if np.random.rand()<p:
                X[i] = 1
            else:
                X[i] = -1

        # regularize estimate to avoid infinities. makes sure that Dkl never diverges
        Xmu = X.mean()
        # floating point precision limits
        if Xmu<(-1+1e-15):
            Xmu = -1+1e-15
        if Xmu>(1-1e-15):
            Xmu = 1-1e-15
        #if Xmu<(-1+2/nBatch):
        #    Xmu = -1+2/nBatch
        #if Xmu>(1-2/nBatch):
        #    Xmu = 1-2/nBatch

        if t==0:
            hhat[t] = np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + alpha * np.arctanh(Xmu)
            H[t] = hhat[t]
        
        term1 = pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t])))
        term2 = pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
    
    return hhat, H, dkl



class Stigmergy(Vision):
    """Learn the probability distribution of a sequence of coin flips that evolves
    according to Brownian spring (or Ornstein-Ulhenbeck process) with feedback from action
    loop that depletes information in the environment. 
    
    One instance of this class has a fixed noise trajectory that is established upon
    initialization.
    """
    def __init__(self,
                 noise,
                 nBatch = 20,
                 T = 10_000,
                 beta = .1,
                 action_params=(1,1),
                 rng=None):
        """
        Parameters
        ----------
        noise : dict
            'ou'
                dragh : float, .001
                    Coeff on linear force pulling h back to 0, inverse time.
                sigmah : float, .1 
                    Std of normal distribution modifying h, specifying environmental noise per
                    batch step, or the timescale of fluctuations.
            'binary'
                tau : float
                scale : float
        nBatch : int, 20
            Number of samples on which the cells learn. Effectively, the time scale for
            cell learning.
        T : int, 10_000
            Number of iterations in terms of cell time (i.e., total number of samples is
            nBatch x T).
        beta : float, .1
            Learning weight for aggregator, effectively setting the time scale for the
            feedback loop.
        action_params : tuple, (1,1)
            The two parameters determining the rate of the action feedback loop (u,v).
            rate = u * ( (h-hhat)^2 + v )
        """
        
        assert nBatch>=1 and T>0
        assert action_params[1]>=0
        self.nBatch = nBatch
        self.beta = beta
        self.alpha = 1 - beta
        self.u = action_params[0]
        self.v = action_params[1]
        self.T = T
        self.rng = rng or np.random.RandomState()
        self.noise = noise

        if noise['type']=='ou':
            sigmah = noise['sigmah']
            dragh = noise['dragh']
            assert dragh>=0 and sigmah>0
            self.sigmah = sigmah
            self.dragh = dragh

            self.dh = np.zeros(T)  # random noise
            self.dh[1:] = self.rng.normal(scale=sigmah, size=T-1)
        elif noise['type']=='binary':
            pass
        else:
            raise Exception("Bad noise type.")
        
    def update_beta(self, beta):
        self.beta = beta
        self.alpha = 1 - beta
   
    def learn(self, beta_range):
        """Learn distribution of environment as it evolves over a range of beta values.

        Parameters
        ----------
        beta_range : ndarray
            Learning weight for aggregator, effectively setting the time scale for the
            feedback loop.
        """

        if beta_range is None:
            beta_range = [self.beta]
        assert np.unique(beta_range).size==len(beta_range)
        assert all([0<=b<=1 for b in beta_range])

        h, hhat, H, dkl = {}, {}, {}, {}
        rng = deepcopy(self.rng)
        
        for beta in beta_range:
            self.update_beta(beta)
            # reset rng every loop so that random trajectory remains the same
            self.rng = deepcopy(rng)
            
            # slow version for debugging
            #h[beta], hhat[beta], H[beta], dkl[beta] = self._learn()
            # call fast jit version
            if self.noise['type']=='ou':
                out = jit_learn_stigmergy(self.rng.randint(2**32-1), self.T, self.u, self.v,
                                          self.dragh, self.dh,
                                          self.nBatch, self.beta, self.alpha)
            else:
                # set up noise for use with njit
                noise = self.noise.copy()
                del noise['type']
                jitnoise = Dict.empty(key_type=types.unicode_type,
                                      value_type=types.float64)
                for k, v in noise.items():
                    jitnoise[k] = v

                out = jit_learn_stigmergy_binary_noise(self.rng.randint(2**32-1), self.T, self.u, self.v,
                                                       jitnoise, self.nBatch, self.beta, self.alpha)
            
            # read output
            h[beta], hhat[beta], H[beta], dkl[beta] = out

        self.h, self.hhat, self.H, self.dkl = h, hhat, H, dkl

    def _learn(self):
        """Learn distribution of environment as it evolves and is learned through
        stigmergetic coupling.  Signal in environment decays with rate that grows with
        strength of signal and with similarity between learned signal and true
        signal.
        """
        
        h = np.zeros(self.T)
        hhat = np.zeros(self.T)
        H = np.zeros(self.T)
        dkl = np.zeros(self.T)
        
        if self.noise['type']=='ou':
            for t in range(self.T):
                if t>0:
                    # action rate grows with the strength of the signal and with similarity
                    # between the two signals
                    actionRate = self.u * h[t-1]**2 / ((h[t-1]-hhat[t-1])**2 + self.v)
                    h[t] = h[t-1] * np.exp(-(self.dragh + actionRate)) + self.dh[t]

                # simulate coin flip using environmental field
                p = [pminus(h[t]), pplus(h[t])]
                X = self.rng.choice([-1,1], size=self.nBatch, p=p)

                # regularize estimate to avoid infinities. makes sure that Dkl never diverges
                Xmu = X.mean()
                if Xmu<(-1+2/self.nBatch):
                    Xmu = -1+2/self.nBatch
                if Xmu>(1-2/self.nBatch):
                    Xmu = 1-2/self.nBatch

                if t==0:
                    hhat[t] = np.arctanh(Xmu)
                    H[t] = hhat[t]
                else:
                    # weighted combination of memory and current measurement
                    hhat[t] = self.beta * H[t-1] + self.alpha * np.arctanh(Xmu)
                    H[t] = hhat[t]

                dkl[t] = np.nansum([pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t]))),
                                    pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))])

        else:
            h[0] = self.noise['scale']

            for t in range(self.T):
                if t>0:
                    # action rate grows with the strength of the signal and with similarity
                    # between the two signals
                    actionRate = self.u * h[t-1]**2 / ((h[t-1]-hhat[t-1])**2 + self.v)
                    if self.rng.rand() < (1 / self.noise['tau'] + actionRate):
                        h[t] = -h[t-1]
                    else:
                        h[t] = h[t-1]

                # simulate coin flip using environmental field
                p = [pminus(h[t]), pplus(h[t])]
                X = self.rng.choice([-1,1], size=self.nBatch, p=p)

                # regularize estimate to avoid infinities. makes sure that Dkl never diverges
                Xmu = X.mean()
                if Xmu<(-1+2/self.nBatch):
                    Xmu = -1+2/self.nBatch
                if Xmu>(1-2/self.nBatch):
                    Xmu = 1-2/self.nBatch

                if t==0:
                    hhat[t] = np.arctanh(Xmu)
                    H[t] = hhat[t]
                else:
                    # weighted combination of memory and current measurement
                    hhat[t] = self.beta * H[t-1] + self.alpha * np.arctanh(Xmu)
                    H[t] = hhat[t]

                dkl[t] = np.nansum([pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t]))),
                                    pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))])
        return h, hhat, H, dkl
#end Stigmergy

@njit
def jit_learn_stigmergy(seed, T, u, v, dragh, dh, nBatch, beta, alpha):
    """Jit version of Stigmergy._learn().
    """

    if seed!=-1:
        np.random.seed(seed)
    
    h = np.zeros(T)
    hhat = np.zeros(T)
    H = np.zeros(T)
    dkl = np.zeros(T)

    for t in range(T):
        if t>0:
            # action rate grows with the strength of the signal and with similarity
            # between the two signals
            actionRate = u * h[t-1]**2 / ((h[t-1]-hhat[t-1])**2 + v)
            h[t] = h[t-1] * np.exp(-(dragh + actionRate)) + dh[t]

        # simulate coin flip using environmental field
        p = pplus(h[t])
        X = np.zeros(nBatch)
        for i in range(nBatch):
            if np.random.rand()<p:
                X[i] = 1
            else:
                X[i] = -1

        # regularize estimate to avoid infinities. makes sure that Dkl never diverges
        Xmu = X.mean()
        if Xmu<(-1+2/nBatch):
            Xmu = -1+2/nBatch
        if Xmu>(1-2/nBatch):
            Xmu = 1-2/nBatch

        if t==0:
            hhat[t] = np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + alpha * np.arctanh(Xmu)
            H[t] = hhat[t]
        
        term1 = pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t])))
        term2 = pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
    
    return h, hhat, H, dkl

@njit
def jit_learn_stigmergy_binary_noise(seed, T, u, v, noise, nBatch, beta, alpha):
    if seed!=-1:
        np.random.seed(seed)
    
    h = np.zeros(T)
    hhat = np.zeros(T)
    H = np.zeros(T)
    dkl = np.zeros(T)

    h[0] = noise['scale']

    for t in range(T):
        if t>0:
            # action rate grows with the strength of the signal and with similarity
            # between the two signals
            actionRate = u * h[t-1]**2 / ((h[t-1]-hhat[t-1])**2 + v)
            if np.random.rand() < (1 / noise['tau'] + actionRate):
                h[t] = -h[t-1]
            else:
                h[t] = h[t-1]

        # simulate coin flip using environmental field
        p = pplus(h[t])
        X = np.zeros(nBatch)
        for i in range(nBatch):
            if np.random.rand()<p:
                X[i] = 1
            else:
                X[i] = -1

        # regularize estimate to avoid infinities. makes sure that Dkl never diverges
        Xmu = X.mean()
        if Xmu<(-1+2/nBatch):
            Xmu = -1+2/nBatch
        if Xmu>(1-2/nBatch):
            Xmu = 1-2/nBatch

        if t==0:
            hhat[t] = np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + alpha * np.arctanh(Xmu)
            H[t] = hhat[t]

        term1 = pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t])))
        term2 = pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
 
    return h, hhat, H, dkl



class TreeEigensolverBinary():
    """Eigenvalue formulation for binary (h0, -h0) environment state.
    """
    def __init__(self, tau, h0, beta, nBatch,
                 L=1, dx=1e-3):
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
        eps = (np.tanh(h0) - np.tanh(dhhat)) / 2  # error in measured field
        self.staycoeff = self.eps_gaussian(eps) * .5 * (1-np.tanh(dhhat)**2)  # term that goes into integral
        eps = (-np.tanh(h0) - np.tanh(dhhat)) / 2  # error in measured field
        self.leavecoeff = self.eps_gaussian(eps) * .5 * (1-np.tanh(dhhat)**2)  # term that goes into integral
        
        # basic precision checks (for one's sanity)
        assert np.isclose(self.eps_gaussian(x).dot(self.M), 1)
        assert np.isclose(np.linalg.norm(x+x[::-1]), 0)
        
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

        if phat_pos is None:
            phat_pos = np.zeros_like(self.x)
            phat_neg = np.zeros_like(self.x)

        # starting with  and staying at 
        d = (1 - 1/self.tau) * self.stay(phat)
        if recurse:
            self.apply_transform_cond_external(d, sign, recurse-1,
                                               phat_pos=phat_pos,
                                               phat_neg=phat_neg)
        else:  # if we've reached leaves of the recursion tree
            if sign==1:
                phat_pos += d
            else:
                phat_neg += d

        # starting with  and switching to -
        d = 1/self.tau * self.leave(phat)
        if recurse:
            self.apply_transform_cond_external(d[::-1], -sign, recurse-1,
                                               phat_pos=phat_pos,
                                               phat_neg=phat_neg)
        else:  # if we've reached leaves of the recursion tree
            if sign==-1:
                phat_pos += d[::-1]
            else:
                phat_neg += d[::-1]

        return phat_pos, phat_neg

    def solve_external_cond(self,
                            recursion_depth,
                            no_of_one_degree_steps=3,
                            tol=1e-3,
                            tmax=20,
                            phat0=None,
                            iprint=True,
                            symmetrize=False):
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
        symmetrize : bool, False
        
        Returns
        -------
        ndarray
            Solution.
        ndarray
            Symmetric solution. Good for checking convergence with recursion depth.
        int
            Error flag.
        """
        
        newphatpos = np.ones_like(self.x)
        newphatpos /= newphatpos.dot(self.M)

        # setup while loop
        counter = 0
        phatpos = np.zeros_like(newphatpos)
        err = np.linalg.norm(newphatpos-phatpos)

        # first get approximate first order solution
        if phat0 is None:
            for i in range(no_of_one_degree_steps):
                phatpos = newphatpos
                newphatpos, newphatneg = self.apply_transform_cond_external(phatpos, 1, False)
                err = np.sqrt(((newphatpos-phatpos)**2).dot(self.M))
        # or use given starting approximation
        else:
            assert phat0.size==newphatpos.size
            newphatpos = phat0

        while counter<=tmax and err>tol:
            phatpos = newphatpos
            newphatpos, newphatneg = self.apply_transform_cond_external(phatpos, 1, recursion_depth)
            newphatpos /= newphatpos.dot(self.M)

            err = np.sqrt( ((phatpos-newphatpos)**2).dot(self.M) )
            counter += 1
        phatneg = newphatneg / newphatneg.dot(self.M)
        phatpos = newphatpos

        # symmetrize
        if symmetrize:
            phatpos += phatpos[::-1]
            phatpos /= 2
            phatneg += phatneg[::-1]
            phatneg /= 2

        # set error flag
        if counter>=tmax and err>tol:
            errflag = 1
        else:
            errflag = 0
        if np.sqrt( ((phatpos-phatneg)**2).dot(self.M) )>tol:
            errflag = 2
        
        if iprint:
            print("Done in %d steps."%counter)
            print("Error of %E."%err)

        self.cache_phatpos[self.beta] = phatpos.copy()
        self.cache_phatneg[self.beta] = phatneg.copy()
        return  phatpos, phatneg, errflag
    
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
        recurse : int
        n_cpus : int, None
        **kwargs

        Returns
        -------
        ndarray
            Array of averaged Kullback-Leibler divergence.
        """
        
        n_cpus = n_cpus or (mp.cpu_count()-1)
        solvedDkl = np.zeros_like(beta_range)
        dkl = (pplus(self.h0) * ( np.log(pplus(self.h0)) - np.log(pplus(self.x)) ) +
               pminus(self.h0) * ( np.log(pminus(self.h0)) - np.log(pminus(self.x)) ))

        def loop_wrapper(args, recurse=recurse):
            i, beta = args
            
            if beta in self.cache_phatneg.keys():
                return ((self.cache_phatneg[beta] + self.cache_phatpos[beta])/2,
                        None,
                        None)

            solver = TreeEigensolverBinary(self.tau, self.h0, beta, self.nBatch)
            phatpos, phatneg, errflag = solver.solve_external_cond(recurse, **kwargs)
            phat = (phatpos+phatneg)/2
            return phat, phatpos, phatneg
        
        with threadpool_limits(limits=1, user_api='blas'):
            with mp.Pool(n_cpus) as pool:
                phat, phatpos, phatneg = list(zip(*pool.map(loop_wrapper, enumerate(beta_range))))
        
        for i, beta in enumerate(beta_range):
            if not beta in self.cache_phatneg.keys():
                self.cache_phatpos[beta] = phatpos[i]
                self.cache_phatneg[beta] = phatneg[i]
            
            # properly weighted average of DKL
            solvedDkl[i] = ( dkl * phat[i] ).dot(self.M)

        return solvedDkl
#end TreeEigensolverBinary
