# ====================================================================================== #
# Module for learning agent based algorithms comparing passive and active adaptation. We
# simulate learnings agents over the course of some lifetime to sample their behavior.
# 
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
                    #actionRate = self.u * h[t-1]**2 / ((h[t-1]-hhat[t-1])**2 + self.v)
                    actionRate = self.u / ((h[t-1]-hhat[t-1])**2 + self.v)
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
            #actionRate = u * h[t-1]**2 / ((h[t-1]-hhat[t-1])**2 + v)
            actionRate = u / ((h[t-1]-hhat[t-1])**2 + v)
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
