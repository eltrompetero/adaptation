# ====================================================================================== #
# Module for learning algorithms.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from copy import deepcopy



class Vision():
    """Learn the probability distribution of a sequence of coin flips that evolves
    according to Brownian spring (or Ornstein-Ulhenbeck process).
    
    One instance has a fixed noise trajectory that is decided once.
    """
    def __init__(self,
                 nBatch = 20,
                 T = 10_000,
                 dragh = .01,
                 sigmah = 1,
                 beta = .1,
                 rng=None):
        """
        Parameters
        ----------
        nBatch : int, 20
            Number of samples on which the cells learn. Effectively, the time scale for
            cell learning.
        T : int, 10_000
            Number of iterations in terms of cell time (i.e., total number of samples is
            nBatch x T).
        dragh : float, .01
            Coeff on linear force pulling h back to 0, inverse time.
        sigmah : float, 1 
            Std of normal distribution modifying h, specifying environmental noise per
            batch step, or rate fluctuations. This also determines the absolute timescale
            where unit steps in time are when sigmah=1.
        """
        
        assert nBatch>=1 and T>0 and 1>=dragh>=0 and sigmah>0
        self.nBatch = nBatch
        self.dragh = dragh
        self.sigmah = sigmah
        self.T = T
        self.rng = rng or np.random.RandomState()
        self.update_beta(beta)
        h = np.zeros(T)  # underlying signal

        for t in range(1, T):
            # one could explicitly put the dependence on time for the std, but because the time
            # step is fixed, it only modulates sigmah by a constant
            #h[t] = h[t-1] * (1-dragh) + self.rng.normal(scale=sigmah * np.sqrt(2*dragh))

            # discrete version of OH dynamics
            h[t] = h[t-1] * (1-dragh) + self.rng.normal(scale=sigmah)
        
        self.h = h
        
    def update_beta(self, beta):
        assert 0<=beta<=1
        self.beta = beta
        self.alpha = 1 - beta

    def learn(self, beta_range, **kwargs):
        """Learn distribution of environment as it evolves over a range of beta values.

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
                                   self.dragh,
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
def jit_learn_vision(seed, T, dragh, h, nBatch, beta, alpha):
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
                 nBatch = 20,
                 T = 10_000,
                 dragh = .001,
                 sigmah = .1,
                 beta = .1,
                 action_params=(1,1),
                 rng=None):
        """
        Parameters
        ----------
        nBatch : int, 20
            Number of samples on which the cells learn. Effectively, the time scale for
            cell learning.
        T : int, 10_000
            Number of iterations in terms of cell time (i.e., total number of samples is
            nBatch x T).
        dragh : float, .001
            Coeff on linear force pulling h back to 0, inverse time.
        sigmah : float, .1 
            Std of normal distribution modifying h, specifying environmental noise per
            batch step, or the timescale of fluctuations.
        beta : float, .1
            Learning weight for aggregator, effectively setting the time scale for the
            feedback loop.
        action_params : tuple, (1,1)
            The two parameters determining the rate of the action feedback loop (u,v).
            rate = u * ( (h-hhat)^2 + v )
        """
        
        assert nBatch>=1 and T>0 and dragh>=0 and sigmah>0
        assert action_params[0]>=0 and action_params[1]>=0
        self.nBatch = nBatch
        self.beta = beta
        self.alpha = 1 - beta
        self.dragh = dragh
        self.sigmah = sigmah
        self.u = action_params[0]
        self.v = action_params[1]
        self.T = T
        self.rng = rng or np.random.RandomState()

        self.dh = np.zeros(T)  # random noise
        self.dh[1:] = self.rng.normal(scale=sigmah, size=T-1)
        
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
            #h[beta], hhat[beta], H[beta], dkl[beta] = self._learn()
            out = jit_learn_stigmergy(self.rng.randint(2**32-1), self.T, self.u, self.v,
                                      self.dragh, self.dh,
                                      self.nBatch, self.beta, self.alpha)
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

