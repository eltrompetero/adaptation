# ====================================================================================== #
# Module for learning agent based algorithms comparing passive and active adaptation. We
# simulate learnings agents over the course of some lifetime to sample their behavior.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from copy import deepcopy
from numba.typed import Dict
from numba import types
from warnings import warn
import multiprocess as mp
from threadpoolctl import threadpool_limits

from .utils import *


class Vision():
    """Learn the probability distribution of a sequence of coin flips that
    evolves according to Brownian spring (or Ornstein-Ulhenbeck process).
    
    One instance has a fixed noise trajectory that is decided once.
    """
    def __init__(self,
                 noise,
                 nBatch=20,
                 T=10_000,
                 beta=.1,
                 rng=None):
        """
        Parameters
        ----------
        noise : dict
            Noise to consider.

            Need to specify 'type' which can be 'OU' for Ornstein-Uhlenbeck or
            'binary'.

            For OU:
                dragh : float, .01
                    Coeff on linear force pulling h back to 0, inverse time.
                sigmah : float, 1 
                    Std of normal distribution modifying h, specifying
                    environmental noise per batch step, or rate fluctuations.
                    This also determines the absolute timescale where unit steps
                    in time are when sigmah=1.
            For binary:
                tau : float
                    Decorrelation time for exponential distribution.
                scale : float
                    Magnitude for binary values of fields.
        nBatch : int, 20
            Number of samples on which the cells learn. Effectively, the time
            scale for cell learning.
        T : int, 10_000
            Number of iterations in terms of cell time (i.e., total number of
            samples is nBatch x T).
        beta : float, 0.1
            Weight on history for learning. This is related to the decorrelation
            time.
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

    def learn(self, beta_range, n_cpus=None, save=True, **kwargs):
        """Learn distribution of environment as it evolves for a range of beta
        values.

        Parameters
        ----------
        beta_range : ndarray
            Learning weight for aggregator, effectively setting the time scale
            for the feedback loop.
        n_cpus : int, None
        save : bool, True
        use_other_dkl : bool, False
            Default is to measure probability distribution of h instead of hhat.
            Note that using hhat as probability distribution can lead to
            infinities from numerical precision errors.

        Returns
        -------
        ndarray
            Averaged Kullback-Leibler divergence per given value of beta. Full
            simulation results are saved in self.dkl.
        """

        if beta_range is None:
            beta_range = [self.beta]
        assert np.unique(beta_range).size==len(beta_range)
        assert all([0<=b<=1 for b in beta_range])

        hhat, H, dkl = {}, {}, {}
        rng = deepcopy(self.rng)
        seed = rng.randint(2**32-1)
        
        if save:
            def loop_wrapper(beta):
                self.update_beta(beta)
                #hhat[beta], H[beta], dkl[beta] = self._learn(**kwargs)
                return jit_learn_vision(seed,
                                        self.T,
                                        self.h,
                                        self.nBatch,
                                        self.beta)
            
            if n_cpus is None or n_cpus > 1:
                with mp.Pool(n_cpus) as pool:
                    hhat_, H_, dkl_ = list(zip(*pool.map(loop_wrapper, beta_range)))
                    hhat = dict(zip(beta_range, hhat_))
                    H = dict(zip(beta_range, H_))
                    dkl = dict(zip(beta_range, dkl_))
            else:
                for beta in beta_range:
                    hhat[beta], H[beta], dkl[beta] = loop_wrapper(beta)
            self.hhat, self.H, self.dkl = hhat, H, dkl
        else:  # don't save full results of calculation
            def loop_wrapper(beta):
                self.update_beta(beta)
                #hhat[beta], H[beta], dkl[beta] = self._learn(**kwargs)
                return jit_learn_vision(seed,
                                        self.T,
                                        self.h,
                                        self.nBatch,
                                        self.beta)[-1]
            
            if n_cpus is None or n_cpus > 1:
                with mp.Pool(n_cpus) as pool:
                    dkl_ = list(pool.map(loop_wrapper, beta_range))
                    dkl = dict(zip(beta_range, dkl_))
            else:
                for beta in beta_range:
                    dkl[beta] = loop_wrapper(beta)
        
        # TODO: need to set a more principled default timescale for averaging
        return np.array([i[1000:].mean() for i in dkl.values()])

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
            if Xmu < (-1+2/self.nBatch):
                Xmu = -1+2/self.nBatch
            if Xmu > (1-2/self.nBatch):
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
def jit_learn_vision(seed, T, h, nBatch, beta):
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
        Xmu = X.mean()
        # finite precision in agent estimates
        if Xmu==-1:
            Xmu += 1e-15
        elif Xmu==1:
            Xmu -= 1e-15

        if t==0:
            hhat[t] = (1-beta) * np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + (1-beta) * np.arctanh(Xmu)
            H[t] = hhat[t]
        
        term1 = pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t])))
        term2 = pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
    
    return hhat, H, dkl



class Stigmergy(Vision):
    """Learn the probability distribution of a sequence of coin flips that
    evolves according to Brownian spring (or Ornstein-Ulhenbeck process) with
    feedback from action loop that depletes information in the environment. 
    
    One instance of this class has a fixed noise trajectory that is established
    upon initialization.
    """
    def __init__(self,
                 noise,
                 nBatch=20,
                 T=10_000,
                 beta=.1,
                 rng=None):
        """
        Parameters
        ----------
        noise : dict
            'ou'
                dragh : float, .001
                    Coeff on linear force pulling h back to 0, inverse time.
                sigmah : float, .1 
                    Std of normal distribution modifying h, specifying
                    environmental noise per batch step, or the timescale of
                    fluctuations.
            'binary'
                tau : float
                scale : float
        nBatch : int, 20
            Number of samples on which the cells learn. Effectively, the time
            scale for cell learning.
        T : int, 10_000
            Number of iterations in terms of cell time (i.e., total number of
            samples is nBatch x T).
        beta : float, .1
            Learning weight for aggregator, effectively setting the time scale
            for the feedback loop.
        action_params : tuple, (1,1)
            The two parameters determining the rate of the action feedback loop
            (u,v).  rate = u * ( (h-hhat)^2 + v )
        """
        
        assert nBatch>=1 and T>0
        self.nBatch = nBatch
        self.beta = beta
        self.alpha = 1 - beta
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
            assert 'v' in noise.keys() and 'scale' in noise.keys()
            assert noise['v']!=0
            if not 'weight' in noise.keys(): noise['weight'] = 1
        elif noise['type']=='binary ant':
            assert 'vd' in noise.keys()
            assert 'vs' in noise.keys()
            assert noise['vd']>=0 and noise['vs']>=0
        else:
            raise Exception("Bad noise type.")
        
    def update_beta(self, beta):
        self.beta = beta
        self.alpha = 1 - beta
   
    def learn(self, beta_range,
              n_cpus=None,
              save=True,
              return_stab_cost=False):
        """Learn distribution of environment as it evolves over a range of beta
        values.

        Parameters
        ----------
        beta_range : ndarray
            Learning weight for aggregator, effectively setting the time scale
            for the feedback loop.
        n_cpus : int, None
        save : bool, True
        return_stab_cost : bool, False
        """

        if beta_range is None:
            beta_range = [self.beta]
        assert np.unique(beta_range).size==len(beta_range)
        assert all([0<=b<=1 for b in beta_range])

        h, hhat, H, dkl, sCost = {}, {}, {}, {}, {}
        rng = deepcopy(self.rng)
        seed = rng.randint(2**32-1)
        
        if save:
            def loop_wrapper(beta):
                self.update_beta(beta)
                
                # slow version for debugging
                #h[beta], hhat[beta], H[beta], dkl[beta] = self._learn()
                # call fast jit version
                if self.noise['type']=='ou':
                    return jit_learn_stigmergy(seed, self.T, self.u, self.v,
                                               self.dragh, self.dh,
                                               self.nBatch, self.beta, self.alpha)
                else:
                    # set up noise for use with njit
                    noise = self.noise.copy()
                    typ = noise['type']
                    del noise['type']
                    jitnoise = Dict.empty(key_type=types.unicode_type,
                                          value_type=types.float64)
                    for k, v in noise.items():
                        jitnoise[k] = v

                    if typ=='binary ant':
                        return jit_learn_stigmergy_binary_noise_ant(seed, self.T,
                                                                    jitnoise,
                                                                    self.nBatch,
                                                                    self.beta)
               
                    else:
                        return jit_learn_stigmergy_binary_noise(seed, self.T,
                                                                jitnoise,
                                                                self.nBatch,
                                                                self.beta)
               
            if n_cpus is None or n_cpus > 1:
                with mp.Pool(n_cpus) as pool:
                    h_, hhat_, H_, dkl_ = list(zip(*pool.map(loop_wrapper, beta_range)))
                # read output
                h = dict(zip(beta_range, h_))
                hhat = dict(zip(beta_range, hhat_))
                H = dict(zip(beta_range, H_))
                dkl = dict(zip(beta_range, dkl_))
            else:
                for beta in beta_range:
                    h[beta], hhat[beta], H[beta], dkl[beta] = loop_wrapper(beta)

            if return_stab_cost:
                # calculate stabilizing cost
                for beta in beta_range:
                    sCost[beta] = stability_cost(self.noise, h[beta], hhat[beta])
                self.sCost = sCost

            self.h, self.hhat, self.H, self.dkl = h, hhat, H, dkl

        else: # don't need to save all the data (and use much memory)
            def loop_wrapper(beta):
                self.update_beta(beta)
                
                # slow version for debugging
                #h[beta], hhat[beta], H[beta], dkl[beta] = self._learn()
                # call fast jit version
                if self.noise['type']=='ou':
                    raise NotImplementedError
                    return jit_learn_stigmergy(seed, self.T, self.u, self.v,
                                               self.dragh, self.dh,
                                               self.nBatch, self.beta, self.alpha)
                else:
                    # set up noise for use with njit
                    noise = self.noise.copy()
                    typ = noise['type']
                    del noise['type']
                    jitnoise = Dict.empty(key_type=types.unicode_type,
                                          value_type=types.float64)
                    for k, v in noise.items():
                        jitnoise[k] = v
                    
                    if typ=='binary ant':
                        h, hhat, H, dkl = jit_learn_stigmergy_binary_noise_ant(seed, self.T,
                                                                               jitnoise,
                                                                               self.nBatch,
                                                                               self.beta)
                        return dkl
                    else:
                        h, hhat, H, dkl = jit_learn_stigmergy_binary_noise(seed, self.T,
                                                                           jitnoise,
                                                                           self.nBatch,
                                                                           self.beta)
                        if return_stab_cost:
                            sCost = stability_cost(self.noise, h, hhat)
                            return dkl, sCost
                        return dkl
               
            if n_cpus is None or n_cpus > 1:
                with mp.Pool(n_cpus) as pool:
                    if return_stab_cost:
                        dkl_, sCost_ = list(zip(*pool.map(loop_wrapper, beta_range)))
                    else:
                        dkl_ = list(pool.map(loop_wrapper, beta_range))
                # read output
                dkl = dict(zip(beta_range, dkl_))
                if return_stab_cost:
                    sCost = dict(zip(beta_range, sCost_))
            else:
                if return_stab_cost:
                    for beta in beta_range:
                        dkl[beta], sCost[beta] = loop_wrapper(beta)
                else:
                    for beta in beta_range:
                        dkl[beta] = loop_wrapper(beta)
         
        # TODO: need to set better default timeescale to not average over
        if return_stab_cost:
            return (np.array([i[1000:].mean() for i in dkl.values()]),
                    np.array([i[1000:].mean() for i in sCost.values()]))
        return np.array([i[1000:].mean() for i in dkl.values()])

    def _learn(self):
        """Learn distribution of environment as it evolves and is learned
        through stigmergetic coupling.  Signal in environment decays with rate
        that grows with strength of signal and with similarity between learned
        signal and true signal.
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
        Xmu = X.mean()
        # finite precision in agent estimates
        if Xmu==-1:
            Xmu += 1e-15
        elif Xmu==1:
            Xmu -= 1e-15

        if t==0:
            hhat[t] = np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + alpha * np.arctanh(Xmu)
            H[t] = hhat[t]
        
        term1 = pplus(hhat[t]) * (np.log(pplus(hhat[t])) - np.log(pplus(h[t])))
        term2 = pminus(hhat[t]) * (np.log(pminus(hhat[t])) - np.log(pminus(h[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
    
    return h, hhat, H, dkl

@njit
def jit_learn_stigmergy_binary_noise(seed, T, noise, nBatch, beta):
    if seed!=-1:
        np.random.seed(seed)
    
    h = np.zeros(T)
    hhat = np.zeros(T)
    H = np.zeros(T)
    dkl = np.zeros(T)

    h[0] = noise['scale']
    v = noise['v']
    tau = noise['tau']
    weight = noise['weight']

    for t in range(T):
        if t>0:
            # action rate determines env switching time
            if v<0:  # dissipator
                decay = 1 - binary_env_stay_rate(h[t-1]-hhat[t-1], tau, -v, -weight)
            else:  # stabilizer
                decay = 1 - binary_env_stay_rate(h[t-1]-hhat[t-1], tau, v, weight)

            if np.random.rand() <= decay:
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
        Xmu = X.mean()
        # finite precision in agent estimates
        if Xmu==-1:
            Xmu += 1e-15
        elif Xmu==1:
            Xmu -= 1e-15

        if t==0:
            hhat[t] = (1-beta) * np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + (1-beta) * np.arctanh(Xmu)
            H[t] = hhat[t]

        term1 = pplus(h[t]) * (np.log(pplus(h[t])) - np.log(pplus(hhat[t])))
        term2 = pminus(h[t]) * (np.log(pminus(h[t])) - np.log(pminus(hhat[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
 
    return h, hhat, H, dkl

@njit
def jit_learn_stigmergy_binary_noise_ant(seed, T, noise, nBatch, beta):
    if seed!=-1:
        np.random.seed(seed)
    
    h = np.zeros(T)
    hhat = np.zeros(T)
    H = np.zeros(T)
    dkl = np.zeros(T)

    h[0] = noise['scale']
    vd = noise['vd']
    vs = noise['vs']
    width = noise['width']
    weight = noise['weight']
    tau = noise['tau']

    for t in range(T):
        if t>0:
            # transition from stabilizer far from h0 to dissipator close to h0
            sdecay = 1 - binary_env_stay_rate(h[t-1]-hhat[t-1], tau, vs, weight)
            ddecay = 1 - binary_env_stay_rate(h[t-1]-hhat[t-1], tau, vd, -weight)
            crossfade = np.exp( -(h[t-1]-hhat[t-1])**2 / 2 / width**2 )
            decay = (1-crossfade) * sdecay + crossfade * ddecay

            if np.random.rand() < decay:
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
        Xmu = X.mean()
        # finite precision in agent estimates
        if Xmu==-1:
            Xmu += 1e-15
        elif Xmu==1:
            Xmu -= 1e-15

        if t==0:
            hhat[t] = (1-beta) * np.arctanh(Xmu)
            H[t] = hhat[t]
        else:
            # weighted combination of memory and current measurement
            hhat[t] = beta * H[t-1] + (1-beta) * np.arctanh(Xmu)
            H[t] = hhat[t]

        term1 = pplus(hhat[t]) * (np.log(pplus(hhat[t])) - np.log(pplus(h[t])))
        term2 = pminus(hhat[t]) * (np.log(pminus(hhat[t])) - np.log(pminus(h[t])))
        if not np.isnan(term1):
            dkl[t] += term1
        if not np.isnan(term2):
            dkl[t] += term2
 
    return h, hhat, H, dkl



# =============== #
# Other functions #
# =============== #
def stability_cost(noise, h, hhat):
    """Calculate stabilizing cost.

    Parameters
    ----------
    noise : dict
        'v', 'weight', 'tau'
    h : ndarray
    hhat : ndarray

    Returns
    -------
    ndarray
    """

    v, weight, tau = noise['v'], noise['weight'], noise['tau']
    assert v>=0

    stay = binary_env_stay_rate(h-hhat, tau, v, weight)  # decay prob
    sCost = np.log( 1/(1-stay) / tau) / tau + (1-1/tau) * np.log((1-1/tau) / stay)
    #sCost = (1-stay) * np.log( (1-stay) * tau) + stay * np.log(stay / (1-1/tau))
    return sCost
