# ====================================================================================== #
# Module for learning algorithms comparing passive and active adaptation.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from numba import njit
from multiprocess import cpu_count


@njit
def pplus(h):
    """Probability of + given the field."""
    return .5 + .5*np.tanh(h)

@njit
def pminus(h):
    """Probability of - given the field."""
    return .5 - .5*np.tanh(h)

def entropy(h, base=2):
    """Entropy of biased coin.

    Parameters
    ----------
    h : float

    Returns
    -------
    float
    """

    return (-pplus(h) * np.log(pplus(h)) - pminus(h) * np.log(pminus(h))) / np.log(base)

def linspace_beta(tau_mn, tau_mx, n):
    """Convenience function for creating range of beta such that the memory timescales are
    equally spaced in log space.

    Parameters
    ----------
    tau_mn : float
        Min memory time scale.
    tau_mx : float
        Max memory time scale.
    n : int
        Number of points to space interval with.
        
    Returns
    -------
    ndarray
    """
    
    assert 0<tau_mn<tau_mx
    assert n>1

    return np.exp(-1/np.logspace(np.log10(tau_mn), np.log10(tau_mx), n))

def lobatto_beta(deg):
    """Lobatto-Gauss points excluding beta=1 (corresponding to tau=infty).
    
    Parameters
    ----------
    deg : int

    Returns
    -------
    ndarray
    """

    coX = -np.cos(np.pi*np.arange(deg+1)/deg)[:-1]
    return (coX+1) / 2

@njit
def binary_env_stay_rate(dh, tau, v, weight=1):
    """Probability at each time step that the environment remains fixed and the
    probability that it switches. This is for a stabilizer. For a dissipator, one should
    simply switch the two probabilities.
    
    Parameters
    ----------
    dh : float
    tau : float
    v : float
    weight : float, 1.
        When weight=0, this is equivalent to Vision (passive) agent. When this is 1,
        dissipation or stabilization saturates at dh=0.

    Returns
    -------
    float
        Stay probability. Decay probability is 1 minus this.
    """
    
    assert v>=0

    return 1 - 1/tau + weight * v / tau / (dh*dh + v)

def memory_cost(t, minpos=0):
    """
    Parameters
    ----------
    t : ndarray
    minpos: float, True
        Set min offset so all values are above specified value.
    """

    memCost = -(1/t) * np.log2(1/t) - (1-1/t) * np.log2(1-1/t)
    memCost[np.isnan(memCost)] = 0
    
    # fixed offset
    memCost += minpos

    return memCost

def sensing_cost(t):
    return np.log(t * 2 * np.pi * np.exp(1)) / 2 / np.log(2)
