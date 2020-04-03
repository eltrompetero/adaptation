# ====================================================================================== #
# Module for learning algorithms comparing passive and active adaptation.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from numba import njit


@njit
def pplus(h):
    """Probability of + given the field."""
    return .5 + .5*np.tanh(h)

@njit
def pminus(h):
    """Probability of - given the field."""
    return .5 - .5*np.tanh(h)

def entropy(h):
    """Entropy of biased coin.

    Parameters
    ----------
    h : float

    Returns
    -------
    float
    """

    return -pplus(h) * np.log2(pplus(h)) - pminus(h) * np.log2(pminus(h))

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
