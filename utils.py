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

    return 1 / (1 + np.exp(-np.linspace(np.log(tau_mn),                
                                        np.log(tau_mx), n)))       

