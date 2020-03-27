# ====================================================================================== #
# Module for learning algorithms.
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

