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
        Memory timescale.
    minpos: float, True
        Set min offset so all values are above specified value.

    Returns
    -------
    np.ndarray
        In units of bits.
    """

    memCost = -(1/t) * np.log2(1/t) - (1-1/t) * np.log2(1-1/t)
    memCost[np.isnan(memCost)] = 0
    
    # fixed offset
    memCost += minpos

    return memCost

def sensing_cost(t, h):
    """
    Parameters
    ----------
    t : ndarray
        Sensory cell timescale.
    h : float
        Field.
    
    Returns
    -------
    ndarray
        Cost in bits.
    """

    cost = -np.log(8 * np.pi * pplus(h) * pminus(h) / t) / 2 + 2 * np.log(1/np.cosh(h))
    return cost / np.log(2)

def interpolate(beta_range, dkl, h0,
                errs=None,
                tol=1e-4,
                deg=None,
                include_infty=True,
                method='chebyshev',
                return_coeffs=False,
                logy=True):
    """Interpolate calculated trajectory using known endpoints. Default settings are for
    unfitness curve.

    Only interpolate through points with sufficiently small errors.

    Parameters
    ----------
    beta_range : ndarray
    dkl : ndarray
    h0 : float
    errs : ndarray
    tol : float, 1e-3
        Error tolerance to use for deciding whether or not to fit points.
    deg : int, None
        Degree of polynomial to fit.
    include_infty : bool, True
    method : str, 'chebyshev'
        Can be 'chebyshev' or 'cubic'
    return_coeffs : bool, False
    logy : bool, False
        If True, fit to the log of y.

    Returns
    -------
    function
        Interpolated function of beta.
    ndarray (optional)
    """

    from scipy.interpolate import interp1d
    from numpy.polynomial.chebyshev import chebfit, chebval
    assert (np.diff(beta_range)>0).all()
    assert beta_range.size==dkl.size
    if not deg is None:
        assert deg<=beta_range.size
    
    if not errs is None:
        keepix = errs<tol
        x = beta_range[keepix]
        y = dkl[keepix]
    else:
        x = beta_range
        y = dkl


    if include_infty and x[-1]!=1:
        x = np.append(x, 1)
        if include_infty is True:
            y = np.append(y, np.log(2) - entropy(h0, base=np.exp(1))) 
        else:
            y = np.append(y, include_infty)

    if method=='cubic':
        fit = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        return fit
    elif method=='chebyshev': 
        if deg is None:
            deg = x.size - 1

        # Chebyshev sometimes seems to extrapolate better but it can show strange
        # divergence
        # fitting to log does much better than linear space (probably because of sharp
        # spike on the right side)
        if logy:
            fit = chebfit(x*2-1, np.log(y), deg)
            if return_coeffs:
                return lambda x, fit=fit: np.exp(chebval(x*2-1, fit)), fit
            return lambda x, fit=fit: np.exp(chebval(x*2-1, fit))

        fit = chebfit(x*2-1, y, deg)
        if return_coeffs:
            return lambda x, fit=fit: chebval(x*2-1, fit), fit
        return lambda x, fit=fit: chebval(x*2-1, fit)
    else:
        raise NotImplementedError

def interpolate_cost(betaRange, cost, errs, h0, tau,
                     agent_prop):
    """Wrapper for calling interpolate on stability cost function.
    """
    
    from .agent import stability_cost
    assert 'weight' in agent_prop.keys()
    assert 'v' in agent_prop.keys()
    info = agent_prop.copy()
    info['tau'] = tau

    inftyCost = stability_cost(info, h0, 0)
    return interpolate(betaRange, cost, h0, errs.ravel(),
                       include_infty=inftyCost, logy=True)

def find_chebmin(*args, **kwargs):
    """Same as interpolate().

    Returns
    -------
    float
    """
    
    from numpy.polynomial.chebyshev import chebder, chebroots

    kwargs['return_coeffs'] = True
    spline, coeffs = interpolate(*args, **kwargs)
    dcoeffs = chebder(coeffs)
    roots = (chebroots(dcoeffs) + 1) / 2

    # only consider real roots within beta in [0,1]
    roots = roots[roots.imag<1e-10]
    roots = roots.real
    roots = roots[(roots>=0)&(roots<=1)]
    
    mnix = np.argmin(spline(roots))
    return roots[mnix], spline(roots[mnix])
