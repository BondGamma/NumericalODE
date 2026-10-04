#Brownian-motion engine


import numpy as np


def standard_bm(n_steps, dt, n_paths=1, seed=None):
    """Independent standard Brownian increments, shape (n_paths, n_steps).

        dW_n = sqrt(dt) * Z_n,   Z_n iid N(0, 1)

    Returns
    -------
    dW : ndarray, shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    return np.sqrt(dt) * Z


def correlated_bm_pair(n_steps, dt, rho, n_paths=1, seed=None):
    """Correlated increments (dW1, dW2) with correlation rho.

        dW1_n = sqrt(dt) Z1_n
        dW2_n = sqrt(dt) (rho Z1_n + sqrt(1 - rho^2) Z2_n),   Z1, Z2 iid N(0,1)

    Returns
    -------
    dW1, dW2 : ndarray, each of shape (n_paths, n_steps)
        With corr(dW1, dW2) = rho and Var(dW2) = dt.
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    dW1 = np.sqrt(dt) * Z1
    dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1.0 - rho * rho) * Z2)
    return dW1, dW2


def extract_dw(dw_log, n_steps):
    """Coarsen fine Brownian increments to `n_steps` steps by summing blocks.

    This is the "same Brownian motion" extracter/generaliser: given increments
    at the finest resolution n_max (the last axis of `dw_log`), return the
    increments at a ROUGHER (coarse) resolution n_steps, where each coarse increment is
    the sum of `n_max / n_steps` consecutive fine increments.  Summing in this
    way guarantees every step size shares the SAME underlying Brownian path
    (the coarse path is exactly the fine path sampled at the coarse grid), so
    convergence studies across dt compare the same realisation.

    Parameters
    ----------
    dw_log : ndarray, shape (..., n_max)
        Fine increments.  Standard BM: (n_sims, n_max).  Correlated pair:
        (2, n_sims, n_max).  A single path (n_max,) is also accepted.
    n_steps : int
        Target number of (coarser) steps.  Must divide n_max.

    Returns
    -------
    ndarray, shape (..., n_steps) — the coarsened increments.
    """
    arr = np.asarray(dw_log, dtype=float)
    n_max = arr.shape[-1]
    if n_steps <= 0 or n_max % n_steps != 0:
        raise ValueError("n_steps=%d must divide n_max=%d" % (n_steps, n_max))
    block = n_max // n_steps
    return arr.reshape(arr.shape[:-1] + (n_steps, block)).sum(axis=-1)
