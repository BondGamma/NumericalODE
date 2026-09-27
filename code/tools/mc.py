"""
Monte-Carlo Brownian-motion simulator (tool).

Generates fine Brownian increments ONCE at the maximum resolution n_max and
saves the log to `data/`:

    "standard" -> data/gbm_dw_log.npy    independent standard BM (for GBM)
    "pairing"  -> data/nasv_dw_log.npy   correlated pair       (for NA-SV)

Those logs are the input to bm_engine.extract_dw, which coarsens them to any
smaller step count n (n | n_max) by summing blocks.  Because every coarse
increment is a sum of fine increments, all step sizes share the SAME underlying
Brownian motion — the key requirement for comparing convergence across dt.

This module only simulates and stores the randomness; the SDE solvers
(code/solvers) and the experiments (code/experiments) consume these logs.

Re-running `simulate` overwrites the same log files (`np.save` replaces rather
than appends), so previous runs are intentionally not kept.
"""

import os

import numpy as np

from code.SDEs import BM_engine as bm_engine
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

STANDARD_LOG = "gbm_dw_log.npy"   # standard BM increments: (n_sims, n_max)
PAIRING_LOG = "nasv_dw_log.npy"   # correlated pair:         (2, n_sims, n_max)


def simulate(bm_type, n_sims, n_max, T=1.0, rho=None, seed=None,
             data_dir=DEFAULT_DATA_DIR):
    """Simulate fine Brownian increments and save the log to `data_dir`.

    Parameters
    ----------
    bm_type : {"standard", "pairing"}
        "standard" -> independent standard BM (saved as gbm_dw_log.npy).
        "pairing"  -> correlated pair (saved as nasv_dw_log.npy).
    n_sims : int
        Number of simulations (paths).
    n_max : int
        Number of fine steps (maximum resolution); dt = T / n_max.
    T : float
        Time horizon.
    rho : float
        Correlation for "pairing" (required, |rho| < 1).
    seed : int or None
        RNG seed.
    data_dir : str
        Output directory (created if missing).

    Returns
    -------
    ndarray — the saved log:
        "standard" -> (n_sims, n_max)
        "pairing"  -> (2, n_sims, n_max), index 0 = dW1, index 1 = dW2.
    """
    dt = T / n_max

    if bm_type == "standard":
        dw = bm_engine.standard_bm(n_max, dt, n_sims, seed=seed)
        fname = STANDARD_LOG
    elif bm_type == "pairing":
        if rho is None:
            raise ValueError("rho is required for bm_type='pairing'")
        dw1, dw2 = bm_engine.correlated_bm_pair(n_max, dt, rho, n_sims, seed=seed)
        dw = np.stack([dw1, dw2], axis=0)
        fname = PAIRING_LOG
    else:
        raise ValueError("unknown bm_type %r; expected 'standard' or 'pairing'"
                         % (bm_type,))

    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, fname), dw)
    return dw


def load(bm_type, data_dir=DEFAULT_DATA_DIR):
    """Load a previously saved increment log.

    Parameters
    ----------
    bm_type : {"standard", "pairing"}
    data_dir : str

    Returns
    -------
    ndarray — the saved log (same shapes as `simulate`).
    """
    if bm_type == "standard":
        fname = STANDARD_LOG
    elif bm_type == "pairing":
        fname = PAIRING_LOG
    else:
        raise ValueError("unknown bm_type %r; expected 'standard' or 'pairing'"
                         % (bm_type,))
    return np.load(os.path.join(data_dir, fname))
