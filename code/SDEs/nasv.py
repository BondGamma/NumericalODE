"""
Non-affine stochastic-volatility (NA-SV) model.

    dX_t = (mu - g(Y_t)^2 / 2) dt + g(Y_t) dW_t^(1)
    dY_t = kappa (theta - Y_t) dt + xi sqrt(1 + Y_t^2) dW_t^(2)

    g(y)  = sigma_min + (sigma_max - sigma_min) / (1 + exp(-y))

    d<W^(1), W^(2)>_t = rho dt.

X_t = log S_t, so the price is recovered as S = exp(X), which stays positive by
construction.  The logistic g keeps instantaneous volatility in
(sigma_min, sigma_max), and the nonlinear Y-diffusion xi sqrt(1 + Y^2) makes the
joint process non-affine; no closed-form transition / practical exact sampler is
known, so numerical SDE integration is part of the model solution.

This module provides the single function:

    dXdY_rhs     Euler-Maruyama increment of the state z = (X, Y) as a length-2
                 numpy array.  There is NO Milstein correction here: the scalar
                 Milstein order-1 result does not carry over to this two-noise
                 system without cross terms and simulated iterated stochastic
                 integrals (see the problem pack).
"""

import numpy as np

NA_SV_S0 = 100.0
NA_SV_Y0 = 0.0
NA_SV_MU = 0.05
NA_SV_KAPPA = 2.0
NA_SV_THETA = -0.2
NA_SV_XI = 0.6
NA_SV_RHO = -0.7
NA_SV_SIGMA_MIN = 0.10
NA_SV_SIGMA_MAX = 0.50
NA_SV_T = 1.0


def g(y, sigma_min=NA_SV_SIGMA_MIN, sigma_max=NA_SV_SIGMA_MAX):
    """Bounded logistic instantaneous volatility g: R -> (sigma_min, sigma_max).

        g(y) = sigma_min + (sigma_max - sigma_min) / (1 + exp(-y))

    Smooth, monotone, saturating: g(-inf) = sigma_min, g(+inf) = sigma_max,
    g(0) = (sigma_min + sigma_max) / 2.
    """
    return sigma_min + (sigma_max - sigma_min) / (1.0 + np.exp(-y))


def dXdY_rhs(X, Y, dt, dW1, dW2,
             mu=NA_SV_MU, kappa=NA_SV_KAPPA, theta=NA_SV_THETA, xi=NA_SV_XI,
             sigma_min=NA_SV_SIGMA_MIN, sigma_max=NA_SV_SIGMA_MAX):
    """Euler-Maruyama increment of the NA-SV state (X, Y), as a length-2 array.

        dX = (mu - g(Y)^2 / 2) dt + g(Y) dW1
        dY = kappa (theta - Y) dt + xi sqrt(1 + Y^2) dW2

    No Milstein correction term.  The correlated increments (dW1, dW2) must come
    from the Brownian engine (BM_engine.correlated_bm_pair):

        dW1 = sqrt(dt) Z1
        dW2 = sqrt(dt) (rho Z1 + sqrt(1 - rho^2) Z2),   Z1, Z2 iid N(0, 1).

    Recover the price with S = exp(X).
    """
    # `X` is part of the (X, Y) state signature but does not enter the X-drift
    # or X-diffusion: both depend on Y alone (through g(Y)), as in the SDE above.
    _ = X
    sig = g(Y, sigma_min, sigma_max)
    dX = (mu - 0.5 * sig * sig) * dt + sig * dW1
    dY = kappa * (theta - Y) * dt + xi * np.sqrt(1.0 + Y * Y) * dW2
    return np.array([dX, dY])
