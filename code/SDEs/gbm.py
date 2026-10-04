"""
Geometric Brownian motion (GBM)

    dS = mu S dt + sigma S dW

with exact solution

    S(t) = S0 exp((mu - sigma^2 / 2) t + sigma W(t)).

This module provides the four building blocks requested for the project:

    dS_rhs         Euler-Maruyama right-hand side on the *price* S
                   (state-dependent noise -> strong order 1/2 for EM)
    dS_correction  Milstein correction term on the price S
                   (dS_rhs + dS_correction -> strong order 1)
    dX_rhs         Euler-Maruyama right-hand side on the log-price X = log S
                   (constant coefficients -> additive noise)
    S_exact        S = exp(X), the exact price from the log-price

Working in the log-price X = log S, Ito's lemma gives the constant-coefficient
SDE

    dX = (mu - sigma^2 / 2) dt + sigma dW,

whose transition is exactly Gaussian, so EM on X *is* the exact sampler and the
Milstein correction vanishes on X (additive noise).  To measure a strong order of
1/2 vs 1 one discretises the price S itself, where the noise sigma * S is
state-dependent.
"""

import numpy as np

# Default benchmark parameters (see problem pack Section 4).
GBM_S0 = 100.0
GBM_MU = 0.05
GBM_SIGMA = 0.20
GBM_T = 1.0


def dS_rhs(S, dt, dW, mu=GBM_MU, sigma=GBM_SIGMA):
    """Euler-Maruyama increment for dS = mu S dt + sigma S dW.

        dS_rhs = mu * S * dt + sigma * S * dW

    The update is S_{n+1} = S_n + dS_rhs(S_n, dt, dW_n).
    """
    return mu * S * dt + sigma * S * dW


def dS_correction(S, dt, dW, sigma=GBM_SIGMA):
    """Milstein correction for GBM on the price S.

        dS_correction = (1/2) sigma^2 S (dW^2 - dt)

    This is the extra term that raises the strong order on S from 1/2 (EM) to 1
    (Milstein).  Note the sigma^2 factor: for the diffusion b(S) = sigma S the
    Milstein correction is

        (1/2) b(S) b'(S) (dW^2 - dt)
        = (1/2) sigma S * sigma * (dW^2 - dt)
        = (1/2) sigma^2 S (dW^2 - dt).
    """
    return 0.5 * sigma * sigma * S * (dW * dW - dt)


def dX_rhs(dt, dW, mu=GBM_MU, sigma=GBM_SIGMA):
    """Euler-Maruyama increment for the log-price SDE.

        dX = (mu - sigma^2 / 2) dt + sigma dW

    Constant drift and constant diffusion, so this does not depend on X and the
    Milstein correction is identically zero.
    """
    return (mu - 0.5 * sigma * sigma) * dt + sigma * dW


def S_exact(X):
    """Exact price from the log-price: S = exp(X)."""
    return np.exp(X)
