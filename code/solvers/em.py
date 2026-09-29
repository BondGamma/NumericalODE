"""
Euler-Maruyama (EM) solver for the SDE models.

The solver takes a `model` name because the two SDEs need different stepping
algorithms:

    "gbm"   dS = mu S dt + sigma S dW                        (scalar state S)

    "nasv"  dX = (mu - g(Y)^2 / 2) dt + g(Y) dW1             (state (X, Y))
            dY = kappa (theta - Y) dt + xi sqrt(1 + Y^2) dW2

For "gbm" the state is the price S itself; EM on S has strong order 1/2 because
the noise sigma * S is state-dependent.  For "nasv" the state is the log-price X
and the volatility factor Y, and the price is recovered as S = exp(X), which
stays positive automatically.

Only the stepping scheme lives here.  Brownian increments come from BM_engine,
and the convergence / Monte-Carlo studies are assembled in code/experiments.
"""

import numpy as np

from code.SDEs import gbm as _gbm
from code.SDEs import nasv as _nasv

_MODELS = ("gbm", "nasv")


def em_step(model, state, dt, dW, **params):
    """One Euler-Maruyama step for `model` from `state` over (dt, dW).

    Parameters
    ----------
    model : {"gbm", "nasv"}
        Which SDE to step.
    state : float (gbm) or (X, Y) pair (nasv)
        Current state.  Scalars for a single path, or arrays over paths.
    dt : float
        Time step.
    dW : float (gbm) or (dW1, dW2) pair (nasv)
        Brownian increment(s) over this step.
    **params
        Model parameters, forwarded to the SDE right-hand side; defaults are
        defined in code/SDEs/gbm.py and code/SDEs/NA_SV.py.

    Returns
    -------
    float (gbm) or (X, Y) pair (nasv) — the next state.
    """
    if model == "gbm":
        S = state
        return S + _gbm.dS_rhs(S, dt, dW, **params)

    if model == "nasv":
        X, Y = state
        dW1, dW2 = dW
        dX, dY = _nasv.dXdY_rhs(X, Y, dt, dW1, dW2, **params)
        return X + dX, Y + dY

    raise ValueError("unknown model %r; expected one of %s" % (model, _MODELS))


def em_solve(model, y0, dt, dW, **params):
    """Euler-Maruyama over a full increment array.

    Parameters
    ----------
    model : {"gbm", "nasv"}
    y0 :
        gbm  — initial price S0 (scalar or (n_paths,) array).
        nasv — (X0, Y0), each scalar or (n_paths,) array.
    dt : float
    dW :
        gbm  — increments, (n_steps,) or (n_paths, n_steps).
        nasv — (dW1, dW2), each (n_steps,) or (n_paths, n_steps).

    Returns
    -------
    gbm  — S trajectory, (n_steps + 1,) or (n_paths, n_steps + 1).
    nasv — (X, Y) trajectories with the same shapes; recover S = exp(X).
    """
    if model == "gbm":
        S = np.asarray(y0, dtype=float)
        dW = np.asarray(dW, dtype=float)
        single = S.ndim == 0
        if single:
            S = S.reshape(1)
            dW = dW.reshape(1, -1)
        n_paths, n_steps = dW.shape
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = S
        for n in range(n_steps):
            out[:, n + 1] = em_step("gbm", out[:, n], dt, dW[:, n], **params)
        return out[0] if single else out

    if model == "nasv":
        X0, Y0 = y0
        dW1, dW2 = dW
        X = np.asarray(X0, dtype=float)
        Y = np.asarray(Y0, dtype=float)
        dW1 = np.asarray(dW1, dtype=float)
        dW2 = np.asarray(dW2, dtype=float)
        single = X.ndim == 0
        if single:
            X = X.reshape(1)
            Y = Y.reshape(1)
            dW1 = dW1.reshape(1, -1)
            dW2 = dW2.reshape(1, -1)
        n_paths, n_steps = dW1.shape
        Xout = np.empty((n_paths, n_steps + 1))
        Yout = np.empty((n_paths, n_steps + 1))
        Xout[:, 0] = X
        Yout[:, 0] = Y
        for n in range(n_steps):
            Xout[:, n + 1], Yout[:, n + 1] = em_step(
                "nasv", (Xout[:, n], Yout[:, n]), dt, (dW1[:, n], dW2[:, n]),
                **params)
        if single:
            return Xout[0], Yout[0]
        return Xout, Yout

    raise ValueError("unknown model %r; expected one of %s" % (model, _MODELS))
