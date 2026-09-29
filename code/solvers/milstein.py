"""
Milstein solver for the SDE models.

For a scalar SDE with state-dependent noise the Milstein correction raises the
strong order from 1/2 (Euler-Maruyama) to 1:

    gbm:  S_{n+1} = S_n + mu S_n dt + sigma S_n dW_n
                    + (1/2) sigma^2 S_n (dW_n^2 - dt)

For "nasv" this solver deliberately refuses to run: the scalar Milstein result
does not extend to that two-noise system (a full multidimensional Milstein
method needs cross terms and simulated iterated stochastic integrals — see the
problem pack).  Use Euler-Maruyama for NA-SV.
"""

import numpy as np

from code.SDEs import gbm as _gbm

_MODELS = ("gbm", "nasv")


def milstein_step(model, state, dt, dW, **params):
    """One Milstein step for `model`.

    Only "gbm" is supported; "nasv" raises NotImplementedError (see the module
    docstring for why).
    """
    if model == "gbm":
        S = state
        sigma = params.get("sigma", _gbm.GBM_SIGMA)
        return (S
                + _gbm.dS_rhs(S, dt, dW, **params)
                + _gbm.dS_correction(S, dt, dW, sigma=sigma))

    if model == "nasv":
        raise NotImplementedError(
            "scalar Milstein does not extend to the NA-SV two-noise system; a "
            "full multidimensional Milstein method needs cross terms and "
            "simulated iterated stochastic integrals"
        )

    raise ValueError("unknown model %r; expected one of %s" % (model, _MODELS))


def milstein_solve(model, y0, dt, dW, **params):
    """Milstein over a full increment array (gbm only; nasv raises)."""
    if model == "nasv":
        raise NotImplementedError(
            "scalar Milstein does not extend to the NA-SV two-noise system; "
            "use Euler-Maruyama (em.py) for NA-SV"
        )
    if model != "gbm":
        raise ValueError("unknown model %r; expected one of %s" % (model, _MODELS))

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
        out[:, n + 1] = milstein_step("gbm", out[:, n], dt, dW[:, n], **params)
    return out[0] if single else out
