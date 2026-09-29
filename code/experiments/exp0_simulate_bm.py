"""
Experiment 0 — smoke test of the "same Brownian motion" pipeline.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)

from code.tools import mc                       # noqa: E402
from code.SDEs import BM_engine as bm_engine                 # noqa: E402
from code.SDEs import gbm                       # noqa: E402
from code.SDEs import nasv                      # noqa: E402
from code.solvers import em as em_solver        # noqa: E402
from code.solvers import milstein as milstein_solver  # noqa: E402
from code.visualizations import plot_paths      # noqa: E402
from code.visualizations import plot_paths_band            # noqa: E402
from code.visualizations import plot_paths_band_slices     # noqa: E402

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")


def _coarse_path(dw_fine, n):
    """Cumulative coarse Brownian path sampled from fine increments at n steps."""
    coarse = bm_engine.extract_dw(dw_fine, n)
    return np.concatenate([[0.0], np.cumsum(coarse)])


def _path_specs(dw_fine, n_max, n_list, dt):
    """Build the (t, y, style) specs for the fine path + coarse overlaid samples."""
    t_fine = dt * np.arange(n_max + 1)
    W_fine = np.concatenate([[0.0], np.cumsum(dw_fine)])
    specs = [(t_fine, W_fine,
              {"lw": 1.0, "color": "0.35", "label": "fine (n=%d)" % n_max})]
    # Larger n -> more sample points -> smaller pins, and plot in ascending-n
    # order so the smallest pins are drawn last (on top).  Otherwise all pins
    # at a shared grid point overlap and only the last colour is visible.
    for size, n in zip(np.linspace(8.0, 4.0, len(n_list)), sorted(n_list)):
        W = _coarse_path(dw_fine, n)
        t = (dt * n_max / n) * np.arange(n + 1)
        specs.append((t, W, {"marker": "o", "ls": "", "ms": size, "label": "n=%d" % n}))
    return specs


# --- advanced path-plotting test: confidence band + cross-sections --------- #

_BAND_MODELS = ("bm", "gbm_exact", "gbm_em", "gbm_milstein", "nasv")


def _simulate_ensemble(model, n_paths, n_steps, T=1.0, seed=321):
    """Simulate an ensemble of 1-D paths for `model`.

    Returns ``(t, paths)`` with ``paths`` of shape (n_paths, n_steps + 1).
    Covers every model the advanced plotters are meant to serve: pure Brownian
    motion, exact GBM, EM/Milstein-approximated GBM, and the NA-SV price.
    """
    dt = T / n_steps
    t = dt * np.arange(n_steps + 1)

    if model == "bm":                       # pure Brownian motion W(t)
        dW = bm_engine.standard_bm(n_steps, dt, n_paths, seed=seed)
        W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)],
                           axis=1)
        return t, W

    if model == "gbm_exact":                # exact GBM from the log-price
        dW = bm_engine.standard_bm(n_steps, dt, n_paths, seed=seed)
        dX = gbm.dX_rhs(dt, dW)             # (mu - sigma^2/2) dt + sigma dW
        X = np.log(gbm.GBM_S0) + np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(dX, axis=1)], axis=1)
        return t, gbm.S_exact(X)

    if model in ("gbm_em", "gbm_milstein"):  # EM / Milstein on the price S
        dW = bm_engine.standard_bm(n_steps, dt, n_paths, seed=seed)
        S0 = np.full(n_paths, gbm.GBM_S0)
        S = (em_solver.em_solve("gbm", S0, dt, dW) if model == "gbm_em"
             else milstein_solver.milstein_solve("gbm", S0, dt, dW))
        return t, S

    if model == "nasv":                     # NA-SV price S = exp(X)
        dW1, dW2 = bm_engine.correlated_bm_pair(
            n_steps, dt, nasv.NA_SV_RHO, n_paths, seed=seed)
        X0 = np.full(n_paths, np.log(nasv.NA_SV_S0))
        Y0 = np.full(n_paths, nasv.NA_SV_Y0)
        X, _ = em_solver.em_solve("nasv", (X0, Y0), dt, (dW1, dW2))
        return t, np.exp(X)

    raise ValueError("unknown model %r; expected one of %s"
                     % (model, _BAND_MODELS))


def _demo_band_plots():
    """Exercise plot_paths_band and plot_paths_band_slices on every model.

    Each model is simulated (``seed``) and sampled (``rng``) with its own
    per-model value.  This matters: ``standard_bm`` and ``correlated_bm_pair``
    both draw from ``np.random.default_rng(seed)``, so a shared seed would make
    the NA-SV's dW1 byte-identical to the GBM's dW and the plots would look
    like the same process when they are not.
    """
    n_paths, n_steps = 1000, 256
    for i, model in enumerate(_BAND_MODELS):
        t, paths = _simulate_ensemble(model, n_paths, n_steps, T=1.0, seed=i)
        ylabel = "W(t)" if model == "bm" else "S(t)"

        fig, _ = plot_paths_band(
            paths, t, q=0.05, poly_deg=8, sample=10, rng=i,
            band_label="90% band", ylabel=ylabel,
            title="Confidence band — %s" % model,
            save_path=os.path.join(FIGURES_DIR, "band_%s.png" % model),
            show=False)
        plt.close(fig)

        fig, _ = plot_paths_band_slices(
            paths, t, times=(1.0 / 3.0, 2.0 / 3.0, 1.0), q=0.05, poly_deg=8,
            sample=10, rng=i, band_label="90% band", ylabel=ylabel,
            title="Cross-sections — %s" % model,
            save_path=os.path.join(FIGURES_DIR, "slices_%s.png" % model),
            show=False)
        plt.close(fig)

    print("wrote %d band/slice figures to %s"
          % (2 * len(_BAND_MODELS), FIGURES_DIR))


def main():
    n_sims = 2
    n_max = 1024
    T = 1.0
    rho = nasv.NA_SV_RHO
    n_list = [128, 256, 512]            
    dt = T / n_max

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1) simulate + save the fine logs, then reload to test the round-trip.
    mc.simulate("standard", n_sims, n_max, T=T, seed=0)
    mc.simulate("pairing", n_sims, n_max, T=T, rho=rho, seed=0)
    gbm_log = mc.load("standard")
    nasv_log = mc.load("pairing")

    # 2) + 3) visualise each simulation: fine path + coarse samples overlaid.
    for i in range(n_sims):
        fig, _ = plot_paths(
            _path_specs(gbm_log[i], n_max, n_list, dt),
            title="Standard BM — simulation %d" % (i + 1),
            xlabel="t", ylabel="W(t)", figsize=(7, 4.5),
            save_path=os.path.join(FIGURES_DIR, "bm_standard_sim%d.png" % (i + 1)),
            show=False)
        plt.close(fig)

    for i in range(n_sims):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        plot_paths(_path_specs(nasv_log[0, i], n_max, n_list, dt),
                   ax=axes[0], title="Correlated BM W1 — sim %d" % (i + 1),
                   xlabel="t", ylabel="W(t)", show=False)
        plot_paths(_path_specs(nasv_log[1, i], n_max, n_list, dt),
                   ax=axes[1], title="Correlated BM W2 — sim %d" % (i + 1),
                   xlabel="t", ylabel="W(t)", show=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "bm_pairing_sim%d.png" % (i + 1)),
                    dpi=150)
        plt.close(fig)

    print("wrote 4 figures to %s" % FIGURES_DIR)

    # 4) advanced plotters MAGIIIIC: confidence band + cross-section distributions.
    _demo_band_plots()


if __name__ == "__main__":
    main()
