"""Experiment 3 — NA-SV (nasv) strong + weak (European call) convergence, EM only.

Mirrors exp1's GBM convergence study but for the NA-SV model:

    dX_t = (mu - g(Y_t)^2 / 2) dt + g(Y_t) dW_t^(1)
    dY_t = kappa (theta - Y_t) dt + xi sqrt(1 + Y_t^2) dW_t^(2)
    S_t  = exp(X_t),   g(y) = sigma_min + (sigma_max - sigma_min)/(1 + exp(-y))
    d<W^(1), W^(2)>_t = rho dt

One shared correlated Brownian pair (dW1, dW2) drives every step size: it is drawn
once at the finest grid 2^12 and coarsened via `extract_dw`, so each dt level sees
the exact same path.  There is no closed-form solution, so two fine EM references
(2^11 and 2^12) serve as the "truth"; the strong error is measured against both, and
the weak functional is the European call payoff E[(S_T - 100)^+].

    strong error = E|S_N - S_ref|            (S_ref = EM at 2^11 or 2^12, same path)
    weak   error = |E[payoff_N] - E[payoff_ref]|,  payoff = max(S_T - K, 0), K = 100

Outputs a strong-convergence plot, a weak-convergence plot and a data table.  EM only
(no Milstein: scalar Milstein does not extend to this two-noise system).
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
from code.SDEs import nasv  # noqa: E402
from code.SDEs.BM_engine import correlated_bm_chunks, extract_dw  # noqa: E402

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")

# ---- parameters (shared with code/SDEs/nasv.py) ----------------------------
T = nasv.NA_SV_T
K = 100.0                      # European call strike

n_ref_hi = 2**12               # 4096 — finest reference (no exact solution)
n_ref_lo = 2**11               # 2048 — coarser reference
n_coarse = [2**k for k in range(2, 9)]   # 4, 8, 16, 32, 64, 128, 256
n_levels = len(n_coarse)
dts = np.array([T / n for n in n_coarse])

n_paths = 2_000_000            # Monte-Carlo sample size (paired Brownian motion)
batch_size = 10_000            # stream the ensemble in path batches
seed = 321


def em_terminal(dW1, dW2, dt):
    """Euler-Maruyama to T over the given increments; returns terminal price S_T.

    Vectorized over the path axis (axis 0 of dW1/dW2); reuses nasv.g for the
    logistic volatility and the nasv parameters, so the discretisation is the one
    in code/SDEs/nasv.py.
    """
    X = np.full(dW1.shape[0], np.log(nasv.NA_SV_S0), dtype=float)
    Y = np.full(dW1.shape[0], nasv.NA_SV_Y0, dtype=float)
    for i in range(dW1.shape[1]):
        sig = nasv.g(Y)
        X = X + (nasv.NA_SV_MU - 0.5 * sig * sig) * dt + sig * dW1[:, i]
        Y = Y + nasv.NA_SV_KAPPA * (nasv.NA_SV_THETA - Y) * dt \
              + nasv.NA_SV_XI * np.sqrt(1.0 + Y * Y) * dW2[:, i]
    return np.exp(X)


def estimate_order(dts, errs):
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


def main():
    strong_hi = np.zeros(n_levels)   # E|S_N - S_ref(2^12)|
    strong_lo = np.zeros(n_levels)   # E|S_N - S_ref(2^11)|
    payoff_sum = np.zeros(n_levels)
    ref_hi_sum = 0.0                 # sum of payoff(2^12)
    ref_lo_sum = 0.0                 # sum of payoff(2^11)

    for dW1, dW2 in correlated_bm_chunks(
        n_ref_hi, T / n_ref_hi, nasv.NA_SV_RHO, n_paths, batch_size, seed=seed
    ):
        # fine references on the SAME Brownian path
        S_ref_hi = em_terminal(dW1, dW2, T / n_ref_hi)
        dW1_lo = extract_dw(dW1, n_ref_lo)
        dW2_lo = extract_dw(dW2, n_ref_lo)
        S_ref_lo = em_terminal(dW1_lo, dW2_lo, T / n_ref_lo)
        del dW1_lo, dW2_lo

        ref_hi_sum += float(np.sum(np.maximum(S_ref_hi - K, 0.0)))
        ref_lo_sum += float(np.sum(np.maximum(S_ref_lo - K, 0.0)))

        for j, n in enumerate(n_coarse):
            dW1c = extract_dw(dW1, n)
            dW2c = extract_dw(dW2, n)
            S_num = em_terminal(dW1c, dW2c, T / n)
            strong_hi[j] += float(np.sum(np.abs(S_num - S_ref_hi)))
            strong_lo[j] += float(np.sum(np.abs(S_num - S_ref_lo)))
            payoff_sum[j] += float(np.sum(np.maximum(S_num - K, 0.0)))

    strong_hi /= n_paths
    strong_lo /= n_paths
    E_ref_hi = ref_hi_sum / n_paths
    E_ref_lo = ref_lo_sum / n_paths
    E_num = payoff_sum / n_paths
    weak_hi = np.abs(E_num - E_ref_hi)   # |E[payoff_N] - E[payoff_ref(2^12)]|
    weak_lo = np.abs(E_num - E_ref_lo)   # |E[payoff_N] - E[payoff_ref(2^11)]|

    strong_order = estimate_order(dts, strong_hi)
    weak_order = estimate_order(dts, weak_hi)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- plotting colours: two shades of blue for the two references ---------
    LIGHT = "#9ecae1"   # vs 2^11 reference
    DARK = "#08519c"    # vs 2^12 reference
    SLOPE_COLOR = "0.3"

    # ---- strong convergence plot -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dts, strong_lo, "o-", color=LIGHT,
              label=r"EM vs $2^{11}$ reference")
    ax.loglog(dts, strong_hi, "o-", color=DARK,
              label=r"EM vs $2^{12}$ reference")
    ax.loglog(dts, strong_hi[0] * (dts / dts[0]) ** 0.5,
              linestyle="--", color=SLOPE_COLOR, label="slope = 0.5")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"strong error $E|S_N - S_{ref}|$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_nasv_strong_convergence.png"),
                dpi=150)
    plt.close(fig)

    # ---- weak convergence plot ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dts, weak_lo, "o-", color=LIGHT,
              label=r"EM vs $2^{11}$ reference")
    ax.loglog(dts, weak_hi, "o-", color=DARK,
              label=r"EM vs $2^{12}$ reference")
    ax.loglog(dts, weak_hi[0] * (dts / dts[0]) ** 1.0,
              linestyle=":", color=SLOPE_COLOR, label="slope = 1.0")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"weak error $|E[\mathrm{payoff}_N] - E[\mathrm{payoff}_{ref}]|$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_nasv_weak_convergence.png"),
                dpi=150)
    plt.close(fig)

    # ---- data table ---------------------------------------------------------
    lines = []
    lines.append(f"K = {K:g}   E[payoff_ref(2^11)] = {E_ref_lo:.6f}   "
                 f"E[payoff_ref(2^12)] = {E_ref_hi:.6f}")
    lines.append("")
    header = (f"{'n':>4} {'dt':>10} {'E[payoff_N]':>12} "
              f"{'strong 2^11':>12} {'strong 2^12':>12} "
              f"{'weak 2^11':>12} {'weak 2^12':>12}")
    lines.append(header)
    lines.append("-" * len(header))
    for j, n in enumerate(n_coarse):
        lines.append(f"{n:>4} {dts[j]:>10.6f} {E_num[j]:>12.6f} "
                     f"{strong_lo[j]:>12.6f} {strong_hi[j]:>12.6f} "
                     f"{weak_lo[j]:>12.6f} {weak_hi[j]:>12.6f}")
    lines.append("")
    lines.append(f"Euler-Maruyama   strong order ~= {strong_order:.3f}, "
                 f"weak order ~= {weak_order:.3f}")

    table = "\n".join(lines)
    print(table)
    with open(os.path.join(FIGURES_DIR, "exp3_nasv_convergence_table.txt"),
              "w") as f:
        f.write(table + "\n")

    print(f"Saved {FIGURES_DIR}/exp3_nasv_strong_convergence.png")
    print(f"Saved {FIGURES_DIR}/exp3_nasv_weak_convergence.png")
    print(f"Saved {FIGURES_DIR}/exp3_nasv_convergence_table.txt")


if __name__ == "__main__":
    main()
