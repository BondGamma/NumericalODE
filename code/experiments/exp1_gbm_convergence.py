"""Experiment 1 — GBM strong + weak (mean) convergence.

One shared Brownian motion drives every step size and both solvers
(`standard_bm_chunks` + `extract_dw`), so each level sees the same Monte-Carlo
error. Outputs a strong-convergence plot, a weak (mean-only) plot and a data
table for later analysis.

    strong error = E|S_N - S(T)|            (same-path exact reference)
    weak   error = |E[S_N] - E[S(T)]|       (E[S(T)] = S0 * exp(mu * T))
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
from code.SDEs.BM_engine import standard_bm_chunks, extract_dw  # noqa: E402

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")

# ---- parameters -----------------------------------------------------------
S0, mu, sigma = 100.0, 0.05, 0.20
T = 1.0
n_max = 2**8
n_paths = 2_000_000
batch_size = 50_000
seed = 321

n_coarse = [2**k for k in range(1, int(np.log2(n_max)) + 1)]
n_levels = len(n_coarse)
dts = np.array([T / n for n in n_coarse])

S_T = S0 * np.exp(mu * T)          # E[S(T)] = analytic expectation of the mean


def exact_gbm(W_T):
    """Closed-form GBM terminal value for a given terminal BM W_T."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)


def euler_maruyama(dW, dt):
    S = np.full(dW.shape[0], S0, dtype=float)
    for k in range(dW.shape[1]):
        S = S + mu * S * dt + sigma * S * dW[:, k]
    return S


def milstein(dW, dt):
    S = np.full(dW.shape[0], S0, dtype=float)
    for k in range(dW.shape[1]):
        dw = dW[:, k]
        S = S + mu * S * dt + sigma * S * dw + 0.5 * sigma**2 * S * (dw**2 - dt)
    return S


def estimate_order(dts, errs):
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


def main():
    solvers = [("Euler-Maruyama", euler_maruyama), ("Milstein", milstein)]
    names = [name for name, _ in solvers]

    strong_sum = {name: np.zeros(n_levels) for name in names}
    mean_sum = {name: np.zeros(n_levels) for name in names}

    for dW_batch in standard_bm_chunks(n_max, T / n_max, n_paths, batch_size, seed=seed):
        W_T = dW_batch.sum(axis=1)
        S_exact = exact_gbm(W_T)
        for j, n in enumerate(n_coarse):
            dt = T / n
            dW = extract_dw(dW_batch, n)
            for name, solver in solvers:
                S_num = solver(dW, dt)
                strong_sum[name][j] += np.sum(np.abs(S_num - S_exact))
                mean_sum[name][j] += np.sum(S_num)

    strong_err = {name: strong_sum[name] / n_paths for name in names}
    S_N = {name: mean_sum[name] / n_paths for name in names}
    weak_err = {name: np.abs(S_N[name] - S_T) for name in names}

    strong_order = {name: estimate_order(dts, strong_err[name]) for name in names}
    weak_order = {name: estimate_order(dts, weak_err[name]) for name in names}

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- strong convergence plot -----------------------------------------
    STRONG_SLOPE = {"Euler-Maruyama": 0.5, "Milstein": 1.0}
    COLORS = {"Euler-Maruyama": "tab:blue", "Milstein": "tab:orange"}
    SLOPE_STYLE = {0.5: "--", 1.0: ":"}
    SLOPE_LABEL = {0.5: "slope = 0.5", 1.0: "slope = 1.0"}

    def legend_no_dupes(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        ax.loglog(dts, strong_err[name], "o-", color=COLORS[name], label=name)
        p = STRONG_SLOPE[name]
        ax.loglog(dts, strong_err[name][0] * (dts / dts[0]) ** p,
                  linestyle=SLOPE_STYLE[p], color="0.3", label=SLOPE_LABEL[p])
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"strong error $E|S_N - S(T)|$")
    legend_no_dupes(ax)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp1_gbm_strong_convergence.png"), dpi=150)
    plt.close(fig)

    # ---- weak (mean) convergence plot ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        ax.loglog(dts, weak_err[name], "o-", color=COLORS[name], label=name)
        ax.loglog(dts, weak_err[name][0] * (dts / dts[0]) ** 1.0,
                  linestyle=SLOPE_STYLE[1.0], color="0.3", label=SLOPE_LABEL[1.0])
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"weak error $|E[S_N] - E[S(T)]|$")
    legend_no_dupes(ax)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp1_gbm_weak_mean.png"), dpi=150)
    plt.close(fig)

    # ---- data table -------------------------------------------------------
    lines = []
    lines.append(f"S_T = E[S(T)] = S0*exp(mu*T) = {S_T:.6f}")
    lines.append("")
    header = (f"{'n':>4} {'dt':>10} {'S_T':>12} "
              f"{'EM S_N':>12} {'EM strong':>12} {'EM weak':>12} "
              f"{'Mil S_N':>12} {'Mil strong':>12} {'Mil weak':>12}")
    lines.append(header)
    lines.append("-" * len(header))
    for j, n in enumerate(n_coarse):
        lines.append(f"{n:>4} {dts[j]:>10.6f} {S_T:>12.6f} "
                     f"{S_N['Euler-Maruyama'][j]:>12.6f} "
                     f"{strong_err['Euler-Maruyama'][j]:>12.6f} "
                     f"{weak_err['Euler-Maruyama'][j]:>12.6f} "
                     f"{S_N['Milstein'][j]:>12.6f} "
                     f"{strong_err['Milstein'][j]:>12.6f} "
                     f"{weak_err['Milstein'][j]:>12.6f}")
    lines.append("")
    for name in names:
        lines.append(f"{name:16s} strong order ~= {strong_order[name]:.3f}, "
                     f"weak order ~= {weak_order[name]:.3f}")

    table = "\n".join(lines)
    print(table)
    with open(os.path.join(FIGURES_DIR, "exp1_gbm_convergence_table.txt"), "w") as f:
        f.write(table + "\n")

    print(f"Saved {FIGURES_DIR}/exp1_gbm_strong_convergence.png")
    print(f"Saved {FIGURES_DIR}/exp1_gbm_weak_mean.png")
    print(f"Saved {FIGURES_DIR}/exp1_gbm_convergence_table.txt")


if __name__ == "__main__":
    main()
