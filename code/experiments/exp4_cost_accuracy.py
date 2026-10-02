"""
Experiment 4 — GBM Cost vs. Accuracy Analysis

"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from code.SDEs.BM_engine import standard_bm, extract_dw

S0, mu, sigma = 1.0, 0.3, 0.4
T = 1.0
FIG_DIR = "figures"

# Exact solution and solvers 
def exact_gbm(S0, mu, sigma, W_T, T):
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)


def euler_maruyama_gbm(S0, mu, sigma, dW, dt):
    S = np.full(dW.shape[0], S0, dtype=float)
    for k in range(dW.shape[1]):
        S = S + mu * S * dt + sigma * S * dW[:, k]
    return S


def milstein_gbm(S0, mu, sigma, dW, dt):
    S = np.full(dW.shape[0], S0, dtype=float)
    for k in range(dW.shape[1]):
        dw = dW[:, k]
        S = S + mu * S * dt + sigma * S * dw + 0.5 * sigma**2 * S * (dw**2 - dt)
    return S

# Cost-accuracy experiment
def cost_accuracy_experiment(
    n_steps_list=(4, 8, 16, 32, 64, 128, 256),
    n_paths=20_000,
    n_fine_ref=8192,
    seed=123,
):
    for n in n_steps_list:
        if n_fine_ref % n != 0:
            raise ValueError(f"n_fine_ref={n_fine_ref} must be a multiple of n={n}")

    dt_fine = T / n_fine_ref
    dW_fine = standard_bm(n_fine_ref, dt_fine, n_paths=n_paths, seed=seed)

    W_T = dW_fine.sum(axis=1)
    S_exact = exact_gbm(S0, mu, sigma, W_T, T)

    results = []

    print("=" * 60)
    print("GBM COST-ACCURACY EXPERIMENT")
    print("=" * 60)
    print(f"  Paths = {n_paths:,}   reference grid = {n_fine_ref}")
    print()

    for n in n_steps_list:
        dt = T / n
        dW = extract_dw(dW_fine, n)

        # --- Euler-Maruyama ---
        t0 = time.perf_counter()
        S_em = euler_maruyama_gbm(S0, mu, sigma, dW, dt)
        em_wall = time.perf_counter() - t0

        # --- Milstein ---
        t0 = time.perf_counter()
        S_mil = milstein_gbm(S0, mu, sigma, dW, dt)
        mil_wall = time.perf_counter() - t0

        for scheme, S_num, wall in [("EM", S_em, em_wall),
                                    ("Milstein", S_mil, mil_wall)]:
            err = S_num - S_exact
            sq = err * err
            mse = sq.mean()
            # Standard error of the MSE, then of the RMSE
            mse_se = sq.std(ddof=1) / np.sqrt(n_paths)
            rmse = np.sqrt(mse)
            rmse_se = mse_se / (2.0 * np.sqrt(mse)) if mse > 0 else 0.0

            iterations = n_paths * n
            ops = n_paths * n * (8 if scheme == "EM" else 15)

            results.append({
                "scheme":      scheme,
                "n_steps":     n,
                "h":           dt,
                "rmse":        rmse,
                "rmse_se":     rmse_se,
                "wall_time_s": wall,
                "iterations":  iterations,
                "operations":  ops,
            })

    return results

# Plotting
def plot_cost_accuracy(results, target_rmse=5e-2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    schemes = ["EM", "Milstein"]
    colors  = {"EM": "steelblue", "Milstein": "darkorange"}
    markers = {"EM": "o", "Milstein": "s"}

    panels = [
        ("wall_time_s", "Wall-clock time (s)"),
        ("iterations",  "Iterations (n_paths * n_steps)"),
        ("operations",  "Operations"),
    ]

    for ax, (cost_key, cost_label) in zip(axes, panels):
        for scheme in schemes:
            subset = sorted([r for r in results if r["scheme"] == scheme],
                            key=lambda r: r["h"])
            cost = np.array([r[cost_key] for r in subset])
            rmse = np.array([r["rmse"]     for r in subset])
            ax.loglog(cost, rmse,
                      marker=markers[scheme], color=colors[scheme],
                      lw=2, ms=8, label=scheme)
        ax.axhline(target_rmse, color="gray", ls=":", lw=1,
                   label=f"Target RMSE = {target_rmse:.0e}")
        ax.set_xlabel(cost_label)
        ax.set_ylabel("Strong RMSE")
        ax.set_title(f"Cost vs. Accuracy\n({cost_label})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp4_gbm_cost_accuracy.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}\n")


def print_cost_accuracy_table(results):
    lines = []
    lines.append("=" * 100)
    lines.append("GBM COST-ACCURACY SUMMARY TABLE")
    lines.append("=" * 100)
    lines.append(
        f"{'Scheme':10s} {'n_steps':>8s} {'h':>10s} "
        f"{'RMSE':>12s} {'RMSE SE':>12s} "
        f"{'Wall(s)':>10s} {'Iters':>14s} {'Ops':>14s}"
    )
    lines.append("-" * 100)

    for r in sorted(results, key=lambda r: (r["scheme"], r["n_steps"])):
        lines.append(
            f"{r['scheme']:10s} {r['n_steps']:8d} {r['h']:10.5f} "
            f"{r['rmse']:12.3e} {r['rmse_se']:12.3e} "
            f"{r['wall_time_s']:10.3f} {r['iterations']:14,d} "
            f"{r['operations']:14,d}"
        )

    lines.append("=" * 100)
    lines.append("")
    lines.append("NOT COUNTED in the above:")
    lines.append("  - Brownian increment generation (standard_bm)")
    lines.append("  - Exact-solution evaluation used as reference")
    lines.append("  - Memory allocation, GC, interpreter startup")
    lines.append("  - NumPy vectorisation efficiency differences between schemes")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "exp4_gbm_cost_accuracy_table.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {out_path}\n")


def _interp_cost_at_target(results, scheme, target, cost_key):
    subset = sorted([r for r in results if r["scheme"] == scheme],
                    key=lambda r: r["rmse"])
    rmse = np.array([r["rmse"]     for r in subset])
    cost = np.array([r[cost_key]   for r in subset])
    if target < rmse.min() or target > rmse.max():
        return None
    log_c = np.interp(np.log(target), np.log(rmse), np.log(cost))
    return float(np.exp(log_c))


def state_conclusion(results, target_rmse=5e-2):
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"Matched accuracy target: RMSE = {target_rmse:.0e}")
    print()

    for cost_key, label in [("wall_time_s", "wall-clock (s)"),
                            ("iterations",  "iterations"),
                            ("operations",  "operations")]:
        c_em  = _interp_cost_at_target(results, "EM",       target_rmse, cost_key)
        c_mil = _interp_cost_at_target(results, "Milstein", target_rmse, cost_key)
        if c_em is None or c_mil is None:
            print(f"  {label:>16s}: target outside tested range; skip.")
            continue
        ratio = c_mil / c_em
        verdict = "Milstein cheaper" if ratio < 1 else "EM cheaper"
        print(f"  {label:>16s}: EM = {c_em:12.3f}   Milstein = {c_mil:12.3f}   "
              f"ratio = {ratio:5.2f}  →  {verdict}")
    print()
    print("Interpretation:")
    print("  EM has strong order 0.5, Milstein has strong order 1.0.")
    print("  At loose tolerances the per-step overhead of Milstein dominates,")
    print("  so EM is cheaper. At tight tolerances Milstein's higher order")
    print("  wins and it becomes the cheaper scheme.")
    print()


# Main
if __name__ == "__main__":
    print("=" * 60)
    print("GBM COST-ACCURACY ANALYSIS — Experiment 4")
    print("=" * 60)
    print(f"Parameters: S0={S0}, mu={mu}, sigma={sigma}, T={T}")
    print()

    results = cost_accuracy_experiment(
        n_steps_list=(4, 8, 16, 32, 64, 128, 256),
        n_paths=20_000,
        n_fine_ref=8192,
        seed=123,
    )

    print_cost_accuracy_table(results)
    plot_cost_accuracy(results, target_rmse=5e-2)
    state_conclusion(results, target_rmse=5e-2)

    print("=" * 60)
    print("DONE")
    print("=" * 60)