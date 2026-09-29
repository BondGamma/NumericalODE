"""
Experiment 4 — NASV Cost vs. Accuracy Analysis
 
Compares Euler-Maruyama (EM) and Milstein schemes for the NASV model
by measuring computational cost at matched accuracy targets.

We used the assessing rubrick where we used KPIs like wall-clock, iterations, operations.

"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# path magic to make it work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_nasv_convergence import (
    g, g_prime,
    _fine_increments, _em_path_from_increments, _em_coarse_from_fine,
    T, S0, Y0, mu, kappa, theta, xi, rho,
    sigma_min, sigma_max,
)
LOG_S0 = np.log(S0)
FIG_DIR = "figures"


# ============================================================
# 1. MILSTEIN COARSE INTEGRATOR
# ============================================================
def _milstein_coarse_from_fine(dW1_fine, dW2_fine, n_coarse):
    """This is basically to get the coarse (rough) incriments from Brownian "engine", and then putting them in the blocks."""
    n_paths, n_fine = dW1_fine.shape
    if n_fine % n_coarse != 0:
        raise ValueError(f"n_fine={n_fine} must be a multiple of n={n_coarse}")
    ref = n_fine // n_coarse
    dt_c = T / n_coarse

    dW1_r = dW1_fine.reshape(n_paths, n_coarse, ref)
    dW2_r = dW2_fine.reshape(n_paths, n_coarse, ref)

    dW1_sum = dW1_r.sum(axis=2)
    dW2_sum = dW2_r.sum(axis=2)

    # 1-D Milstein correction on Y: sum over fine sub-steps of
    #   (dW2)^2 - dt_fine  = (dW2_coarse)^2 - dt_coarse + cross-terms
    sum_dW2_sq_minus_dt = (dW2_r * dW2_r).sum(axis=2) - dt_c

    X = np.full(n_paths, LOG_S0)
    Y = np.full(n_paths, Y0)
    for j in range(n_coarse):
        gY    = g(Y)
        diffY = np.sqrt(1.0 + Y * Y)

        X = X + (mu - 0.5 * gY * gY) * dt_c + gY * dW1_sum[:, j]

        Y = Y + kappa * (theta - Y) * dt_c + xi * diffY * dW2_sum[:, j]
        Y = Y + 0.5 * xi * xi * Y * sum_dW2_sq_minus_dt[:, j]

    return X, Y


# ============================================================
# 2. COST-ACCURACY EXPERIMENT
# ============================================================
def cost_accuracy_experiment(
    n_steps_list=(4, 8, 16, 32, 64),
    n_paths=50_000,
    batch_size=5_000,
    n_fine_ref=8192,
    K=100.0,
    seed=123,
):
    """For each scheme and resolution, measure bias (so like a distance from the payoff), MC error (the simulated noise), and cost."""
    if n_paths % batch_size != 0:
        raise ValueError("n_paths must be a multiple of batch_size")
    for n in n_steps_list:
        if n_fine_ref % n != 0:
            raise ValueError(f"n_fine_ref={n_fine_ref} must be a multiple of n={n}")

    n_batches = n_paths // batch_size

    # Collecting stuff for EM and Milstein (paired differences vs reference)
    em_sum  = {n: 0.0 for n in n_steps_list}
    em_sq   = {n: 0.0 for n in n_steps_list}
    mil_sum = {n: 0.0 for n in n_steps_list}
    mil_sq  = {n: 0.0 for n in n_steps_list}

    # Accumulated wall time per scheme per resolution
    em_times  = {n: 0.0 for n in n_steps_list}
    mil_times = {n: 0.0 for n in n_steps_list}

    print("=" * 60)
    print("COST-ACCURACY EXPERIMENT")
    print("=" * 60)
    print(f"  Paths = {n_paths:,}   batch = {batch_size:,}   "
          f"batches = {n_batches}")
    print(f"  Reference grid n_fine_ref = {n_fine_ref}")
    print()

    t_start_all = time.perf_counter()

    for b in range(n_batches):
        seed_b = seed + b
        dW1, dW2, _ = _fine_increments(n_fine_ref, batch_size, seed_b)

        # EUROPEAN CALL payoff on the fine grid (EM, consistent with exp3)
        X_ref, _ = _em_path_from_increments(dW1, dW2, T / n_fine_ref)
        S_ref    = np.exp(X_ref)
        payoff_ref = np.maximum(S_ref - K, 0.0)

        for n in n_steps_list:
            # --- EM ---
            t0 = time.perf_counter()
            Xe, _ = _em_coarse_from_fine(dW1, dW2, n)
            em_times[n] += time.perf_counter() - t0

            payoff_e = np.maximum(np.exp(Xe) - K, 0.0)
            de = payoff_e - payoff_ref
            em_sum[n] += de.sum()
            em_sq[n]  += (de * de).sum()

            # --- Milstein ---
            t0 = time.perf_counter()
            Xm, _ = _milstein_coarse_from_fine(dW1, dW2, n)
            mil_times[n] += time.perf_counter() - t0

            payoff_m = np.maximum(np.exp(Xm) - K, 0.0)
            dm = payoff_m - payoff_ref
            mil_sum[n] += dm.sum()
            mil_sq[n]  += (dm * dm).sum()

        if (b + 1) % max(1, n_batches // 5) == 0:
            print(f"  batch {b+1:3d}/{n_batches}  "
                  f"({time.perf_counter() - t_start_all:5.1f} s)")

    # --- Compute metrics ---
    results = []
    for n in n_steps_list:
        h = T / n
        for scheme, sum_, sq_, t_ in [
            ("EM",       em_sum[n],  em_sq[n],  em_times[n]),
            ("Milstein", mil_sum[n], mil_sq[n], mil_times[n]),
        ]:
            mean_diff = sum_ / n_paths
            var = max(sq_ / n_paths - mean_diff * mean_diff, 0.0)
            se  = np.sqrt(var / n_paths)
            bias = abs(mean_diff)
            rmse = np.sqrt(bias * bias + se * se)

            iterations = n_paths * n
            # SUPER IMPORTANT!!!!!!! BOYS CHECK THIS : Rough operation count: EM ~ 8 flops/step, Milstein ~ 15.
            ops = n_paths * n * (8 if scheme == "EM" else 15)

            results.append({
                "scheme":      scheme,
                "n_steps":     n,
                "h":           h,
                "bias":        bias,
                "mc_se":       se,
                "rmse":        rmse,
                "wall_time_s": t_,
                "iterations":  iterations,
                "operations":  ops,
            })

    return results


# ============================================================
# 3. PLOTTING
# ============================================================
def plot_cost_accuracy(results, target_rmse=1e-3):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    schemes = ["EM", "Milstein"]
    colors  = {"EM": "steelblue", "Milstein": "darkorange"}
    markers = {"EM": "o", "Milstein": "s"}

    panels = [
        ("wall_time_s", "Wall-clock time (s)"),
        ("iterations",  "Iteration count (n_paths * n_steps)"),
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
        ax.set_ylabel("RMSE")
        ax.set_title(f"Cost vs. Accuracy\n({cost_label})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "exp4_cost_accuracy.png")
    fig.savefig(out, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved {out}\n")


# ============================================================
# 4. TABLE
# ============================================================
def print_cost_accuracy_table(results):
    lines = []
    lines.append("=" * 100)
    lines.append("COST-ACCURACY SUMMARY TABLE")
    lines.append("=" * 100)
    lines.append(
        f"{'Scheme':10s} {'n_steps':>8s} {'h':>10s} "
        f"{'Bias':>12s} {'MC SE':>12s} {'RMSE':>12s} "
        f"{'Wall(s)':>10s} {'Iters':>14s} {'Ops':>14s}"
    )
    lines.append("-" * 100)

    for r in sorted(results, key=lambda r: (r["scheme"], r["n_steps"])):
        lines.append(
            f"{r['scheme']:10s} {r['n_steps']:8d} {r['h']:10.5f} "
            f"{r['bias']:12.3e} {r['mc_se']:12.3e} {r['rmse']:12.3e} "
            f"{r['wall_time_s']:10.3f} {r['iterations']:14,d} "
            f"{r['operations']:14,d}"
        )

    lines.append("=" * 100)
    lines.append("")
    lines.append("NOT COUNTED in the above:")
    lines.append("  - Reference solution generation (n_fine_ref = 8192)")
    lines.append("  - Brownian increment generation (_fine_increments)")
    lines.append("  - Memory allocation, GC, interpreter startup")
    lines.append("  - NumPy vectorisation efficiency differences between schemes")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "exp4_cost_accuracy_table.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {out_path}\n")


# ============================================================
# 5. CONCLUSION
# ============================================================
def state_conclusion(results):
    n_target = 32
    em  = next((r for r in results
                if r["scheme"] == "EM" and r["n_steps"] == n_target), None)
    mil = next((r for r in results
                if r["scheme"] == "Milstein" and r["n_steps"] == n_target), None)
    if em is None or mil is None:
        print("(conclusion skipped: n_steps=32 not in results)\n")
        return

    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"At n_steps = {n_target} (h = {em['h']:.4f}):")
    print(f"  EM:       bias = {em['bias']:.3e},  RMSE = {em['rmse']:.3e},  "
          f"wall = {em['wall_time_s']:.3f}s")
    print(f"  Milstein: bias = {mil['bias']:.3e},  RMSE = {mil['rmse']:.3e},  "
          f"wall = {mil['wall_time_s']:.3f}s")
    print()

    if mil["rmse"] < em["rmse"] and mil["wall_time_s"] < em["wall_time_s"]:
        print("  Milstein is MORE efficient (lower RMSE and lower cost).")
    elif mil["rmse"] < em["rmse"]:
        print("  Milstein achieves lower RMSE but at higher cost.")
        print("  For a matched accuracy target, EM may be more efficient.")
    else:
        print("  EM is MORE efficient (lower or equal RMSE at lower cost).")
        print("  This is a NEGATIVE result for Milstein: despite its higher")
        print("  strong order, the weak order is the same (1.0) and the")
        print("  extra per-step cost does not pay off for this European payoff.")
    print()


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NASV COST-ACCURACY ANALYSIS — Experiment 4")
    print("=" * 60)
    print(f"Parameters: S0={S0}, Y0={Y0}, mu={mu}, kappa={kappa}")
    print(f"            theta={theta}, xi={xi}, rho={rho}")
    print(f"            sigma_min={sigma_min}, sigma_max={sigma_max}, T={T}")
    print()

    results = cost_accuracy_experiment(
        n_steps_list=(4, 8, 16, 32, 64),
        n_paths=50_000,
        batch_size=5_000,
        n_fine_ref=8192,
        K=100.0,
        seed=123,
    )

    print_cost_accuracy_table(results)
    plot_cost_accuracy(results, target_rmse=1e-3)
    state_conclusion(results)

    print("=" * 60)
    print("DONE")
    print("=" * 60)