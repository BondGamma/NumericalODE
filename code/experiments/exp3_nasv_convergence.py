"""
Experiment 3 — NASV Convergence Analysis
Direction ④: Non-Affine Stochastic Volatility Model
====================================================
Model:
    dX_t = (mu - 0.5 * g(Y_t)^2) dt + g(Y_t) dW_t^(1)
    dY_t = kappa (theta - Y_t) dt + xi * sqrt(1 + Y_t^2) dW_t^(2)
    g(y) = sigma_min + (sigma_max - sigma_min) / (1 + exp(-y))
    d<W^(1), W^(2)>_t = rho dt

We apply Euler–Maruyama to (X, Y) and recover S = exp(X).

Experiments:
  1. Strong convergence: E|S_N^{(h)} - S_N^{(h_ref)}| vs h, coupled paths
  2. Weak convergence:   E[(S_T - K)^+] vs h
  3. Distribution check at T: histogram + Q–Q vs. GBM benchmark

Outputs (saved to repo-root figures/, same convention as exp1):
  - figures/exp3_nasv_convergence.png
  - figures/exp3_nasv_distribution_check.png

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. MODEL PARAMETERS (reproducible benchmark from the brief)
# ============================================================
S0        = 100.0
Y0        = 0.0
mu        = 0.05
kappa     = 2.0
theta     = -0.2
xi        = 0.6
rho       = -0.7
sigma_min = 0.10
sigma_max = 0.50
T         = 1.0

FIG_DIR = 'figures'

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def g(y):
    """Bounded logistic volatility function g(y)."""
    return sigma_min + (sigma_max - sigma_min) / (1.0 + np.exp(-y))


def g_prime(y):
    """Derivative g'(y) — diagnostics only, not needed for EM."""
    s = 1.0 / (1.0 + np.exp(-y))
    return (sigma_max - sigma_min) * s * (1.0 - s)


def simulate_nasv(n_steps, n_paths, seed=None):
    """Euler–Maruyama for the NASV system (single resolution)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    X = np.zeros((n_paths, n_steps + 1))
    Y = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = np.log(S0)
    Y[:, 0] = Y0

    for i in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2)

        Xc = X[:, i]
        Yc = Y[:, i]
        gY = g(Yc)

        drift_X = mu - 0.5 * gY**2
        diff_X  = gY
        drift_Y = kappa * (theta - Yc)
        diff_Y  = xi * np.sqrt(1.0 + Yc**2)

        X[:, i + 1] = Xc + drift_X * dt + diff_X * dW1
        Y[:, i + 1] = Yc + drift_Y * dt + diff_Y * dW2

    return np.exp(X[:, -1]), X, Y


def simulate_nasv_coupled(n_coarse, n_paths, refinement=32, seed=None):
    """Coupled coarse/fine EM paths sharing the same Brownian increments."""
    rng = np.random.default_rng(seed)

    n_fine = n_coarse * refinement
    dt_coarse = T / n_coarse
    dt_fine   = T / n_fine
    sqrt_dt_f = np.sqrt(dt_fine)

    Z1_fine = rng.standard_normal((n_paths, n_fine))
    Z2_fine = rng.standard_normal((n_paths, n_fine))

    dW1_fine = sqrt_dt_f * Z1_fine
    dW2_fine = sqrt_dt_f * (rho * Z1_fine + np.sqrt(1.0 - rho**2) * Z2_fine)

    # ---- Fine reference path ----
    X_fine = np.full(n_paths, np.log(S0))
    Y_fine = np.full(n_paths, Y0)

    for i in range(n_fine):
        gY = g(Y_fine)
        drift_X = mu - 0.5 * gY**2
        diff_X  = gY
        drift_Y = kappa * (theta - Y_fine)
        diff_Y  = xi * np.sqrt(1.0 + Y_fine**2)

        X_fine = X_fine + drift_X * dt_fine + diff_X * dW1_fine[:, i]
        Y_fine = Y_fine + drift_Y * dt_fine + diff_Y * dW2_fine[:, i]

    S_fine = np.exp(X_fine)

    # ---- Coarse path driven by summed fine increments ----
    X_coarse = np.full(n_paths, np.log(S0))
    Y_coarse = np.full(n_paths, Y0)

    for j in range(n_coarse):
        start = j * refinement
        end   = (j + 1) * refinement
        dW1_sum = np.sum(dW1_fine[:, start:end], axis=1)
        dW2_sum = np.sum(dW2_fine[:, start:end], axis=1)

        gY = g(Y_coarse)
        drift_X = mu - 0.5 * gY**2
        diff_X  = gY
        drift_Y = kappa * (theta - Y_coarse)
        diff_Y  = xi * np.sqrt(1.0 + Y_coarse**2)

        X_coarse = X_coarse + drift_X * dt_coarse + diff_X * dW1_sum
        Y_coarse = Y_coarse + drift_Y * dt_coarse + diff_Y * dW2_sum

    return np.exp(X_coarse), S_fine


# ============================================================
# 3. STRONG CONVERGENCE
# ============================================================
def strong_convergence_experiment(
    n_steps_list=(4, 8, 16, 32, 64),
    n_paths=2000,
    refinement=32,
    seed=42,
):
    print("=" * 60)
    print("STRONG CONVERGENCE")
    print("=" * 60)

    hs, errors, ci_lower, ci_upper = [], [], [], []

    for n in n_steps_list:
        h = T / n
        S_coarse, S_fine = simulate_nasv_coupled(
            n_coarse=n, n_paths=n_paths, refinement=refinement, seed=seed
        )
        abs_diff = np.abs(S_coarse - S_fine)
        mean_err = float(np.mean(abs_diff))
        se = float(np.std(abs_diff, ddof=1) / np.sqrt(n_paths))
        z = 1.96
        lo, hi = mean_err - z * se, mean_err + z * se

        hs.append(h); errors.append(mean_err)
        ci_lower.append(lo); ci_upper.append(hi)

        print(f"  n_steps = {n:4d}  h = {h:.6f}  "
              f"E|Δ| = {mean_err:.6f}  95% CI = [{lo:.6f}, {hi:.6f}]")

    slope, _ = np.polyfit(np.log(hs), np.log(errors), 1)
    print(f"\n  Estimated strong order (slope) = {slope:.3f}")
    print(f"  (Expected ≈ 0.5 for Euler–Maruyama)\n")

    return {
        'h': np.array(hs),
        'strong_error': np.array(errors),
        'ci_lower': np.array(ci_lower),
        'ci_upper': np.array(ci_upper),
        'slope': float(slope),
    }


# ============================================================
# 4. WEAK CONVERGENCE
# ============================================================
def weak_convergence_experiment(
    n_steps_list=(4, 8, 16, 32, 64),
    n_paths=50000,
    refinement=32,
    K=100.0,
    seed=123,
):
    print("=" * 60)
    print("WEAK CONVERGENCE (Call Option Payoff)")
    print("=" * 60)

    hs, payoffs, ci_lower, ci_upper = [], [], [], []
    reference_payoff = None

    for n in n_steps_list:
        h = T / n
        S_coarse, S_fine = simulate_nasv_coupled(
            n_coarse=n, n_paths=n_paths, refinement=refinement, seed=seed
        )
        payoff_coarse = np.maximum(S_coarse - K, 0.0)
        payoff_fine   = np.maximum(S_fine   - K, 0.0)

        mean_payoff = float(np.mean(payoff_coarse))
        se = float(np.std(payoff_coarse, ddof=1) / np.sqrt(n_paths))
        z = 1.96
        lo, hi = mean_payoff - z * se, mean_payoff + z * se

        hs.append(h); payoffs.append(mean_payoff)
        ci_lower.append(lo); ci_upper.append(hi)

        if reference_payoff is None:
            reference_payoff = float(np.mean(payoff_fine))

        print(f"  n_steps = {n:4d}  h = {h:.6f}  "
              f"E[payoff] = {mean_payoff:.6f}  95% CI = [{lo:.6f}, {hi:.6f}]")

    bias = np.abs(np.array(payoffs) - reference_payoff)
    slope, _ = np.polyfit(np.log(hs), np.log(bias), 1)

    print(f"\n  Reference payoff (fine EM) = {reference_payoff:.6f}")
    print(f"  Estimated weak order (slope) = {slope:.3f}")
    print(f"  (Expected ≈ 1.0 for Euler–Maruyama)\n")

    return {
        'h': np.array(hs),
        'payoff': np.array(payoffs),
        'ci_lower': np.array(ci_lower),
        'ci_upper': np.array(ci_upper),
        'reference_payoff': reference_payoff,
        'slope': float(slope),
    }


# ============================================================
# 5. DISTRIBUTION CHECK AT T
# ============================================================
def distribution_check(n_steps=256, n_paths=100000, seed=7):
    print("=" * 60)
    print("DISTRIBUTION CHECK AT T=1")
    print("=" * 60)

    S, _, Y_paths = simulate_nasv(n_steps=n_steps, n_paths=n_paths, seed=seed)

    gY_T = g(Y_paths[:, -1])
    sigma_eff = float(np.mean(gY_T))
    mu_log    = np.log(S0) + (mu - 0.5 * sigma_eff**2) * T
    sigma_log = sigma_eff * np.sqrt(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Histogram ----
    axes[0].hist(S, bins=80, density=True, alpha=0.6, color='steelblue',
                 label='NASV (EM)')
    x = np.linspace(0.01, np.percentile(S, 99.5), 300)
    pdf_gbm = stats.lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))
    axes[0].plot(x, pdf_gbm, 'r-', lw=2,
                 label=f'GBM benchmark (σ_eff={sigma_eff:.3f})')
    axes[0].set_xlabel('S_T')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of S_T at T = 1')
    axes[0].legend()
    axes[0].set_xlim(0, np.percentile(S, 99.5))

    # ---- Q–Q plot ----
    S_sorted = np.sort(S)
    probs = np.linspace(0.001, 0.999, 5000)
    theoretical_q = stats.lognorm.ppf(probs, s=sigma_log, scale=np.exp(mu_log))
    sample_q = np.quantile(S_sorted, probs)

    axes[1].scatter(theoretical_q, sample_q, s=2, alpha=0.5, color='steelblue')
    max_val = max(theoretical_q.max(), sample_q.max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', lw=1, label='y = x')
    axes[1].set_xlabel('Theoretical lognormal quantiles')
    axes[1].set_ylabel('Sample NASV quantiles')
    axes[1].set_title('Q–Q plot vs. lognormal benchmark')
    axes[1].legend()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp3_nasv_distribution_check.png'),
                dpi=150)
    plt.show()
    print(f"Saved {FIG_DIR}/exp3_nasv_distribution_check.png\n")

    print("Sanity checks")
    print(f"  Sample mean S_T   = {np.mean(S):.4f}")
    print(f"  GBM E[S_T]        = {S0 * np.exp(mu * T):.4f}")
    print(f"  Sample std S_T    = {np.std(S, ddof=1):.4f}")
    print(f"  Fraction S_T < 0  = {np.mean(S < 0):.6f}")
    print(f"  Fraction S_T > 0  = {np.mean(S > 0):.6f}\n")


# ============================================================
# 6. CONVERGENCE PLOTS
# ============================================================
def plot_convergence(strong_res, weak_res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Strong ----
    ax = axes[0]
    h   = strong_res['h']
    err = strong_res['strong_error']
    lo  = strong_res['ci_lower']
    hi  = strong_res['ci_upper']

    ax.loglog(h, err, 'o-', color='steelblue', lw=2, ms=8,
              label='EM strong error')
    ax.fill_between(h, lo, hi, alpha=0.3, color='steelblue', label='95% CI')

    h_ref = np.array([h.min(), h.max()])
    ax.loglog(h_ref, 0.5 * (h_ref / h_ref[0])**0.5 * err[0],
              'k--', lw=1.5, label='slope 0.5')
    ax.loglog(h_ref, 1.0 * (h_ref / h_ref[0])**1.0 * err[0],
              'k:', lw=1.5, label='slope 1.0')

    ax.set_xlabel('Time step h')
    ax.set_ylabel(r'$E|S_N^{(h)} - S(T)|$')
    ax.set_title(f'Strong convergence (slope ≈ {strong_res["slope"]:.3f})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # ---- Weak ----
    ax = axes[1]
    h    = weak_res['h']
    ref  = weak_res['reference_payoff']
    bias = np.abs(weak_res['payoff'] - ref)

    ax.loglog(h, bias, 's-', color='darkorange', lw=2, ms=8,
              label='|E[payoff(h)] - ref|')
    ax.fill_between(h,
                    np.abs(weak_res['ci_lower'] - ref),
                    np.abs(weak_res['ci_upper'] - ref),
                    alpha=0.3, color='darkorange', label='95% CI (bias)')

    ax.loglog(h_ref, 1.0 * (h_ref / h_ref[0])**1.0 * bias[0],
              'k--', lw=1.5, label='slope 1.0')

    ax.set_xlabel('Time step h')
    ax.set_ylabel('Weak bias')
    ax.set_title(f'Weak convergence (slope ≈ {weak_res["slope"]:.3f})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp3_nasv_convergence.png'), dpi=150)
    plt.show()
    print(f"Saved {FIG_DIR}/exp3_nasv_convergence.png\n")


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NASV CONVERGENCE ANALYSIS — Experiment 3")
    print("=" * 60)
    print(f"Parameters: S0={S0}, Y0={Y0}, mu={mu}, kappa={kappa}")
    print(f"            theta={theta}, xi={xi}, rho={rho}")
    print(f"            sigma_min={sigma_min}, sigma_max={sigma_max}, T={T}")
    print()

    strong_res = strong_convergence_experiment(
        n_steps_list=(4, 8, 16, 32, 64),
        n_paths=2000,
        refinement=32,
        seed=42,
    )

    weak_res = weak_convergence_experiment(
        n_steps_list=(4, 8, 16, 32, 64),
        n_paths=50000,
        refinement=32,
        K=100.0,
        seed=123,
    )

    plot_convergence(strong_res, weak_res)
    distribution_check(n_steps=256, n_paths=100000, seed=7)

    print("=" * 60)
    print("DONE")
    print("=" * 60)