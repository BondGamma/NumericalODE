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

Design (both strong and weak):
  * ONE fixed fine reference grid  n_fine_ref  chosen once.
  * Fine Brownian increments generated ONCE per batch and reused at every
    coarse level (common random numbers / CRN).
  * Coarse paths are obtained by summing the fine increments into blocks.
  * Weak experiment: two references on the SAME Brownian path
    (main + tolerance check). Their difference is pure discretisation.
  * Weak experiment is BATCHED over paths: memory stays small, cache
    stays warm, numpy RNG uses the fast float32 path.

Outputs (saved to repo-root figures/):
  figures/exp3_nasv_convergence.png
  figures/exp3_nasv_distribution_check.png

Run (from repo root):
    python code/experiments/exp3_nasv_convergence.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. MODEL PARAMETERS
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

FIG_DIR = "figures"


# ============================================================
# 2. HELPERS
# ============================================================
def g(y):
    return sigma_min + (sigma_max - sigma_min) / (1.0 + np.exp(-y))


def g_prime(y):
    s = 1.0 / (1.0 + np.exp(-y))
    return (sigma_max - sigma_min) * s * (1.0 - s)


def simulate_nasv(n_steps, n_paths, seed=None):
    """Standalone EM for the distribution check only."""
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

        X[:, i + 1] = Xc + (mu - 0.5 * gY**2) * dt + gY * dW1
        Y[:, i + 1] = Yc + kappa * (theta - Yc) * dt \
                        + xi * np.sqrt(1.0 + Yc**2) * dW2

    return np.exp(X[:, -1]), X, Y


def _fine_increments(n_fine, n_paths, seed, dtype=np.float64):
    """Generate fine Brownian increments once. dtype=float32 for speed."""
    rng = np.random.default_rng(seed)
    dt_fine = T / n_fine
    sq = dtype(T / n_fine) ** dtype(0.5) if False else np.sqrt(np.array(dt_fine, dtype=dtype))
    Z1 = rng.standard_normal((n_paths, n_fine)).astype(dtype, copy=False)
    Z2 = rng.standard_normal((n_paths, n_fine)).astype(dtype, copy=False)
    dW1 = sq * Z1
    dW2 = sq * (dtype(rho) * Z1 + dtype(np.sqrt(1.0 - rho**2)) * Z2)
    return dW1, dW2, dt_fine


def _em_path_from_increments(dW1, dW2, dt, dtype=np.float64):
    """EM on the NASV system from supplied increments. Arithmetic in dtype."""
    n_paths, n_steps = dW1.shape
    X = np.full(n_paths, np.log(S0), dtype=dtype)
    Y = np.full(n_paths, Y0, dtype=dtype)
    dt_d = dtype(dt)
    for i in range(n_steps):
        gY = g(Y).astype(dtype, copy=False)
        X = X + (dtype(mu) - dtype(0.5) * gY * gY) * dt_d + gY * dW1[:, i]
        Y = Y + dtype(kappa) * (dtype(theta) - Y) * dt_d \
              + dtype(xi) * np.sqrt(dtype(1.0) + Y * Y) * dW2[:, i]
    return X, Y


def _em_coarse_from_fine(dW1_fine, dW2_fine, n_coarse, dtype=np.float64):
    """Coarse EM path driven by the SAME fine increments, summed into blocks."""
    n_fine = dW1_fine.shape[1]
    if n_fine % n_coarse != 0:
        raise ValueError(f"n_fine={n_fine} must be a multiple of n={n_coarse}")
    ref = n_fine // n_coarse
    dt_coarse = T / n_coarse

    n_paths = dW1_fine.shape[0]
    X = np.full(n_paths, np.log(S0), dtype=dtype)
    Y = np.full(n_paths, Y0, dtype=dtype)

    for j in range(n_coarse):
        s = j * ref
        e = (j + 1) * ref
        dW1_sum = dW1_fine[:, s:e].sum(axis=1)
        dW2_sum = dW2_fine[:, s:e].sum(axis=1)

        gY = g(Y).astype(dtype, copy=False)
        X = X + (dtype(mu) - dtype(0.5) * gY * gY) * dtype(dt_coarse) + gY * dW1_sum
        Y = Y + dtype(kappa) * (dtype(theta) - Y) * dtype(dt_coarse) \
              + dtype(xi) * np.sqrt(dtype(1.0) + Y * Y) * dW2_sum

    return X, Y


# ============================================================
# 3. STRONG CONVERGENCE (unchanged design)
# ============================================================
def strong_convergence_experiment(
    n_steps_list=(4, 8, 16, 32, 64),
    n_paths=2000,
    n_fine_ref=8192,
    seed=42,
):
    print("=" * 60)
    print("STRONG CONVERGENCE (fixed reference, CRN)")
    print("=" * 60)

    dW1_fine, dW2_fine, dt_fine = _fine_increments(n_fine_ref, n_paths, seed)

    X_ref, _ = _em_path_from_increments(dW1_fine, dW2_fine, dt_fine)
    S_ref = np.exp(X_ref)
    print(f"  Reference grid  n_fine_ref = {n_fine_ref}  "
          f"(refinement = {n_fine_ref // max(n_steps_list)}x coarsest)")

    hs, errors, ci_lower, ci_upper = [], [], [], []

    for n in n_steps_list:
        if n_fine_ref % n != 0:
            raise ValueError(f"n_fine_ref={n_fine_ref} must be a multiple of n={n}")
        h = T / n
        Xc, _ = _em_coarse_from_fine(dW1_fine, dW2_fine, n)
        S_coarse = np.exp(Xc)

        abs_diff = np.abs(S_coarse - S_ref)
        mean_err = float(np.mean(abs_diff))
        se = float(np.std(abs_diff, ddof=1) / np.sqrt(n_paths))
        z = 1.96
        lo, hi = mean_err - z * se, mean_err + z * se

        hs.append(h); errors.append(mean_err)
        ci_lower.append(lo); ci_upper.append(hi)

        print(f"  n_steps = {n:4d}  h = {h:.6f}  "
              f"E|Δ| = {mean_err:.6f}  95% CI = [{lo:.6f}, {hi:.6f}]")

    slope, _ = np.polyfit(np.log(hs), np.log(errors), 1)
    print()
    print(f"  Estimated strong order (slope) = {slope:.3f}    "
          f"(expected ≈ 0.5 for Euler–Maruyama)")
    print()

    return {
        "h": np.array(hs),
        "strong_error": np.array(errors),
        "ci_lower": np.array(ci_lower),
        "ci_upper": np.array(ci_upper),
        "slope": float(slope),
    }


# ============================================================
# 4. WEAK CONVERGENCE (batched + float32)
# ============================================================
def _weak_batch(batch_size, n_steps_list, n_fine_ref, n_fine_ref_check,
                K, seed_batch, dtype):
    """Run one batch of the weak experiment. Returns per-path payoffs."""
    # Generate the Brownian path at the check grid — this is the ONLY RNG.
    dW1_ck, dW2_ck, _ = _fine_increments(n_fine_ref_check, batch_size,
                                          seed_batch, dtype=dtype)

    # Reference A: EM at the check grid
    X_A, _ = _em_path_from_increments(dW1_ck, dW2_ck, T / n_fine_ref_check,
                                       dtype=dtype)
    payoff_A = np.maximum(np.exp(X_A.astype(np.float64, copy=False)) - K, 0.0)

    # Downsample to the main grid (contiguous pair/block sums)
    ratio = n_fine_ref_check // n_fine_ref
    dW1_lo = dW1_ck.reshape(batch_size, n_fine_ref, ratio).sum(axis=2)
    dW2_lo = dW2_ck.reshape(batch_size, n_fine_ref, ratio).sum(axis=2)
    del dW1_ck, dW2_ck

    # Reference B: EM at the main grid, same Brownian path
    X_B, _ = _em_path_from_increments(dW1_lo, dW2_lo, T / n_fine_ref,
                                       dtype=dtype)
    payoff_B = np.maximum(np.exp(X_B.astype(np.float64, copy=False)) - K, 0.0)

    # Coarse levels reuse dW_lo
    coarse = {}
    for n in n_steps_list:
        Xc, _ = _em_coarse_from_fine(dW1_lo, dW2_lo, n, dtype=dtype)
        coarse[n] = np.maximum(np.exp(Xc.astype(np.float64, copy=False)) - K, 0.0)

    return payoff_A, payoff_B, coarse


def weak_convergence_experiment(
    n_steps_list=(4, 8, 16, 32, 64),
    n_paths=50_000,
    batch_size=5_000,
    n_fine_ref=2048,
    n_fine_ref_check=4096,
    K=100.0,
    seed=123,
    dtype=np.float32,
):
    """Weak error of the European call payoff  E[(S_T - K)^+].

    Batched + float32 for speed. CRN and reference-agreement logic are
    identical to the unbunched version.
    """
    if n_fine_ref_check % n_fine_ref != 0:
        raise ValueError("n_fine_ref_check must be a multiple of n_fine_ref")
    if n_paths % batch_size != 0:
        raise ValueError("n_paths must be a multiple of batch_size")
    n_batches = n_paths // batch_size

    print("=" * 60)
    print("WEAK CONVERGENCE (Call Option Payoff, batched + float32)")
    print("=" * 60)
    print(f"  Paths = {n_paths:,}   batch_size = {batch_size:,}   "
          f"batches = {n_batches}")
    print(f"  n_fine_ref = {n_fine_ref}   n_fine_ref_check = {n_fine_ref_check}   "
          f"dtype = {np.dtype(dtype).name}")
    print()

    t_start = time.perf_counter()

    # Accumulators
    payoffs_A = np.empty(n_paths, dtype=np.float64)
    payoffs_B = np.empty(n_paths, dtype=np.float64)
    coarse_all = {n: np.empty(n_paths, dtype=np.float64) for n in n_steps_list}

    for b in range(n_batches):
        seed_b = seed + b
        pA, pB, coarse = _weak_batch(
            batch_size, n_steps_list, n_fine_ref, n_fine_ref_check,
            K, seed_b, dtype,
        )
        s = b * batch_size
        e = s + batch_size
        payoffs_A[s:e] = pA
        payoffs_B[s:e] = pB
        for n in n_steps_list:
            coarse_all[n][s:e] = coarse[n]

        print(f"  batch {b + 1:2d}/{n_batches}  "
              f"E[payoff_hi] = {pA.mean():.4f}  "
              f"E[payoff_lo] = {pB.mean():.4f}  "
              f"({time.perf_counter() - t_start:5.1f} s elapsed)")

    ref_hi = float(payoffs_A.mean())
    ref_lo = float(payoffs_B.mean())
    tol = abs(ref_hi - ref_lo)
    n_digits = int(np.floor(-np.log10(tol))) if tol > 0 else 15

    print()
    print(f"  Tight reference   n_ref = {n_fine_ref_check:6d}   "
          f"E[payoff_ref] = {ref_hi:.6f}")
    print(f"  Main reference    n_ref = {n_fine_ref:6d}   "
          f"E[payoff_ref] = {ref_lo:.6f}")
    print(f"  Reference agreement |Δ| = {tol:.3e}  →  "
          f"report at most {n_digits} fractional digits")
    print()

    payoff_ref = payoffs_A        # the tighter one is the truth
    reference_payoff = ref_hi

    hs, payoffs, biases = [], [], []
    bias_lo, bias_hi = [], []

    for n in n_steps_list:
        h = T / n
        pc = coarse_all[n]
        diff = pc - payoff_ref
        mean_diff = float(diff.mean())
        se_diff = float(diff.std(ddof=1) / np.sqrt(n_paths))
        z = 1.96
        lo, hi = mean_diff - z * se_diff, mean_diff + z * se_diff

        hs.append(h)
        payoffs.append(float(pc.mean()))
        biases.append(abs(mean_diff))
        bias_lo.append(min(abs(lo), abs(hi)))
        bias_hi.append(max(abs(lo), abs(hi)))

        print(f"  n_steps = {n:4d}  h = {h:.6f}  "
              f"E[payoff] = {payoffs[-1]:.6f}  "
              f"|bias| = {abs(mean_diff):.3e}  "
              f"95% CI = [{min(abs(lo), abs(hi)):.3e}, "
              f"{max(abs(lo), abs(hi)):.3e}]")

    hs = np.array(hs)
    biases_arr = np.array(biases)
    slope, _ = np.polyfit(np.log(hs), np.log(biases_arr), 1)

    print()
    print(f"  Estimated weak order (slope) = {slope:.3f}    "
          f"(expected ≈ 1.0 for Euler–Maruyama)")
    print(f"  Total wall time = {time.perf_counter() - t_start:.1f} s")
    print()

    return {
        "h": hs,
        "payoff": np.array(payoffs),
        "bias": biases_arr,
        "bias_ci_lower": np.array(bias_lo),
        "bias_ci_upper": np.array(bias_hi),
        "reference_payoff": reference_payoff,
        "reference_digits": n_digits,
        "slope": float(slope),
    }


# ============================================================
# 5. DISTRIBUTION CHECK
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

    axes[0].hist(S, bins=80, density=True, alpha=0.6, color="steelblue",
                 label="NASV (EM)")
    x = np.linspace(0.01, np.percentile(S, 99.5), 300)
    pdf_gbm = stats.lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))
    axes[0].plot(x, pdf_gbm, "r-", lw=2,
                 label=f"GBM benchmark (σ_eff={sigma_eff:.3f})")
    axes[0].set_xlabel("S_T"); axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of S_T at T = 1")
    axes[0].legend(); axes[0].set_xlim(0, np.percentile(S, 99.5))

    S_sorted = np.sort(S)
    probs = np.linspace(0.001, 0.999, 5000)
    theoretical_q = stats.lognorm.ppf(probs, s=sigma_log, scale=np.exp(mu_log))
    sample_q = np.quantile(S_sorted, probs)

    axes[1].scatter(theoretical_q, sample_q, s=2, alpha=0.5, color="steelblue")
    max_val = max(theoretical_q.max(), sample_q.max())
    axes[1].plot([0, max_val], [0, max_val], "r--", lw=1, label="y = x")
    axes[1].set_xlabel("Theoretical lognormal quantiles")
    axes[1].set_ylabel("Sample NASV quantiles")
    axes[1].set_title("Q–Q plot vs. lognormal benchmark")
    axes[1].legend()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp3_nasv_distribution_check.png"),
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
# 6. PLOTS
# ============================================================
def plot_convergence(strong_res, weak_res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    h, err = strong_res["h"], strong_res["strong_error"]
    ax.loglog(h, err, "o-", color="steelblue", lw=2, ms=8,
              label="EM strong error")
    ax.fill_between(h, strong_res["ci_lower"], strong_res["ci_upper"],
                    alpha=0.3, color="steelblue", label="95% CI")
    h_ref = np.array([h.min(), h.max()])
    ax.loglog(h_ref, 0.5 * (h_ref / h_ref[0]) ** 0.5 * err[0],
              "k--", lw=1.5, label="slope 0.5")
    ax.loglog(h_ref, 1.0 * (h_ref / h_ref[0]) ** 1.0 * err[0],
              "k:", lw=1.5, label="slope 1.0")
    ax.set_xlabel("Time step h")
    ax.set_ylabel(r"$E|S_N^{(h)} - S(T)|$")
    ax.set_title(f'Strong convergence (slope ≈ {strong_res["slope"]:.3f})')
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    h, bias = weak_res["h"], weak_res["bias"]
    ax.loglog(h, bias, "s-", color="darkorange", lw=2, ms=8,
              label=r"$|E[\mathrm{payoff}(h)] - \mathrm{ref}|$")
    ax.fill_between(h, weak_res["bias_ci_lower"], weak_res["bias_ci_upper"],
                    alpha=0.3, color="darkorange", label="95% CI on bias")
    ax.loglog(h_ref, 1.0 * (h_ref / h_ref[0]) ** 1.0 * bias[0],
              "k--", lw=1.5, label="slope 1.0")
    ax.set_xlabel("Time step h")
    ax.set_ylabel("Weak bias")
    ax.set_title(f'Weak convergence (slope ≈ {weak_res["slope"]:.3f})')
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exp3_nasv_convergence.png"), dpi=150)
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
        n_fine_ref=8192,
        seed=42,
    )

    weak_res = weak_convergence_experiment(
        n_steps_list=(4, 8, 16, 32, 64),
        n_paths=50_000,
        batch_size=5_000,
        n_fine_ref=2048,
        n_fine_ref_check=4096,
        K=100.0,
        seed=123,
    )

    plot_convergence(strong_res, weak_res)
    distribution_check(n_steps=256, n_paths=100000, seed=7)

    print("=" * 60)
    print("DONE")
    print("=" * 60)