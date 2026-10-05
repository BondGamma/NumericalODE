# Experiment 2 — Positivity-Preserving Analysis of Geometric Brownian Motion

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from code.SDEs.BM_engine import standard_bm, extract_dw
from code.solvers import em, milstein

#Parameters
S0    = 100.0     # initial price
MU    = 0.05      # drift
T     = 1.0       # horizon
SIGMA = 2.0       # volatility 

DT_C  = 1.0 / (SIGMA**2 - 2.0 * MU)   # Milstein positivity 

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)



# Helpers
def p_em(dt):
    dt = np.asarray(dt, dtype=float)
    return norm.cdf(-(1.0 + MU * dt) / (SIGMA * np.sqrt(dt))) #probability per step, like how negative it is


def milstein_roots(dt, mu=MU, sigma=SIGMA):
    d = (sigma**2 - 2.0 * mu) * dt - 1.0
    if d <= 0:
        return np.array([])
    return np.array([(-1.0 - np.sqrt(d)) / (sigma * np.sqrt(dt)),
                     (-1.0 + np.sqrt(d)) / (sigma * np.sqrt(dt))])


def milstein_g(z, dt, mu=MU, sigma=SIGMA):
    return ((0.5 * sigma**2 * dt) * z**2
            + (sigma * np.sqrt(dt)) * z
            + (1.0 + (mu - 0.5 * sigma**2) * dt))


def p_mil_scalar(dt):
#probability at single dt
    d = (SIGMA**2 - 2.0 * MU) * dt - 1.0
    if d <= 0:
        return 0.0
    r1 = (-1.0 - np.sqrt(d)) / (SIGMA * np.sqrt(dt))
    r2 = (-1.0 + np.sqrt(d)) / (SIGMA * np.sqrt(dt))
    return float(norm.cdf(r2) - norm.cdf(r1))


def p_mil_vec(dt):
    #same as above just vectorized
    dt = np.asarray(dt, dtype=float)
    out = np.zeros_like(dt)
    d = (SIGMA**2 - 2.0 * MU) * dt - 1.0
    ok = d > 0
    if np.any(ok):
        r1 = (-1.0 - np.sqrt(d[ok])) / (SIGMA * np.sqrt(dt[ok]))
        r2 = (-1.0 + np.sqrt(d[ok])) / (SIGMA * np.sqrt(dt[ok]))
        out[ok] = norm.cdf(r2) - norm.cdf(r1)
    return out


def eu_put_call(S0_, K, T_, mu, sigma):
    d1 = (np.log(S0_ / K) + (mu + 0.5 * sigma**2) * T_) / (sigma * np.sqrt(T_))
    d2 = d1 - sigma * np.sqrt(T_)
    call = S0_ * np.exp(mu * T_) * norm.cdf(d1) - K * norm.cdf(d2)
    put  = K * norm.cdf(-d2) - S0_ * np.exp(mu * T_) * norm.cdf(-d1)
    return call, put

def run_mc(dt, M, seed):
    n = int(round(T / dt))
    dW = standard_bm(n, dt, M, seed=seed)

    S_em = np.full(M, S0);  ever_em = np.zeros(M, bool)
    S_mi = np.full(M, S0);  ever_mi = np.zeros(M, bool)
    neg_em = 0;  neg_mi = 0

    for k in range(n):
        d = dW[:, k]
        neg_em += int((1.0 + MU * dt + SIGMA * d < 0).sum())
        neg_mi += int((1.0 + (MU - 0.5 * SIGMA**2) * dt
                       + SIGMA * d + 0.5 * SIGMA**2 * d * d < 0).sum())
        S_em = em.em_step("gbm", S_em, dt, d, mu=MU, sigma=SIGMA)
        S_mi = milstein.milstein_step("gbm", S_mi, dt, d, mu=MU, sigma=SIGMA)
        ever_em |= (S_em < 0)
        ever_mi |= (S_mi < 0)

    return dict(em_p=neg_em / (M * n), mil_p=neg_mi / (M * n),
                em_ever=ever_em.mean(), mil_ever=ever_mi.mean())


#Terminal S_T under EM / Milstein / exact on the SAME Brownian increments
def simulate_terminal(dt, M, seed):
    n = int(round(T / dt))
    dW = standard_bm(n, dt, M, seed=seed)
    S_em = np.full(M, S0);  S_mi = np.full(M, S0);  S_ex = np.full(M, S0)
    for k in range(n):
        S_em = em.em_step("gbm", S_em, dt, dW[:, k], mu=MU, sigma=SIGMA)
        S_mi = milstein.milstein_step("gbm", S_mi, dt, dW[:, k], mu=MU, sigma=SIGMA)
        S_ex = S_ex * np.exp((MU - 0.5 * SIGMA**2) * dt + SIGMA * dW[:, k])
    return S_em, S_mi, S_ex


def coarse_run(dW_fine, n):
#generates rough paths for Brownian motion increments
    dW = extract_dw(dW_fine, n)
    S_em = S0;  S_mi = S0
    em_path = [S0];  mi_path = [S0]
    for k in range(n):
        S_em = em.em_step("gbm", S_em, T / n, dW[k], mu=MU, sigma=SIGMA)
        S_mi = milstein.milstein_step("gbm", S_mi, T / n, dW[k], mu=MU, sigma=SIGMA)
        em_path.append(S_em);  mi_path.append(S_mi)
    return np.array(em_path), np.array(mi_path)


#EM FAILURE REGION ON N(0, 1)
def section_em_analysis():
    print("=" * 70)
    print("EM — failure region on the standard-normal density")
    print("=" * 70)
    z = np.linspace(-6, 4, 600)
    pdf = norm.pdf(z)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, dt in zip(axes, [0.25, 0.05]):
        a = (1.0 + MU * dt) / (SIGMA * np.sqrt(dt))
        ax.plot(z, pdf, color="C0", lw=1.8)
        mask = z <= -a
        ax.fill_between(z[mask], 0, pdf[mask], alpha=0.35, color="C3",
                        label=r"negative bracket: $Z\leq -a$")
        ax.axvline(-a, color="C3", ls="--", lw=1)
        ax.set_title(r"$\Delta t=%.2f,\ p_{\mathrm{EM}}=%.4f$"
                     % (dt, norm.cdf(-a)))
        ax.set_xlabel(r"$Z\sim N(0,1)$"); ax.set_ylabel("density")
        ax.set_xlim(-6, 4); ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize="small")

    fig.suptitle("EM failure region on the standard-normal density", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "exp2_em_tail.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}\n")


# MILSTEIN FAILURE REGION
def section_milstein_analysis():
    print("=" * 70)
    print("Milstein — failure region on the parabola and on N(0, 1)")
    print("=" * 70)
    z = np.linspace(-6, 4, 800)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Case (a): dt < dt_c  parabola never crosses zero
    dt_a = 0.25
    ax = axes[0]
    ax.plot(z, milstein_g(z, dt_a), color="C0", lw=2, label="$g(z)$")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.axvline(-1.0 / (SIGMA * np.sqrt(dt_a)), color="C0", ls=":", lw=1)
    ax.set_title(r"Case (a)  $\Delta t=%.2f < \Delta t_c$: never negative" % dt_a)
    ax.set_xlabel("$z$"); ax.set_ylabel("$g(z)$"); ax.set_ylim(-0.05, 3.0)
    ax.grid(True, ls=":", alpha=0.5); ax.legend()

    # Case (b): dt > dt_c parabola dips below 0 between the roots
    dt_b = 0.5
    r1, r2 = milstein_roots(dt_b)
    ax = axes[1]
    ax.plot(z, milstein_g(z, dt_b), color="C1", lw=2, label="$g(z)$")
    ax.axhline(0, color="0.3", lw=0.8)
    mask = (z >= r1) & (z <= r2)
    ax.fill_between(z[mask], 0, milstein_g(z[mask], dt_b), color="C1", alpha=0.35,
                    label=r"negative region $z\in(z_1,z_2)$")
    ax.plot([r1, r2], [0, 0], "ko", ms=5)
    ax.set_title(r"Case (b)  $\Delta t=%.2f > \Delta t_c$" % dt_b)
    ax.set_xlabel("$z$"); ax.set_ylabel("$g(z)$"); ax.set_ylim(-0.3, 2.5)
    ax.grid(True, ls=":", alpha=0.5); ax.legend()

    fig.suptitle(r"Milstein bracket $g(z)$ — upward-opening parabola", y=1.02)
    fig.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "exp2_milstein_parabola.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # Same failure region viewed on N(0, 1)
    z = np.linspace(-6, 4, 600)
    pdf = norm.pdf(z)
    r1, r2 = milstein_roots(0.5)
    p_mil_05 = float(norm.cdf(r2) - norm.cdf(r1))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(z, pdf, color="C0", lw=1.8)
    mask = (z >= r1) & (z <= r2)
    ax.fill_between(z[mask], 0, pdf[mask], color="C1", alpha=0.35,
                    label=r"$z\in(z_1,z_2)$:  $g(z)<0$")
    ax.axvline(r1, color="C1", ls="--", lw=1)
    ax.axvline(r2, color="C1", ls="--", lw=1)
    ax.set_title(r"$\Delta t=0.5$:  $p_{\mathrm{Mil}}=%.4f$" % p_mil_05)
    ax.set_xlabel(r"$Z\sim N(0,1)$"); ax.set_ylabel("density")
    ax.set_xlim(-6, 4); ax.grid(True, ls=":", alpha=0.5); ax.legend()
    fig.tight_layout()
    out2 = os.path.join(FIGURES_DIR, "exp2_milstein_tail.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}\n")


# STRESS-TEST THEORY TABLE
def section_stress_theory():
    print("=" * 70)
    print("Stress-regime theory: sigma^2 - 2 mu = %.3f  ->  dt_c = %.5f"
          % (SIGMA**2 - 2.0 * MU, DT_C))
    print("=" * 70)
    print()
    print("EM per-step negativity  p_EM(dt) = Phi(-(1 + mu dt) / (sigma sqrt(dt)))")
    for dt in [1.0, 0.5, 1/3, 0.25, 0.2, 0.1, 0.05, 0.02]:
        print("  dt = %5.3f   p_EM = %.3e" % (dt, float(p_em(dt))))

    print()
    print("Milstein per-step negativity  p_Mil(dt)   (0 whenever dt <= dt_c)")
    for dt in [1.0, 0.5, 1/3, 0.25, 0.2, 0.1, 0.05, 0.02]:
        print("  dt = %5.3f   p_Mil = %.4f" % (dt, p_mil_scalar(dt)))

    print()
    print("Benchmark sigma = 0.20 for comparison:")
    print("  EM   p_EM(1.0) = %.3e"
          % norm.cdf(-(1.0 + 0.05 * 1.0) / (0.20 * np.sqrt(1.0))))
    print("  Milstein       = 0 for all dt   (sigma^2 - 2 mu = %.2f < 0)"
          % (0.20**2 - 2 * 0.05))
    print()


#EXPERIMENT A — MC NEGATIVITY STATISTICS
def section_mc_tables(M=1_000_000):
    print("=" * 70)
    print(f"Experiment A — Monte-Carlo negativity vs dt  (M = {M:,} paths)")
    print("=" * 70)

    grid_n = [1, 2, 3, 4, 5, 10, 20, 50]
    rows = []
    for n in grid_n:
        dt = 1.0 / n
        r = run_mc(dt, M, seed=123 + n)
        p_em_th = float(p_em(dt))
        p_mi_th = p_mil_scalar(dt)
        ever_em_th = 1.0 - (1.0 - p_em_th) ** n
        ever_mi_th = 1.0 - (1.0 - p_mi_th) ** n
        rows.append((dt, n, p_em_th, r["em_p"], p_mi_th, r["mil_p"],
                     ever_em_th, r["em_ever"], ever_mi_th, r["mil_ever"]))

    print()
    print("Per-step bracket negativity   p(dt)   (theory vs MC):")
    print("  dt        n    EM p(th)   EM p(MC)   Mil p(th)  Mil p(MC)")
    for dt, n, ept, epm, mpt, mpm, *_ in rows:
        print("  %-6.3f  %4d   %.4f     %.4f     %.4f     %.4f"
              % (dt, n, ept, epm, mpt, mpm))

    print()
    print("Path-level P(min S < 0) on [0, T]   (theory vs MC):")
    print("  dt        n    EM (th)    EM (MC)    Mil (th)   Mil (MC)")
    for dt, n, _, _, _, _, eet, eem, met, mem in rows:
        print("  %-6.3f  %4d   %.4f     %.4f     %.4f     %.4f"
              % (dt, n, eet, eem, met, mem))

    print()
    print("(MC std error on p ~ %.1e for p = 0.2)"
          % np.sqrt(0.2 * 0.8 / M))
    print()
    return rows


# EXPERIMENT B — NEGATIVITY PROBABILITY vs chang e in time
def section_negativity_vs_dt(rows):
    print("=" * 70)
    print("Experiment B — negativity probability vs dt (curves + MC markers)")
    print("=" * 70)
    dt_grid = np.array([r[0] for r in rows])
    em_p_mc   = np.array([r[3] for r in rows])
    mil_p_mc  = np.array([r[5] for r in rows])
    em_ev_mc  = np.array([r[7] for r in rows])
    mil_ev_mc = np.array([r[9] for r in rows])

    dt_dense = np.logspace(np.log10(0.015), 0.0, 300)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left per-step bracket negativity
    ax = axes[0]
    ax.semilogx(dt_dense, p_em(dt_dense), color="C0", lw=2,
                label=r"EM  $p_{\mathrm{EM}}$")
    ax.semilogx(dt_dense, p_mil_vec(dt_dense), color="C1", lw=2,
                label=r"Milstein  $p_{\mathrm{Mil}}$")
    ax.semilogx(dt_grid, em_p_mc,  "o", color="C0", ms=6, label="EM (MC)")
    ax.semilogx(dt_grid, mil_p_mc, "s", color="C1", ms=6, label="Milstein (MC)")
    ax.axvline(DT_C, color="0.3", ls="--", lw=1)
    ax.text(DT_C * 1.03, 0.45, r"$\Delta t_c\approx0.256$",
            rotation=90, fontsize=9)
    ax.set_xlabel(r"$\Delta t$ (log)")
    ax.set_ylabel("per-step negativity probability")
    ax.set_title(r"Per-step $p(\Delta t)$")
    ax.legend(loc="lower left"); ax.grid(True, ls=":", alpha=0.5)

    # Right path-level P
    n_dense = 1.0 / dt_dense
    ax = axes[1]
    ax.semilogx(dt_dense, 1.0 - (1.0 - p_em(dt_dense)) ** n_dense,
                color="C0", lw=2, label="EM")
    ax.semilogx(dt_dense, 1.0 - (1.0 - p_mil_vec(dt_dense)) ** n_dense,
                color="C1", lw=2, label="Milstein")
    ax.semilogx(dt_grid, em_ev_mc,  "o", color="C0", ms=6, label="EM (MC)")
    ax.semilogx(dt_grid, mil_ev_mc, "s", color="C1", ms=6, label="Milstein (MC)")
    ax.axvline(DT_C, color="0.3", ls="--", lw=1)
    ax.set_xlabel(r"$\Delta t$ (log)")
    ax.set_ylabel(r"$P(\min S<0)$")
    ax.set_title(r"Path-level: ever negative on $[0,T]$")
    ax.legend(loc="lower left"); ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle(r"Negativity probability vs $\Delta t$  (stress $\sigma=2.0$)",
                 y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "exp2_negativity_vs_dt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}\n")


#EXPERIMENT C FOR PATH WHERE EM NEGATIVE
def section_negative_path():
    print("=" * 70)
    print("Experiment C — a path where EM goes negative but exact/Milstein stay positive")
    print("=" * 70)
    n_fine = 512
    dt_fine = T / n_fine
    n_coarse = 16                 # dt = 0.0625 < dt_c    Milstein stays positive

    best_seed = None
    best_min = np.inf
    best_W = best_em = best_mi = None

    for seed in range(400):
        dW_fine = standard_bm(n_fine, dt_fine, 1, seed=seed)[0]
        W_fine = np.concatenate([[0.0], np.cumsum(dW_fine)])
        em_path, mi_path = coarse_run(dW_fine, n_coarse)
        m = em_path.min()
        if m < best_min:
            best_min = m
            best_seed = seed
            best_W, best_em, best_mi = W_fine, em_path, mi_path

    if best_seed is None or best_min >= 0:
        print("WARNING: no negative EM path found among 400 seeds; skipping plot\n")
        return

    seed, W_fine = best_seed, best_W
    em_path, mi_path = best_em, best_mi

    # Exact path on the same Brownian increments
    t_ex = dt_fine * np.arange(n_fine + 1)
    S_ex = S0 * np.exp((MU - 0.5 * SIGMA**2) * t_ex + SIGMA * W_fine)
    t_co = (T / n_coarse) * np.arange(n_coarse + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_ex, S_ex, color="0.2", lw=1.8, label=r"exact $S(t)$ (positive)")
    ax.plot(t_co, em_path, color="C3", marker="o", ls="-", lw=1.4, ms=4,
            label=r"EM ($\Delta t=0.0625$)")
    ax.plot(t_co, mi_path, color="C0", marker="s", ls="-", lw=1.4, ms=4,
            label=r"Milstein ($\Delta t=0.0625$)")
    ax.axhline(0, color="0.3", lw=0.8)

    neg = em_path < 0
    if neg.any():
        for i in range(n_coarse):
            if neg[i] or neg[i + 1]:
                ax.axvspan(t_co[i], t_co[i + 1], color="C3", alpha=0.15)

    ax.set_xlabel("$t$"); ax.set_ylabel("$S$")
    ax.set_title("EM crosses zero; exact and Milstein stay positive "
                 f"(seed = {seed})")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "exp2_negative_path.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print("seed=%d  |  min EM=%.3f   min Milstein=%.3f   min exact=%.3f"
          % (seed, em_path.min(), mi_path.min(), S_ex.min()))
    print()

# OPTION PRICES — EFFECT OF NEGATIVE SAMPLES
def section_option_prices(M=1_000_000, K=100.0):
    print("=" * 70)
    print("Option prices — isolating the impact of negative samples")
    print("=" * 70)

    call_ref, put_ref = eu_put_call(S0, K, T, MU, SIGMA)
    print("Analytic expectation:  call E[(S_T-K)^+] = %.4f    "
          "put E[(K-S_T)^+] = %.4f\n" % (call_ref, put_ref))

    # --- Put decomposition ------------------------------------------------
    print("PUT — raw vs floored S_T (gap = E[|S_T| 1{S_T<0}] = pure negativity) :")
    print("  dt        n   scheme    P(S_T<0)   put raw   put floor    gap      analytic")
    for dt in [1.0, 0.5, 1/3, 0.25, 0.1]:
        n = int(round(T / dt))
        S_em, S_mi, S_ex = simulate_terminal(dt, M, seed=77)
        for name, S in [("EM", S_em), ("Milstein", S_mi), ("exact", S_ex)]:
            neg       = float((S < 0).mean())
            put_raw   = float(np.maximum(K - S, 0).mean())
            put_floor = float(np.maximum(K - np.maximum(S, 0), 0).mean())
            gap       = put_raw - put_floor
            print("  %-6.3f  %4d   %-8s  %.4f     %.3f      %.3f      %.3f   %.3f"
                  % (dt, n, name, neg, put_raw, put_floor, gap, put_ref))

    print()
    print("CALL — unaffected by flooring (S_T<0 already pays 0); "
          "residual error is pure discretisation:")
    print("  dt        call EM    call Milstein   call exact   call analytic")
    for dt in [1.0, 0.5, 1/3, 0.25, 0.1]:
        S_em, S_mi, S_ex = simulate_terminal(dt, M, seed=78)
        print("  %-6.3f   %.3f     %.3f           %.3f        %.3f"
              % (dt,
                 np.maximum(S_em - K, 0).mean(),
                 np.maximum(S_mi - K, 0).mean(),
                 np.maximum(S_ex - K, 0).mean(),
                 call_ref))
    print()

def main():
    print("=" * 70)
    print("EXPERIMENT 2 — Positivity-preserving analysis of GBM")
    print("=" * 70)
    print(f"Parameters:  S0={S0}, mu={MU}, T={T}, sigma={SIGMA} (stress regime)")
    print(f"Milstein positivity threshold:  dt_c = 1/(sigma^2 - 2 mu) = {DT_C:.5f}")
    print()

    section_em_analysis()
    section_milstein_analysis()
    section_stress_theory()

    rows = section_mc_tables(M=1_000_000)
    section_negativity_vs_dt(rows)

    section_negative_path()
    section_option_prices(M=1_000_000, K=100.0)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()