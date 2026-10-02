"""Experiment 1 — GBM convergence analysis"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from code.SDEs.BM_engine import standard_bm, extract_dw

S0, mu, sigma = 1.0, 0.3, 0.4
T = 1.0
n_max = 2**12
n_paths = 100000
seed = 42
K = 1.0


def exact_gbm(S0, mu, sigma, W_T, T):
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)


def exact_logpdf(x, S0, mu, sigma, T):
    m = np.log(S0) + (mu - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    return np.exp(-0.5 * ((np.log(x) - m) / s) ** 2) / (x * s * np.sqrt(2 * np.pi))


def exact_logcdf(x, S0, mu, sigma, T):
    m = np.log(S0) + (mu - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    return norm.cdf((np.log(x) - m) / s)


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


def compute_strong_error(solver_fn, dW_fine, n_coarse_list):
    dts, errs = [], []
    for n in n_coarse_list:
        dt = T / n
        dW = extract_dw(dW_fine, n)
        W_T = dW.sum(axis=1)
        S_exact = exact_gbm(S0, mu, sigma, W_T, T)
        S_num = solver_fn(S0, mu, sigma, dW, dt)
        dts.append(dt)
        errs.append(np.mean(np.abs(S_num - S_exact)))
    return np.array(dts), np.array(errs)


def estimate_order(dts, errs):
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


def exact_moments(S0, mu, sigma, T):
    E1 = S0 * np.exp(mu * T)
    E2 = S0**2 * np.exp(2 * mu * T + sigma**2 * T)
    E3 = S0**3 * np.exp(3 * mu * T + 3 * sigma**2 * T)
    E4 = S0**4 * np.exp(4 * mu * T + 6 * sigma**2 * T)
    return E1, E2, E3, E4


def exact_call(S0, mu, sigma, T, K):
    d1 = (np.log(S0 / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(mu * T) * norm.cdf(d1) - K * norm.cdf(d2)


FUNCTIONALS = {
    "mean": lambda S: S,
    "second moment": lambda S: S**2,
    "call (K=1)": lambda S: np.maximum(S - K, 0.0),
}


def compute_weak_error(solver_fn, dW_fine, n_coarse_list, functionals, exact_vals):
    results = {name: {"dts": [], "errs": []} for name in functionals}
    for n in n_coarse_list:
        dt = T / n
        dW = extract_dw(dW_fine, n)
        S_num = solver_fn(S0, mu, sigma, dW, dt)
        for name, g in functionals.items():
            approx = np.mean(g(S_num))
            err = abs(approx - exact_vals[name])
            results[name]["dts"].append(dt)
            results[name]["errs"].append(err)
    for name in results:
        results[name]["dts"] = np.array(results[name]["dts"])
        results[name]["errs"] = np.array(results[name]["errs"])
    return results


def compute_distribution_error(solver_fn, dW_fine, n_coarse_list):
    dts, ks_stats = [], []
    for n in n_coarse_list:
        dt = T / n
        dW = extract_dw(dW_fine, n)
        S_num = solver_fn(S0, mu, sigma, dW, dt)
        ks, _ = kstest(S_num, lambda x: exact_logcdf(x, S0, mu, sigma, T))
        dts.append(dt)
        ks_stats.append(ks)
    return np.array(dts), np.array(ks_stats)


if __name__ == "__main__":
    dt_fine = T / n_max
    dW_fine = standard_bm(n_max, dt_fine, n_paths=n_paths, seed=seed)
    n_coarse = [2**k for k in range(4, 12)]

    solvers = [("Euler-Maruyama", euler_maruyama_gbm),
               ("Milstein", milstein_gbm)]

    strong_results = {}
    for name, solver in solvers:
        dts, errs = compute_strong_error(solver, dW_fine, n_coarse)
        order = estimate_order(dts, errs)
        strong_results[name] = (dts, errs, order)
        print(f"{name:16s} estimated strong order ~= {order:.3f}")

    E1, E2, E3, E4 = exact_moments(S0, mu, sigma, T)
    exact_vals = {
        "mean": E1,
        "second moment": E2,
        "call (K=1)": exact_call(S0, mu, sigma, T, K),
    }

    weak_results = {}
    for name, solver in solvers:
        weak_results[name] = compute_weak_error(solver, dW_fine, n_coarse, FUNCTIONALS, exact_vals)
        for func_name, data in weak_results[name].items():
            order = estimate_order(data["dts"], data["errs"])
            print(f"{name:16s} weak order for {func_name:14s} ~= {order:.3f}")

    dist_results = {}
    for name, solver in solvers:
        dts, ks = compute_distribution_error(solver, dW_fine, n_coarse)
        dist_results[name] = (dts, ks)
        print(f"{name:16s} final KS statistic at dt={dts[-1]:.5f}: {ks[-1]:.4f}")

    os.makedirs('figures', exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (dts, errs, order) in strong_results.items():
        ax.loglog(dts, errs, 'o-', label=f"{name} (order ~= {order:.2f})")
    dt_ref = np.array([min(dts), max(dts)])
    dt0, err0 = strong_results["Euler-Maruyama"][0][0], strong_results["Euler-Maruyama"][1][0]
    ax.loglog(dt_ref, err0 * np.sqrt(dt_ref / dt0), 'k--', label='slope 0.5')
    ax.loglog(dt_ref, err0 * (dt_ref / dt0), 'k:', label='slope 1.0')
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('Strong error')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/exp1_gbm_strong_convergence.png', dpi=150)

    for func_name in FUNCTIONALS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in weak_results:
            data = weak_results[name][func_name]
            order = estimate_order(data["dts"], data["errs"])
            ax.loglog(data["dts"], data["errs"], 'o-', label=f"{name} (order ~= {order:.2f})")
        dts_ref = weak_results["Euler-Maruyama"][func_name]["dts"]
        errs_ref = weak_results["Euler-Maruyama"][func_name]["errs"]
        ax.loglog(dts_ref, errs_ref[0] * (dts_ref / dts_ref[0]), 'k:', label='slope 1.0')
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel(f'Weak error – {func_name}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fname = f'figures/exp1_gbm_weak_{func_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        fig.savefig(fname, dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (dts, ks) in dist_results.items():
        ax.loglog(dts, ks, 'o-', label=name)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('KS statistic vs exact lognormal')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/exp1_gbm_distribution_ks.png', dpi=150)

    n_fine = n_coarse[-1]
    dt_n = T / n_fine
    dW_n = extract_dw(dW_fine, n_fine)
    S_euler = euler_maruyama_gbm(S0, mu, sigma, dW_n, dt_n)
    S_milstein = milstein_gbm(S0, mu, sigma, dW_n, dt_n)
    W_T = dW_n.sum(axis=1)
    S_exact = exact_gbm(S0, mu, sigma, W_T, T)

    x_grid = np.linspace(0.05, np.percentile(S_exact, 99.5), 400)
    pdf_vals = exact_logpdf(x_grid, S0, mu, sigma, T)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(S_euler, bins=40, density=True, alpha=0.4, label='Euler-Maruyama')
    ax.hist(S_milstein, bins=40, density=True, alpha=0.4, label='Milstein')
    ax.plot(x_grid, pdf_vals, 'k-', lw=2, label='exact lognormal PDF')
    ax.set_xlabel(r'$S_T$')
    ax.set_ylabel('density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/exp1_gbm_distribution_hist.png', dpi=150)
    print("Saved all figures.")