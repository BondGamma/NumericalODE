"""
Experiment 1 — GBM convergence analysis (self-contained).

Measures strong convergence order of Euler-Maruyama and Milstein solvers
for Geometric Brownian Motion against the exact solution.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from code.SDEs.BM_engine import standard_bm, extract_dw

# --- Parameters (match Exp 0 GBM) ---
S0, mu, sigma = 1.0, 0.3, 0.4
T = 1.0
n_max = 2**12          
n_paths = 1000
seed = 42


# --- Exact GBM solution ---
def exact_gbm(S0, mu, sigma, W_T, T):
    """S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*W_T)."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)


# --- Vectorized solvers: dW has shape (n_paths, n_steps) ---
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
    """Mean absolute error E[|S_num(T) - S_exact(T)|] for each dt."""
    dts, errs = [], []
    for n in n_coarse_list:
        dt = T / n
        dW = extract_dw(dW_fine, n)            # (n_paths, n)
        W_T = dW.sum(axis=1)                   # full BM value at T
        S_exact = exact_gbm(S0, mu, sigma, W_T, T)
        S_num = solver_fn(S0, mu, sigma, dW, dt)
        dts.append(dt)
        errs.append(np.mean(np.abs(S_num - S_exact)))
    return np.array(dts), np.array(errs)


def estimate_order(dts, errs):
    """Slope of log(err) vs log(dt)."""
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


if __name__ == "__main__":
    dt_fine = T / n_max
    dW_fine = standard_bm(n_max, dt_fine, n_paths=n_paths, seed=seed)

    n_coarse = [2**k for k in range(4, 12)]   # 16 ... 2048, all divide n_max

    results = {}
    for name, solver in [("Euler-Maruyama", euler_maruyama_gbm),
                         ("Milstein",       milstein_gbm)]:
        dts, errs = compute_strong_error(solver, dW_fine, n_coarse)
        order = estimate_order(dts, errs)
        results[name] = (dts, errs, order)
        print(f"{name:16s}  estimated strong order ~= {order:.3f}")

    #  plot 
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (dts, errs, order) in results.items():
        ax.loglog(dts, errs, 'o-', label=f"{name} (order ~= {order:.2f})")

    # reference slopes anchored to the first EM point (visual guide)
    dt_ref = np.array([min(dts), max(dts)])
    dt0, err0 = results["Euler-Maruyama"][0][0], results["Euler-Maruyama"][1][0]
    ax.loglog(dt_ref, err0 * np.sqrt(dt_ref / dt0), 'k--', label='slope 0.5')
    ax.loglog(dt_ref, err0 * (dt_ref / dt0),        'k:',  label='slope 1.0')

    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('Strong error')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    os.makedirs('figures', exist_ok=True)
    fig.tight_layout()
    fig.savefig('figures/exp1_gbm_convergence.png', dpi=150)
    print("Saved figures/exp1_gbm_convergence.png")