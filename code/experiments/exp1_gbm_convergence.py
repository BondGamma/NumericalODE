#Experiment 1: Weak convergence of the Euler–Maruyama 

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Parameters
MU = 0.3            # drift
SIGMA = 0.4         # diffusion
T = 1.0             # final time
X0 = 1.0            # initial value
N_PATHS = 200_000   # number of Monte Carlo paths 

# Time step 
DT_VALUES = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])

# Exact solution of GBM
def gbm_exact(X0: float, mu: float, sigma: float, T: float, W_T: np.ndarray) -> np.ndarray:
    return X0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * W_T)


# Euler–Maruyama for GBM
def gbm_euler_maruyama(X0: float, mu: float, sigma: float, T: float, dt: float, dW: np.ndarray) -> np.ndarray:
    
    N_STEPS = int(round(T / dt))
    X = np.full(dW.shape[0], X0, dtype=float)
    for n in range(N_STEPS):
        X += mu * X * dt + sigma * X * dW[:, n]
    return X


# Weak convergence 
def weak_convergence_study() -> Tuple[np.ndarray, np.ndarray]:
    weak_errors = []

    # Analytical mean of the exact GBM
    analytic_mean = X0 * np.exp(MU * T)

    for dt in DT_VALUES:
        N_STEPS = int(round(T / dt))


        #Ref for independent Brownian increments
        dW_ref = np.random.normal(0.0, np.sqrt(dt), size=(N_PATHS, N_STEPS))
        W_T_ref = np.sum(dW_ref, axis=1)
        X_T_ref = gbm_exact(X0, MU, SIGMA, T, W_T_ref)

        # Numerical solution for different increments
        dW_num = np.random.normal(0.0, np.sqrt(dt), size=(N_PATHS, N_STEPS))
        X_T_num = gbm_euler_maruyama(X0, MU, SIGMA, T, dt, dW_num)

        # Monte Carlo estimates of the expectation
        mean_ref = np.mean(X_T_ref)
        mean_num = np.mean(X_T_num)

        
        weak_err = np.abs(analytic_mean - mean_num) #difference of expectations
        weak_errors.append(weak_err)

        print(f"dt = {dt:.6f}  |  E[exact] = {analytic_mean:.6f}  "
              f"|  E[EM] = {mean_num:.6f}  |  weak error = {weak_err:.6e}")

    return DT_VALUES, np.array(weak_errors)

# Main
def main() -> None:
    dt_values, errors = weak_convergence_study()

    #trick from the seminar 
    log_dt = np.log(dt_values)
    log_err = np.log(errors)
    slope, intercept = np.polyfit(log_dt, log_err, 1)

    print(f"\nEstimated weak order (slope) = {slope:.3f}  "
          f"(theoretical order = 1.0)")

    # Magic plotting 
    plt.figure(figsize=(7, 5))
    plt.loglog(dt_values, errors, 'o-', label='Euler–Maruyama')
    plt.loglog(dt_values, np.exp(intercept) * dt_values ** 1.0,
               '--', label='slope 1.0 (theory)')
    plt.xlabel('Time step Δt')
    plt.ylabel('Weak error |E[X_T] – E[X̂_T]|')
    plt.title('Weak convergence of Euler–Maruyama for GBM')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/exp1_weak_convergence.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()