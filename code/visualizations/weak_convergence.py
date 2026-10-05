"""Weak-convergence confidence-interval visualization (Experiment 1, GBM).

Reads ``figures/exp1_gbm_convergence_table.txt`` (produced by
``code/experiments/exp1_gbm_convergence.py``) and separates the deterministic weak
bias of the Euler-Maruyama and Milstein schemes from the Monte-Carlo sampling
noise, using a 95% confidence-interval treatment.

For GBM ``dS = mu S dt + sigma S dW`` both schemes satisfy
``E[S_{k+1} | S_k] = S_k (1 + mu dt)``, so their *exact* weak bias is identical and,
to leading order in the step size,

    E[S_N] = S0 exp(mu T) [1 - 1/2 mu^2 T dt + O(dt^2)],

i.e. the deterministic weak bias shrinks linearly in ``dt``:

    b(dt) = E[S_N] - E[S(T)] = -1/2 S0 exp(mu T) mu^2 T dt + O(dt^2).

Plotting this leading-order curve (rather than the exact ``(1 + mu dt)^n``
expression) keeps the bias a clean power law on the semi-log and log-log axes.

The table's sample means ``S_N`` sit a random offset above this curve; because one
shared Brownian motion drives every step size (the ``extract_dw`` coarsening), that
offset is essentially a single draw of the Monte-Carlo error. Rather than collapse
it to one scalar, the difference ``b_obs(dt) - b(dt)`` is evaluated at every step
size, projected onto the log-log axis and connected; a horizontal line is then
fitted through it (least squares in log space, i.e. the geometric mean of
``|resid|``), and that fitted level is the reported draw, given as a percentage of
``S_T`` ("% draw"). The 95% confidence interval instead uses the *theoretical*
standard deviation of the sample-mean estimator,

    sigma_th  = sqrt(Var(S(T))) / sqrt(n_paths),
    Var(S(T)) = S0^2 exp(2 mu T) (exp(sigma^2 T) - 1),

so the CI half-width ``1.96 * sigma_th`` does not depend on this particular
simulation's realised draw.

Outputs, for each scheme (Euler-Maruyama, Milstein), three figures to
``figures/``:

    * ``exp1_gbm_weak_semilog_<method>.png``  -- weak error on a semi-log
      (log x, linear y) axis, with the 95% CI band and the bias curve;
    * ``exp1_gbm_weak_loglog_<method>.png``   -- |weak error| on a log-log axis,
      with the 95% CI shaded in the scheme's color, the per-dt differences
      connected, and their horizontal fit (the empirical draw);
    * ``exp1_gbm_weak_combined_<method>.png`` -- the two panels stacked, sharing
      the dt x-axis.
"""

import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)
from code.SDEs.gbm import GBM_S0, GBM_MU, GBM_SIGMA, GBM_T  # noqa: E402

FIGURES_DIR = os.path.join(_PROJECT_ROOT, "figures")
TABLE_PATH = os.path.join(FIGURES_DIR, "exp1_gbm_convergence_table.txt")

# Experiment 1 parameters (mirror code/experiments/exp1_gbm_convergence.py).
S0 = GBM_S0
MU = GBM_MU
SIGMA = GBM_SIGMA
T = GBM_T
N_PATHS = 2_000_000

METHODS = ["Euler-Maruyama", "Milstein"]
METHOD_SLUGS = {"Euler-Maruyama": "euler_maruyama", "Milstein": "milstein"}
COLORS = {"Euler-Maruyama": "tab:blue", "Milstein": "tab:orange"}
MARKERS = {"Euler-Maruyama": "o", "Milstein": "s"}
Z = 1.96  # 95% confidence level
NOISE_COLOR = "C3"  # red used for every empirical-noise / %-draw marking

_N_CURVE = 400  # sampling density of the continuous weak-bias curve


def read_table(path=TABLE_PATH):
    """Parse the exp1 table; return (n, dt, S_N_EM, S_N_Mil).

    ``dt`` is recomputed as ``T / n`` (the exact step size the experiment used)
    rather than read from the table column, which is rounded to six decimals.
    """
    data = np.loadtxt(path, skiprows=4, max_rows=8)
    n = data[:, 0].astype(int)
    dt = T / n
    S_N_EM = data[:, 3]
    S_N_Mil = data[:, 6]
    return n, dt, S_N_EM, S_N_Mil


def weak_bias(dt, S0=S0, mu=MU, T=T):
    """Leading-order deterministic weak bias b(dt) = E[S_N] - E[S(T)].

    From ``E[S_N] = S0 exp(mu T) [1 - 1/2 mu^2 T dt + O(dt^2)]`` the signed weak
    bias is negative and linear in ``dt``:

        b(dt) = -1/2 S0 exp(mu T) mu^2 T dt + O(dt^2).
    """
    return -0.5 * S0 * np.exp(mu * T) * mu ** 2 * T * dt


def theoretical_sigma(S0=S0, mu=MU, sigma=SIGMA, T=T, n_paths=N_PATHS):
    """Std of the Monte-Carlo sample mean of S_N, from the exact GBM variance.

    ``Var(S(T)) = S0^2 exp(2 mu T) (exp(sigma^2 T) - 1)``; averaging ``n_paths``
    independent paths reduces the std by ``sqrt(n_paths)``. This -- not the RMS of
    the single shared draw -- is what sets the 95% CI half-width.
    """
    var = S0 ** 2 * np.exp(2.0 * mu * T) * (np.exp(sigma ** 2 * T) - 1.0)
    return float(np.sqrt(var / n_paths))


def empirical_draw(b_obs, b):
    """Per-dt Monte-Carlo difference and its log-log horizontal fit.

    ``b_obs(dt) - b(dt)`` is evaluated at every step size; a single shared
    Brownian motion drives all of them, so they are one MC draw. Projecting the
    absolute differences onto the log-log axis and fitting a horizontal line
    (least squares in log space) gives the draw level as the geometric mean of
    ``|resid|``. Returns ``(resid, draw)``.
    """
    resid = np.asarray(b_obs) - np.asarray(b)
    draw = float(np.exp(np.mean(np.log(np.abs(resid)))))
    return resid, draw


def prob_noise_less_than(sigma_noise, sigma_th):
    """P(draw < sigma_noise) for a draw ~ N(0, sigma_th^2).

    Theoretical probability (assuming normality) that a Monte-Carlo draw lands
    below the observed empirical noise level, i.e. Phi(sigma_noise / sigma_th)
    for the standard normal CDF.
    """
    z = sigma_noise / sigma_th
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _bias_curve(dt):
    """Densely sampled (dt, b) of the continuous weak-bias curve for a smooth line."""
    dt_fine = np.geomspace(np.min(dt), np.max(dt), _N_CURVE)
    return dt_fine, weak_bias(dt_fine)


def _draw_semilog(ax, dt, b, b_obs, w, name, draw, S_T):
    """Signed weak bias for one scheme on a semi-log (log x, linear y) axis."""
    ax.set_xscale("log")
    ax.axhline(0.0, color="0.3", lw=0.8, zorder=1)

    # Continuous leading-order weak-bias curve, densely sampled for smoothness.
    dt_fine, b_fine = _bias_curve(dt)
    ax.plot(dt_fine, b_fine, color="0.2", ls="--", lw=1.6, zorder=3,
            label=r"weak bias $E[S_N]-E[S(T)]$")

    c = COLORS[name]
    m = MARKERS[name]
    # 95% CI band around the observed mean, from the theoretical sigma.
    ax.fill_between(dt, b_obs - w, b_obs + w, color=c, alpha=0.15, zorder=2,
                    label=r"95% CI ($1.96\sigma_{\mathrm{th}}$)")
    ax.plot(dt, b_obs, marker=m, ls="-", lw=1.2, ms=5, color=c, zorder=4,
            label=name)

    # The vertical gap to the bias curve is the (single) MC draw, marked in red.
    ax.vlines(dt, b, b_obs, color=NOISE_COLOR, lw=0.7, alpha=0.7, zorder=1)

    # Annotate the fitted draw at the finest step, where the bias curve is ~ 0
    # (there the gap to the observed mean equals the draw).
    xi = dt[-1]
    y_bot = b[-1]
    y_top = b_obs[-1]
    ax.annotate("", xy=(xi, y_bot), xytext=(xi, y_top),
                arrowprops=dict(arrowstyle="<->", color=NOISE_COLOR, lw=1.0))
    pct = 100.0 * draw / S_T
    p_less = 100.0 * prob_noise_less_than(draw, w / Z)
    ax.text(xi * 0.55, 0.5 * (y_bot + y_top),
            "draw = %.4f%%\nP(noise < draw) = %.1f%%" % (pct, p_less),
            fontsize="small", ha="right", va="center", color=NOISE_COLOR)

    ax.set_ylabel(r"weak bias $E[S_N]-E[S(T)]$")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize="small", loc="best")


def _draw_loglog(ax, dt, b_obs, resid, w, name, draw):
    """|weak error| on a log-log axis: CI shaded, per-dt differences connected,
    and their horizontal fit as the empirical draw."""
    c = COLORS[name]

    # Continuous leading-order |bias| curve, densely sampled for smoothness.
    dt_fine, b_fine = _bias_curve(dt)
    ax.loglog(dt_fine, np.abs(b_fine), color="0.2", ls="--", lw=1.6, zorder=3,
              label="weak bias (abs)")
    ax.loglog(dt, np.abs(b_obs), marker=MARKERS[name], ls="-", lw=1.2,
              ms=5, color=c, zorder=4, label=name)

    # 95% CI noise floor (theoretical), shaded in the method's signature color.
    y_bottom = ax.get_ylim()[0]
    ax.fill_between(dt, y_bottom, w, color=c, alpha=0.15, zorder=1,
                    label=r"95% CI noise floor ($1.96\sigma_{\mathrm{th}}$)")

    # Per-dt differences projected onto the log-log axis: connect the dots, then
    # fit a horizontal line through them -- the empirical draw level.
    ax.loglog(dt, np.abs(resid), marker="x", ls="-", lw=1.2, ms=5,
              color=NOISE_COLOR, zorder=2, label="per-dt difference")
    ax.axhline(draw, color=NOISE_COLOR, ls=":", lw=1.4, zorder=2,
               label="empirical draw (fit)")

    ax.set_ylabel(r"weak error $|E[S_N]-E[S(T)]|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small", loc="best")


def plot_semilog(dt, b, b_obs, w, name, draw, S_T, figsize=(7, 5)):
    """Standalone semi-log (log x, linear y) weak-convergence plot."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_semilog(ax, dt, b, b_obs, w, name, draw, S_T)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_title(f"{name}: weak error (semi-log)")
    return fig, ax


def plot_loglog(dt, b_obs, resid, w, name, draw, figsize=(7, 5)):
    """Standalone log-log weak-convergence plot."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_loglog(ax, dt, b_obs, resid, w, name, draw)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_title(f"{name}: |weak error| (log-log)")
    return fig, ax


def plot_combined(dt, b, b_obs, resid, w, name, draw, S_T, figsize=(7, 8)):
    """Integrated top-bottom plot: log-log on top, semi-log below, sharing dt."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
    ax_top, ax_bottom = axes

    _draw_loglog(ax_top, dt, b_obs, resid, w, name, draw)
    ax_top.set_title("|weak error| (log-log)")

    _draw_semilog(ax_bottom, dt, b, b_obs, w, name, draw, S_T)
    ax_bottom.set_title("weak error (semi-log)")
    ax_bottom.set_xlabel(r"$\Delta t$")

    fig.suptitle(f"{name}: weak error")
    return fig, axes


def main():
    import matplotlib
    matplotlib.use("Agg")  # headless save; run_all already sets MPLBACKEND=Agg

    n, dt, S_N_EM, S_N_Mil = read_table()
    S_T = S0 * np.exp(MU * T)          # E[S(T)], analytic expectation of the mean

    b = weak_bias(dt)
    sigma_th = theoretical_sigma()
    w = Z * sigma_th

    S_N = {"Euler-Maruyama": S_N_EM, "Milstein": S_N_Mil}
    b_obs = {name: S_N[name] - S_T for name in METHODS}
    resid, draw = {}, {}
    for name in METHODS:
        resid[name], draw[name] = empirical_draw(b_obs[name], b)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for name in METHODS:
        slug = METHOD_SLUGS[name]
        outputs = {
            f"exp1_gbm_weak_semilog_{slug}.png":
                plot_semilog(dt, b, b_obs[name], w, name, draw[name], S_T),
            f"exp1_gbm_weak_loglog_{slug}.png":
                plot_loglog(dt, b_obs[name], resid[name], w, name, draw[name]),
            f"exp1_gbm_weak_combined_{slug}.png":
                plot_combined(dt, b, b_obs[name], resid[name], w, name,
                              draw[name], S_T),
        }
        for fname, (fig, _) in outputs.items():
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            print("Saved %s" % os.path.join(FIGURES_DIR, fname))

    print("theoretical sigma  sigma_th = %.4f" % sigma_th)
    print("95%% CI half-width  1.96 * sigma_th = %.4f" % w)
    for name in METHODS:
        pct = 100.0 * draw[name] / S_T
        p_less = 100.0 * prob_noise_less_than(draw[name], sigma_th)
        print("%-16s empirical draw = %.4f  (%.4f%% of S_T), "
              "P(noise < draw) = %.1f%%" % (name, draw[name], pct, p_less))


if __name__ == "__main__":
    main()
