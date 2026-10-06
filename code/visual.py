"""
Animated Monte-Carlo simulation of financial markets.
Saves a high-quality MP4 video (H.264) for macOS.

Model:   dS = mu * S dt + sigma * S dW
Exact:   S(t+dt) = S(t) * exp( (mu - sigma^2/2) dt + sigma sqrt(dt) Z )
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------------
# 1.  Model / simulation parameters
# ----------------------------------------------------------------------
S0      = 100.0     # initial asset price
mu      = 0.08      # expected annual return (drift)
sigma   = 0.25      # annual volatility
T       = 1.0       # time horizon [years]
n_steps = 252       # time steps (trading days in a year)
n_paths = 200       # number of Monte-Carlo paths
seed    = 7
fps     = 30

rng = np.random.default_rng(seed)

dt = T / n_steps
t  = np.linspace(0.0, T, n_steps + 1)

# ----------------------------------------------------------------------
# 2.  Simulate the paths  (exact solution of the SDE)
# ----------------------------------------------------------------------
Z         = rng.standard_normal((n_paths, n_steps))
log_incr  = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
log_paths = np.concatenate([np.zeros((n_paths, 1)),
                            np.cumsum(log_incr, axis=1)], axis=1)
paths     = S0 * np.exp(log_paths)          # shape (n_paths, n_steps+1)

mean_path = paths.mean(axis=0)
q05       = np.percentile(paths,  5, axis=0)
q95       = np.percentile(paths, 95, axis=0)

y_min, y_max = paths.min() * 0.95, paths.max() * 1.05

# ----------------------------------------------------------------------
# 3.  Figure layout
# ----------------------------------------------------------------------
BG, PANEL, FG, MUTED = "#0f1117", "#161a23", "#e8ecf3", "#9aa4b2"
ACCENT = "#4c8bf5"

fig = plt.figure(figsize=(13, 6.8), facecolor=BG, dpi=150)
gs  = GridSpec(1, 2, width_ratios=[3.2, 1], wspace=0.05,
               left=0.065, right=0.965, top=0.86, bottom=0.10)

ax  = fig.add_subplot(gs[0])
axh = fig.add_subplot(gs[1])          # histogram panel


def style_axis(a, show_y=True):
    a.set_facecolor(PANEL)
    a.tick_params(colors=MUTED, labelsize=9)
    a.grid(True, color="#232a38", lw=0.6)
    for s in a.spines.values():
        s.set_color("#2a3040")
    if not show_y:
        a.tick_params(labelleft=False)


style_axis(ax)
style_axis(axh, show_y=False)

ax.set_xlim(0, T)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("time  [years]", color=MUTED)
ax.set_ylabel("price", color=MUTED)
axh.set_ylim(y_min, y_max)
axh.set_xlabel("density", color=MUTED)

fig.suptitle(
    f"Monte-Carlo simulation of a financial market  —  {n_paths} GBM paths   "
    f"($S_0$={S0:g},  $\\mu$={mu:.0%},  $\\sigma$={sigma:.0%})",
    color=FG, fontsize=13, y=0.955)

# colour every path according to its terminal value
final  = paths[:, -1]
colors = plt.cm.turbo(plt.Normalize(final.min(), final.max())(final))

lines = [ax.plot([], [], lw=0.9, color=colors[i], alpha=0.55, zorder=2)[0]
         for i in range(n_paths)]

mean_line, = ax.plot([], [], color="white", lw=2.0, zorder=5,
                     label="mean path")
ax.legend(loc="upper left", facecolor=PANEL, edgecolor="#2a3040",
          labelcolor=FG, fontsize=9)

time_text = ax.text(0.99, 0.03, "", transform=ax.transAxes,
                    ha="right", va="bottom", color=MUTED, fontsize=10)

band = {"c": None}          # holds the current quantile-band collection


# ----------------------------------------------------------------------
# 4.  Animation callback
# ----------------------------------------------------------------------
def update(k):
    # --- grow every price path up to step k -------------------------
    for i, ln in enumerate(lines):
        ln.set_data(t[:k + 1], paths[i, :k + 1])
    mean_line.set_data(t[:k + 1], mean_path[:k + 1])

    # --- 5 %–95 % quantile band -------------------------------------
    if band["c"] is not None:
        band["c"].remove()
    band["c"] = ax.fill_between(t[:k + 1], q05[:k + 1], q95[:k + 1],
                                color=ACCENT, alpha=0.16, lw=0, zorder=1)

    time_text.set_text(f"t = {t[k] * 12:4.1f} months")

    # --- live cross-sectional histogram ------------------------------
    axh.clear()
    style_axis(axh, show_y=False)
    axh.set_ylim(y_min, y_max)
    axh.set_xlabel("density", color=MUTED)

    prices = paths[:, k]
    axh.hist(prices, bins=35, orientation="horizontal",
             color=ACCENT, alpha=0.6, density=True, zorder=2)

    # theoretical log-normal density of S(t)
    if t[k] > 0:
        s = sigma * np.sqrt(t[k])
        m = np.log(S0) + (mu - 0.5 * sigma**2) * t[k]
        y = np.linspace(y_min, y_max, 300)
        pdf = np.exp(-(np.log(y) - m) ** 2 / (2 * s**2)) / \
              (y * s * np.sqrt(2 * np.pi))
        axh.plot(pdf, y, color="#ffd166", lw=1.6, zorder=3)

    return lines + [mean_line, time_text]


# ----------------------------------------------------------------------
# 5.  Create animation and save as high-quality MP4
# ----------------------------------------------------------------------
anim = FuncAnimation(fig, update, frames=n_steps + 1,
                     interval=1000 / fps, blit=False,
                     repeat=False, cache_frame_data=False)

output_file = "monte_carlo_simulation.mp4"

# High-quality H.264 encoding (works well on macOS / QuickTime)
writer = FFMpegWriter(
    fps=fps,
    codec="h264",
    bitrate=8000,                      # 8 Mbps — very good quality
    extra_args=["-pix_fmt", "yuv420p"] # ensures compatibility with QuickTime
)

print(f"Saving video to {output_file} ...")
anim.save(output_file, writer=writer, dpi=150)
print("Done.")

# If you also want to view the animation interactively after saving,
# comment out the line above and uncomment the following:
# plt.show()