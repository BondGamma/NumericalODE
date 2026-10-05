"""Shared plotting helpers used across the experiments.

Public API:
    plot_paths             universal path plotter (all / sample / selected)
    plot_paths_band        ensemble confidence band + sampled paths
    plot_paths_band_slices confidence band + cross-section distributions
    plot_distribution      density/curve plotter with highlighted regions
    plot_semilog           weak-convergence signed bias on a single-log axis
    plot_loglog            weak-convergence |error| on a log-log axis
    plot_combined          weak-convergence panels stacked, shared x-axis
"""

from code.visualizations.paths import plot_paths
from code.visualizations.paths import plot_paths_band
from code.visualizations.paths import plot_paths_band_slices
from code.visualizations.distributions import plot_distribution
from code.visualizations.weak_convergence import plot_semilog
from code.visualizations.weak_convergence import plot_loglog
from code.visualizations.weak_convergence import plot_combined

__all__ = ["plot_paths", "plot_paths_band", "plot_paths_band_slices",
           "plot_distribution", "plot_semilog", "plot_loglog", "plot_combined"]
