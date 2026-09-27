# NumericalODE — Team 11
```text
╔══════════════╗     y
║ Solving SDE  ║     │        ╱╲
║    💻 ⚙️     ║     │      ╱    ╲
╚══════════════╝     │   ╱╲╱      ╲
                     │ ╱            ╲
                     └───────────────► t
```


MTH321 Project 1: numerical methods for Ordinary Differential Equation (ODE)
initial value problems.

> The implementation has grown into **stochastic** differential equations
> (SDEs) — Brownian motion, geometric Brownian motion, and a noise-assisted
> stochastic-volatility model — solved with Euler–Maruyama and Milstein.

## Current status

_Last updated 2026-09-25._

| Workstream | Status | Where |
|---|---|---|
| Experiment 0 — "same Brownian motion" smoke test + confidence band and cross-sections illustration| ✅ Done | `code/experiments/exp0_simulate_bm.py` |
| Experiment 1 — GBM convergence analysis | 🚧 Stub | — |
| Experiment 2 — GBM positivity analysis | ✅ Done | `code/experiments/exp2_positivity_analysis.ipynb` |
| Experiment 3 — NASV convergence analysis | 🚧 Stub | — |
| Experiment 4 — cost vs. accuracy analysis | 🚧 Stub | — |
| `run_all.py` entry point | 🚧 Stub | not wired to experiments yet |
| Slides & report | 🚧 Draft | `report/projection_1.tex` |

## Repository layout

```
project-repo/
├── README.md          # how to run the code + who does what + status
├── .gitignore
├── code/              # all source
│   ├── run_all.py     # entry point (currently a stub)
│   ├── SDEs/          # model definitions + Brownian-motion engine
│   │   ├── bm_engine.py   # standard_bm, correlated_bm_pair, extract_dw
│   │   ├── gbm.py         # geometric Brownian motion (SDE)
│   │   └── nasv.py        # none-affine stochastic volatility (SDE)
│   ├── solvers/       # time-stepping schemes
│   │   ├── em.py          # Euler–Maruyama
│   │   └── milstein.py    # Milstein
│   ├── tools/         # shared helpers
│   │   └── mc.py          # Monte-Carlo BM simulator (writes logs to data/)
│   ├── visualizations/# plotting helpers
│   │   ├── paths.py       # plot_paths (+ band / slices variants)
│   │   └── distributions.py  # plot_distribution
│   └── experiments/   # one script/notebook per experiment
│       ├── exp0_simulate_bm.py
│       └── exp2_positivity_analysis.ipynb
├── data/              # generated BM increment logs (gitignored)
├── figures/           # generated figures (final ones go in the report)
├── report/            # LaTeX slides (projection_1.tex)
└── notes/             # meeting notes, issue tickets (optional)
```

Keep `code/` for source only — never commit output files (`data/` is ignored,
see `.gitignore`).

## How to run the code

```bash
# 1. clone
git clone <repo-url>
cd NumericalPDE

# 2. create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

# 3. install dependencies
pip install numpy scipy matplotlib notebook

# 4. run an experiment (run_all.py is not wired up yet)
python code/experiments/exp0_simulate_bm.py
jupyter notebook code/experiments/exp2_positivity_analysis.ipynb
```

Generated figures are written to `figures/`; the final ones get copied into
`report/`. LaTeX build artefacts (`.aux`, `.log`, `.toc`, `.bbl`, `.blg`) are
ignored by `.gitignore`.

## Experiments

### Experiment 0 — "same Brownian motion" smoke test

Verifies the core assumption behind the whole convergence study: that every
step size `dt` sees the **same underlying Brownian path**. If that fails, a
strong-convergence comparison across `dt` is meaningless, so this is checked
first.

The pipeline it exercises:

1. `tools/mc.simulate` draws fine Brownian increments **once** at the maximum
   resolution `n_max` and saves them to `data/`:
   - `"standard"` → `data/gbm_dw_log.npy` (independent standard BM, for GBM)
   - `"pairing"`  → `data/nasv_dw_log.npy` (correlated pair, for NA-SV)
2. `SDEs/bm_engine.extract_dw` coarsens a log to any smaller step count `n`
   (which must divide `n_max`) by summing blocks of fine increments, so the
   coarse path is exactly the fine path sampled on the coarser grid.
3. The script plots each fine path and overlays the coarse samples at the
   chosen `n`'s — they must coincide, confirming the unification.

Run it:

```bash
python code/experiments/exp0_simulate_bm.py
```

Output: 4 PNGs in `figures/` — 2 simulations × 2 engines
(`bm_standard_sim{1,2}.png`, `bm_pairing_sim{1,2}.png`).

It also exercises the advanced path plotters (`code/visualizations/paths.py`)
on every 1-D model — pure Brownian motion, exact GBM, EM/Milstein GBM, and
NA-SV — emitting a confidence-band figure (`band_<model>.png`) and a
cross-section figure (`slices_<model>.png`) for each.

### Experiment 2 — GBM positivity analysis

Studies whether Euler–Maruyama and Milstein can step the geometric Brownian
motion into a negative price, comparing the theoretical probability
`P(min S < 0)` against Monte-Carlo estimates across a range of step sizes
`dt`.  Delivered as a notebook:

```bash
jupyter notebook code/experiments/exp2_positivity_analysis.ipynb
```

Output: `figures/exp2_*.png` — an example negative path, the EM/Milstein
left-tail densities, the Milstein bracket parabola, and the negativity-vs-`dt`
comparison.

## Who does what

| Member        | Username          | Role                      |
|---------------|-------------------|---------------------------|
| Zizhao Wang   | Pwzza             | Project manager           |
| Hao He        | HH888-prog        | Mathematical theory       |
| Artem Bobrov  | BondGamma         | Algorithm implementation  |
| Zhenhui Yuan  | god-of-profound   | Visualization & report    |
| Mingzhen Lin  | AmazingFatCat     | Testing & validation      |

What each role owns:

- **Project manager** — plan, meetings, presentations, weekly records
- **Mathematical theory** — model, stability regions, Jacobian, error estimates
- **Algorithm implementation** — G / RK4 / implicit / Newton / adaptivity + git
- **Visualization & report** — figures, animations, LaTeX report, slides
- **Testing & validation** — building your own oracle, cross-checks, edge cases
