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

_Last updated 2026-09-27._

| Workstream | Status | Where |
|---|---|---|
| Experiment 0 — "same Brownian motion" smoke test + confidence band and cross-sections illustration| ✅ Done | `code/experiments/exp0_simulate_bm.py` |
| Experiment 1 — GBM convergence analysis | ✅ Done	 |  implemented  |
| Experiment 2 — GBM positivity analysis | ✅ Done | implemented|
| Experiment 3 — NASV convergence analysis | ✅ Done	 | implemented  |
| Experiment 4 — cost vs. accuracy analysis | ✅ Done	 |  implemented  |
| `run_all.py` entry point | ✅ Done | Will add pip Requirements later |
| Slides & report | 🚧 Draft | `report/projection_1.tex` |


## How to run the code

```bash
# 1. clone
git clone <repo-url>
cd NumericalPDE

# 2. create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

# 3. run an experiment (run_all.py is not wired up yet)
python3 code/run_all.py
```
Generated figures are written to `figures/`; the final ones get copied into
`report/`. LaTeX build artefacts (`.aux`, `.log`, `.toc`, `.bbl`, `.blg`) are
ignored by `.gitignore`.

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
