"""
run_all.py — Master runner for NumericalPDE experiments.

Runs every .py file in code/experiments/ using the current Python
interpreter. Captures each experiment's stdout+stderr to logs/run_all/.
Pure standard library — no extra packages.

Usage (from repo root):
    python code/run_all.py
    python code/run_all.py --list
    python code/run_all.py --only exp1
    python code/run_all.py --skip exp0
    python code/run_all.py --continue-on-error
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --- paths ---
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "code" / "experiments"
LOGS_DIR        = PROJECT_ROOT / "logs" / "run_all"

# Run order: files listed here run first, in this order.
# Any other .py file in experiments/ is appended alphabetically.
DEFAULT_ORDER = [
    "exp0_simulate_bm.py",
    "exp1_gbm_convergence.py",
    # "exp2_....py",
]

TIMEOUT = 3600  # seconds per experiment


# --------------------------------------------------------------------------
def discover():
    """Return .py files in code/experiments/ in run order."""
    if not EXPERIMENTS_DIR.is_dir():
        raise SystemExit(f"Experiments dir not found: {EXPERIMENTS_DIR}")

    files = sorted(
        p for p in EXPERIMENTS_DIR.iterdir()
        if p.suffix == ".py" and not p.name.startswith("_")
    )

    order_map = {name: i for i, name in enumerate(DEFAULT_ORDER)}
    files.sort(key=lambda p: (order_map.get(p.name, len(DEFAULT_ORDER)), p.name))

    # Warn about any .ipynb we skip (no nbconvert installed by design).
    notebooks = [p for p in EXPERIMENTS_DIR.iterdir() if p.suffix == ".ipynb"]
    if notebooks:
        print("Skipping notebooks (no nbconvert installed):")
        for nb in notebooks:
            print(f"  · {nb.name}")
        print()

    return files


# --------------------------------------------------------------------------
def run_one(path: Path) -> dict:
    """Run a single experiment; return a result dict."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{path.stem}.log"

    print(f"\n{'=' * 70}")
    print(f"▶  {path.name}")
    print(f"   log → {log_path}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, str(path)]
    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# {path.name}\n")
        log_file.write(f"# command: {' '.join(cmd)}\n")
        log_file.write(f"# started: {datetime.now().isoformat()}\n\n")
        log_file.flush()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
            status = "OK" if proc.returncode == 0 else "FAIL"
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            status, rc = "TIMEOUT", None
        except Exception as e:
            status, rc = "ERROR", None
            with log_path.open("a") as f:
                f.write(f"\n# runner exception: {e}\n")

    elapsed = time.perf_counter() - t0
    print(f"   → {status}  ({elapsed:.1f}s)")
    return {"name": path.name, "status": status, "rc": rc, "elapsed": elapsed}


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run all NumericalPDE experiments.")
    p.add_argument("--list", action="store_true", help="List and exit.")
    p.add_argument("--only", nargs="+", metavar="STEM",
                   help="Run only these experiment stems (e.g. exp1).")
    p.add_argument("--skip", nargs="+", metavar="STEM",
                   help="Skip these experiment stems.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going after a failure.")
    return p.parse_args()


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    experiments = discover()

    if args.only:
        wanted = {s.lower() for s in args.only}
        experiments = [p for p in experiments if p.stem.lower() in wanted]
    if args.skip:
        unwanted = {s.lower() for s in args.skip}
        experiments = [p for p in experiments if p.stem.lower() not in unwanted]

    if not experiments:
        print("Nothing to run.")
        return 0

    if args.list:
        print("Discovered experiments (in run order):")
        for p in experiments:
            print(f"  · {p.name}")
        return 0

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python:       {sys.executable}")
    print(f"Experiments:  {len(experiments)}")

    results = []
    t_start = time.perf_counter()
    for path in experiments:
        res = run_one(path)
        results.append(res)
        if res["status"] != "OK" and not args.continue_on_error:
            print("\nStopping on first failure (use --continue-on-error to override).")
            break

    # --- summary ---
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(f"  {r['status']:8s}  {r['elapsed']:7.1f}s  {r['name']}")
    print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print(f"\n{len(failed)} failure(s). See {LOGS_DIR}")
        return 1
    print("\nAll experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())