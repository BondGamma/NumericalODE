#run_all.py — Master runner for NumericalPDE experiments to make Prof. Hui life easier :)
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT    = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "code" / "experiments"
LOGS_DIR        = PROJECT_ROOT / "logs" / "run_all"
REQUIREMENTS    = PROJECT_ROOT / "requirements.txt"

DEFAULT_ORDER = [
    "exp0_simulate_bm.py",
    "exp1_gbm_convergence.py",
    "exp3_nasv_convergence.py",
    "exp4_cost_accuracy.py",
]

TIMEOUT = 3600

TIMEOUTS = {
    "exp3_nasv_convergence.py": 6 * 3600,
    "exp4_cost_accuracy.py":    6 * 3600,
}


def install_requirements():
    if not REQUIREMENTS.is_file():
        print(f"No requirements.txt found at {REQUIREMENTS}, skipping install.")
        return True

    print(f"{'=' * 70}")
    print(f"▶  Installing requirements from {REQUIREMENTS.name}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    except Exception as e:
        print(f"   → ERROR while installing requirements: {e}")
        return False

    if proc.returncode != 0:
        print(f"   → FAIL  (pip returned {proc.returncode})")
        return False

    print("   → OK  (requirements installed)\n")
    return True


# whatever it optimized the search of the directory ..
def discover():
    if not EXPERIMENTS_DIR.is_dir():
        raise SystemExit(f"Experiments dir not found: {EXPERIMENTS_DIR}")

    files = sorted(
        p for p in EXPERIMENTS_DIR.iterdir()
        if p.suffix == ".py" and not p.name.startswith("_")
    )

    order_map = {name: i for i, name in enumerate(DEFAULT_ORDER)}
    files.sort(key=lambda p: (order_map.get(p.name, len(DEFAULT_ORDER)), p.name))

    notebooks = [p for p in EXPERIMENTS_DIR.iterdir() if p.suffix == ".ipynb"]
    if notebooks:
        print("Skipping notebooks (no nbconvert installed):")
        for nb in notebooks:
            print(f"  · {nb.name}")
        print()

    return files


def run_one(path: Path) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{path.stem}.log"
    timeout = TIMEOUTS.get(path.name, TIMEOUT)

    print(f"\n{'=' * 70}")
    print(f"▶  {path.name}   (timeout {timeout}s)")
    print(f"   log → {log_path}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, str(path)]

    # AI advice on working with the buffer
    env = os.environ.copy()
    env["MPLBACKEND"]       = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

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
                timeout=timeout,
                env=env,
            )
            status = "OK" if proc.returncode == 0 else "FAIL"
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            status, rc = "TIMEOUT", None
            with log_path.open("a") as f:
                f.write(f"\n# runner: timed out after {timeout}s\n")
        except Exception as e:
            status, rc = "ERROR", None
            with log_path.open("a") as f:
                f.write(f"\n# runner exception: {e}\n")

    elapsed = time.perf_counter() - t0
    print(f"   → {status}  ({elapsed:.1f}s)")
    return {"name": path.name, "status": status, "rc": rc, "elapsed": elapsed}


def parse_args():
    p = argparse.ArgumentParser(description="Run all NumericalPDE experiments.")
    p.add_argument("--list", action="store_true", help="List and exit.")
    p.add_argument("--only", nargs="+", metavar="STEM",
                   help="Run only these experiment stems (e.g. exp4_cost_accuracy).")
    p.add_argument("--skip", nargs="+", metavar="STEM",
                   help="Skip these experiment stems.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going after a failure.")
    p.add_argument("--no-install", action="store_true",
                   help="Skip installing requirements.txt before running.")
    return p.parse_args()


# THE MAIN RUNER READY TO RUN
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

    # Install dependencies before running anything
    if not args.no_install:
        if not install_requirements() and not args.continue_on_error:
            print("Stopping because requirements installation failed "
                  "(use --continue-on-error to override).")
            return 1

    results = []
    t_start = time.perf_counter()
    for path in experiments:
        res = run_one(path)
        results.append(res)
        if res["status"] != "OK" and not args.continue_on_error:
            print("\nStopping on first failure (use --continue-on-error to override).")
            break

    # PRINTING SUMARRRYYY HEREEEE
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(f"  {r['status']:8s}  {r['elapsed']:8.1f}s  {r['name']}")
    print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print(f"\n{len(failed)} failure(s). See {LOGS_DIR}")
        return 1
    print("\nAll experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())