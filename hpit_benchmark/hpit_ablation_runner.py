#!/usr/bin/env python3
"""
hpit_ablation_runner.py — Thin wrapper: HPIT data-only baseline (no physics).

Calls hpit_pde_benchmark.py without --pde-physics or --physics, redirecting
output to hpit_results_ablation.csv so the data-only runs appear in the
collect_results.py unified table as "HPIT (no physics)" — distinct from
the "HPIT" row which comes from the main results CSV.

This satisfies Brandon's explicit ablation requirement: the results table must
have both "HPIT (with physics)" and "HPIT (without physics)" rows.

Usage (mirrors hpit_pde_benchmark.py CLI):
    python hpit_benchmark/hpit_ablation_runner.py --dry-run --pde Burgers1D
    python hpit_benchmark/hpit_ablation_runner.py --pde HeatComplexGeometry --epochs 500
    python hpit_benchmark/hpit_ablation_runner.py  # all 5 PDEs
"""
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
ABLATION_CSV  = str(BENCHMARK_DIR / "results" / "hpit_results_ablation.csv")
MAIN_SCRIPT   = str(BENCHMARK_DIR / "hpit_pde_benchmark.py")


def main():
    # Pass all args through, but:
    # 1. Force --results-csv to the ablation CSV
    # 2. Strip any physics flags — ablation is data-only by definition
    _physics_flags = {
        "--pde-physics", "--physics",
        "--physics-mass", "--physics-energy", "--physics-elevation",
    }
    passthrough = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--results-csv"):
            # Skip caller-supplied --results-csv (we force our own)
            if "=" not in arg:
                skip_next = True  # next token is the value
            continue
        if arg in _physics_flags:
            continue
        passthrough.append(arg)

    cmd = [sys.executable, MAIN_SCRIPT,
           "--results-csv", ABLATION_CSV] + passthrough
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
