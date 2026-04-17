#!/usr/bin/env bash
# run_mambano.sh — Run Mamba-NO benchmark on all 5 PDEs
# Usage:
#   bash hpit_benchmark/run_mambano.sh           # full run
#   bash hpit_benchmark/run_mambano.sh --dry-run # pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"
EXTRA_ARGS="${*}"

echo "=== Mamba-NO Benchmark ==="
for pde in $PDELIST; do
    echo "--- Mamba-NO: $pde ---"
    python3 hpit_benchmark/mamba_no_benchmark.py \
        --pde "$pde" --epochs 500 $EXTRA_ARGS
done

echo ""
echo "=== Mamba-NO runs complete ==="
