#!/usr/bin/env bash
# run_fno.sh — Run FNO benchmark on all 5 PDEs
# Usage:
#   bash hpit_benchmark/run_fno.sh           # full run
#   bash hpit_benchmark/run_fno.sh --dry-run # pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"
EXTRA_ARGS="${*}"

echo "=== FNO Benchmark ==="
for pde in $PDELIST; do
    echo "--- FNO: $pde ---"
    python3 hpit_benchmark/fno_benchmark.py \
        --pde "$pde" --epochs 500 $EXTRA_ARGS
done

echo ""
echo "=== FNO runs complete ==="
