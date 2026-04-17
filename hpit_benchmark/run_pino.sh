#!/usr/bin/env bash
# run_pino.sh — Run PINO benchmark on all 5 PDEs
# Usage:
#   bash hpit_benchmark/run_pino.sh           # full run
#   bash hpit_benchmark/run_pino.sh --dry-run # pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"
EXTRA_ARGS="${*}"

echo "=== PINO Benchmark ==="
for pde in $PDELIST; do
    echo "--- PINO: $pde ---"
    python3 hpit_benchmark/pino_benchmark.py \
        --pde "$pde" --epochs 500 $EXTRA_ARGS
done

echo ""
echo "=== PINO runs complete ==="
