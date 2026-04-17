#!/usr/bin/env bash
# run_gnot.sh — Run GNOT benchmark on all 5 PDEs
# Usage:
#   bash hpit_benchmark/run_gnot.sh           # full run
#   bash hpit_benchmark/run_gnot.sh --dry-run # pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"
EXTRA_ARGS="${*}"

echo "=== GNOT Benchmark ==="
for pde in $PDELIST; do
    echo "--- GNOT: $pde ---"
    python3 hpit_benchmark/gnot_benchmark.py \
        --pde "$pde" --epochs 500 $EXTRA_ARGS
done

echo ""
echo "=== GNOT runs complete ==="
