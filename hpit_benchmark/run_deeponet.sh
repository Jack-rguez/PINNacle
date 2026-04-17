#!/usr/bin/env bash
# run_deeponet.sh — Run DeepONet benchmark on all 5 PDEs
# Usage:
#   bash hpit_benchmark/run_deeponet.sh           # full run
#   bash hpit_benchmark/run_deeponet.sh --dry-run # pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"
EXTRA_ARGS="${*}"

echo "=== DeepONet Benchmark ==="
for pde in $PDELIST; do
    echo "--- DeepONet: $pde ---"
    python3 hpit_benchmark/deeponet_benchmark.py \
        --pde "$pde" --epochs 500 $EXTRA_ARGS
done

echo ""
echo "=== DeepONet runs complete ==="
