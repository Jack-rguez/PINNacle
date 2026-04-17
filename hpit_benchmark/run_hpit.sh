#!/usr/bin/env bash
# run_hpit.sh — HPIT with PDE physics (main results → hpit_results.csv)
#
# Usage:
#   bash hpit_benchmark/run_hpit.sh               # all 5 PDEs
#   bash hpit_benchmark/run_hpit.sh --dry-run     # pipeline check
#
# To re-run a single PDE (e.g. after cancellation):
#   python3 hpit_benchmark/hpit_pde_benchmark.py --pde NavierStokes2D --pde-physics --epochs 500
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"

echo "=== HPIT Benchmark (with PDE physics) → hpit_results.csv ==="
for pde in $PDELIST; do
    echo "--- HPIT+physics: $pde ---"
    python3 hpit_benchmark/hpit_pde_benchmark.py \
        --pde "$pde" --pde-physics --epochs 500 "$@"
done

echo ""
echo "=== HPIT (physics) runs complete ==="
