#!/usr/bin/env bash
# run_hpit_ablation.sh — HPIT without physics (ablation → hpit_results_ablation.csv)
#
# Usage:
#   bash hpit_benchmark/run_hpit_ablation.sh               # all 5 PDEs
#   bash hpit_benchmark/run_hpit_ablation.sh --dry-run     # pipeline check
#
# To re-run a single PDE (e.g. after cancellation):
#   python3 hpit_benchmark/hpit_ablation_runner.py --pde NavierStokes2D --epochs 500
set -euo pipefail
cd "$(dirname "$0")/.."

PDELIST="Burgers1D Burgers2D HeatComplexGeometry KuramotoSivashinsky NavierStokes2D"

echo "=== HPIT Ablation (no physics) → hpit_results_ablation.csv ==="
for pde in $PDELIST; do
    echo "--- HPIT no-physics: $pde ---"
    python3 hpit_benchmark/hpit_ablation_runner.py \
        --pde "$pde" --epochs 500 "$@"
done

echo ""
echo "=== HPIT ablation runs complete ==="
