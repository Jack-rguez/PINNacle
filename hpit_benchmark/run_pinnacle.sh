#!/usr/bin/env bash
# run_pinnacle.sh — Run PINNacle PINN variants on the 5 benchmark PDEs
#
# Uses benchmark_hpit_pdeset.py (NOT benchmark.py — do not modify the original).
# All 4 PINN methods run sequentially on the same GPU.
# Results land in runs/{date}-{name}/ and are read by collect_results.py.
#
# Usage:
#   bash hpit_benchmark/run_pinnacle.sh           # full run (20000 iter each)
#   bash hpit_benchmark/run_pinnacle.sh --dry-run # 100 iter for pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."

export DDEBACKEND=pytorch

ITER=20000
DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
        ITER=100
    fi
done

echo "=== PINNacle PINN Benchmark ==="
echo "Script: benchmark_hpit_pdeset.py"
echo "Iterations: ${ITER}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Verifying PINNacle imports only..."
    python3 -c "
import os; os.environ['DDEBACKEND'] = 'pytorch'
from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_ComplexGeometry
from src.pde.ns import NS2D_LidDriven
print('PINNacle PDE imports: OK')
"
    echo "[DRY RUN] Import check passed."
fi

echo "--- Vanilla PINN (Adam) ---"
python3 benchmark_hpit_pdeset.py \
    --name vanilla_pinn \
    --method adam \
    --iter "${ITER}" \
    --log-every 500 \
    --plot-every 2000
echo "Vanilla PINN done."

echo "--- PINN-LRA ---"
python3 benchmark_hpit_pdeset.py \
    --name pinn_lra \
    --method lra \
    --iter "${ITER}" \
    --log-every 500 \
    --plot-every 2000
echo "PINN-LRA done."

echo "--- RAR ---"
python3 benchmark_hpit_pdeset.py \
    --name rar \
    --method rar \
    --iter "${ITER}" \
    --log-every 500 \
    --plot-every 2000
echo "RAR done."

# FBPINN uses a separate entry point in fbpinns/
echo "--- FBPINN ---"
echo "NOTE: FBPINN uses fbpinns/run.py, not benchmark_hpit_pdeset.py."
echo "      Check fbpinns/runs/ for per-PDE runner scripts."
echo "      Collect FBPINN results with: python3 hpit_benchmark/collect_fbpinn.py"

echo ""
echo "Collecting all results..."
python3 hpit_benchmark/collect_results.py

echo ""
echo "=== PINNacle runs complete ==="
echo "Results in: runs/*/"
echo "Unified table: hpit_benchmark/results/unified_results.md"
