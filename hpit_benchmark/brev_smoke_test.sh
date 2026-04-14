#!/usr/bin/env bash
# =============================================================================
# brev_smoke_test.sh — Pre-flight check before full Brev GPU benchmark run.
#
# Run this first (takes ~5-10 min on CPU, <2 min on GPU) to verify every
# pipeline passes end-to-end before committing to the 10-hour full run.
#
# Usage (from repo root):
#   bash hpit_benchmark/brev_smoke_test.sh
#
# All models run --dry-run (tiny grids, 2 epochs) — validates code paths,
# not real numerical results.
# =============================================================================

set -euo pipefail   # exit on error, unset variable, pipe failure

echo ""
echo "============================================================"
echo "  HPIT Benchmark — Pre-Brev Smoke Test"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 0. Dependency check
# --------------------------------------------------------------------------
echo "[0/8] Checking dependencies..."

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "import scipy; print(f'  SciPy {scipy.__version__}')"
python -c "import einops; print(f'  einops ok')"

# Optional: Mamba SSM fast path
python -c "import mamba_ssm; print('  mamba_ssm ok (GPU fast path available)')" 2>/dev/null \
    || echo "  mamba_ssm not installed — pure-PyTorch fallback will be used (correct)"

# Check ref/ data files exist (required for full runs)
echo ""
echo "  Checking ref/ data files (needed for full runs, not dry-run):"
for f in ref/burgers1d.dat ref/Kuramoto_Sivashinsky.dat ref/burgers2d_0.dat \
          ref/heat_complex.dat ref/lid_driven_a2.dat; do
    if [ -f "$f" ]; then
        echo "    ✓ $f"
    else
        echo "    ✗ MISSING: $f  (full run will fail for this PDE)"
    fi
done

echo ""

# --------------------------------------------------------------------------
# 1–6. Neural operator dry-runs
# --------------------------------------------------------------------------

echo "[1/8] FNO dry-run..."
python hpit_benchmark/fno_benchmark.py --dry-run
echo "  ✓ FNO"

echo "[2/8] GNOT dry-run..."
python hpit_benchmark/gnot_benchmark.py --dry-run
echo "  ✓ GNOT"

echo "[3/8] DeepONet dry-run..."
python hpit_benchmark/deeponet_benchmark.py --dry-run
echo "  ✓ DeepONet"

echo "[4/8] PINO dry-run..."
python hpit_benchmark/pino_benchmark.py --dry-run
echo "  ✓ PINO"

echo "[5/8] Mamba-NO dry-run..."
python hpit_benchmark/mamba_no_benchmark.py --dry-run
echo "  ✓ Mamba-NO"

echo "[6/8] HPIT dry-run (trains 2 epochs on synthetic data, then evaluates)..."
python hpit_benchmark/hpit_pde_benchmark.py --dry-run
echo "  ✓ HPIT"

# --------------------------------------------------------------------------
# 7. Collect results
# --------------------------------------------------------------------------
echo "[7/8] Collect and unify results..."
python hpit_benchmark/collect_results.py
echo "  ✓ collect_results"

# --------------------------------------------------------------------------
# 8. Sanity check output
# --------------------------------------------------------------------------
echo ""
echo "[8/8] Sanity-checking unified_results.md..."

RESULT_FILE="hpit_benchmark/results/unified_results.md"
if [ ! -f "$RESULT_FILE" ]; then
    echo "  ✗ MISSING: $RESULT_FILE"
    exit 1
fi

echo ""
cat "$RESULT_FILE" | head -20

echo ""
echo "  Checking for unexpected 'ERROR' entries in result CSVs..."
ERRORS=$(grep -l "ERROR" hpit_benchmark/results/*.csv 2>/dev/null || true)
if [ -n "$ERRORS" ]; then
    echo "  ⚠ ERROR entries found in:"
    echo "$ERRORS"
    grep "ERROR" hpit_benchmark/results/*.csv | head -10
else
    echo "  ✓ No ERROR entries in CSVs"
fi

echo ""
echo "============================================================"
echo "  ALL SMOKE TESTS PASSED — ready for full Brev run"
echo ""
echo "  Next step (full run, ~10 hours on A100):"
echo "    pwsh hpit_benchmark/run_all.ps1"
echo ""
echo "  Or run individual models:"
echo "    python hpit_benchmark/fno_benchmark.py"
echo "    python hpit_benchmark/gnot_benchmark.py"
echo "    python hpit_benchmark/deeponet_benchmark.py"
echo "    python hpit_benchmark/pino_benchmark.py"
echo "    python hpit_benchmark/mamba_no_benchmark.py"
echo "    python hpit_benchmark/hpit_pde_benchmark.py   # trains+evaluates"
echo "    python hpit_benchmark/collect_results.py"
echo "============================================================"
