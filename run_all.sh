#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full HPIT + PINNacle benchmark pipeline for Brev GPU instance.
#
# Usage (from repo root, inside tmux):
#   bash run_all.sh
#
# Set DRY_RUN=true for a fast pipeline check (~5-10 min):
#   DRY_RUN=true bash run_all.sh
#
# Estimated time (A100 40 GB, full run): ~10-11 hours
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-false}"
ITER=20000
DRY_FLAG=""
if [ "$DRY_RUN" = "true" ]; then
    ITER=100
    DRY_FLAG="--dry-run"
    echo ""
    echo "*** DRY RUN MODE — 100 iterations, tiny grids ***"
fi

step() { echo ""; echo "============================================================"; echo "  $1"; echo "============================================================"; }
run()  { echo "> $@"; "$@"; }

# --------------------------------------------------------------------------
# STEP 1 — Install requirements
# --------------------------------------------------------------------------
step "STEP 1: Install requirements"
run pip install -r requirements.txt --quiet
run pip install scipy einops --quiet
# Uncomment after confirming CUDA version on Brev:
# run pip install mamba-ssm causal-conv1d --quiet

# --------------------------------------------------------------------------
# STEP 2-4 — PINNacle PINN baselines
# NOTE: benchmark.py has no --dry-run flag; use --iter for fast runs.
# --------------------------------------------------------------------------
step "STEP 2: Vanilla PINN (Adam, $ITER iter)"
run python3 benchmark.py --name vanilla_pinn --method adam --iter $ITER

step "STEP 3: PINN-LRA ($ITER iter)"
run python3 benchmark.py --name pinn_lra --method lra --iter $ITER

step "STEP 4: RAR ($ITER iter)"
run python3 benchmark.py --name rar --method rar --iter $ITER

# --------------------------------------------------------------------------
# STEP 5 — FBPINN (must run from fbpinns/ directory)
# Skipped in dry-run — no dry-run support in FBPINN scripts.
# Writes results to fbpinns/results/models/<RUN>/loss_*.npy
# collect_fbpinn.py converts these to standard CSV after all runs.
# Burgers2D has no FBPINN script — that cell will be '-' in the table.
# --------------------------------------------------------------------------
step "STEP 5: FBPINN benchmark"

if [ "$DRY_RUN" = "true" ]; then
    echo "  Skipping FBPINN in dry-run mode."
else
    pushd fbpinns > /dev/null
    for script in runs/run_burger.py runs/run_heat.py runs/run_chaotic.py runs/run_ns.py; do
        if [ -f "$script" ]; then
            echo "  Running $script ..."
            run python3 "$script"
        else
            echo "  WARNING: $script not found — skipping"
        fi
    done
    popd > /dev/null

    # Collect FBPINN .npy results → standard CSV
    run python3 hpit_benchmark/collect_fbpinn.py
fi

# --------------------------------------------------------------------------
# STEP 6-10 — Neural operators (all use PINNacle .dat data)
# --------------------------------------------------------------------------
step "STEP 6: FNO benchmark"
run python3 hpit_benchmark/fno_benchmark.py $DRY_FLAG

step "STEP 7: GNOT benchmark"
run python3 hpit_benchmark/gnot_benchmark.py $DRY_FLAG

step "STEP 8: DeepONet benchmark"
run python3 hpit_benchmark/deeponet_benchmark.py $DRY_FLAG

step "STEP 9: PINO benchmark"
run python3 hpit_benchmark/pino_benchmark.py $DRY_FLAG

step "STEP 10: Mamba-NO benchmark"
run python3 hpit_benchmark/mamba_no_benchmark.py $DRY_FLAG

# --------------------------------------------------------------------------
# STEP 11 — HPIT (trains from scratch on PINNacle data, saves checkpoint)
# --------------------------------------------------------------------------
step "STEP 11: HPIT benchmark (trains + evaluates)"
run python3 hpit_benchmark/hpit_pde_benchmark.py $DRY_FLAG

# --------------------------------------------------------------------------
# STEP 12 — Collect all results into unified table
# --------------------------------------------------------------------------
step "STEP 12: Collect and unify results"
run python3 hpit_benchmark/collect_results.py

echo ""
echo "============================================================"
echo "  All steps complete."
echo "  Results: hpit_benchmark/results/unified_results.md"
echo "           hpit_benchmark/results/unified_results.csv"
echo "============================================================"
