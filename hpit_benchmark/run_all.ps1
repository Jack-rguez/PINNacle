<#
.SYNOPSIS
    Full HPIT + PINNacle benchmark pipeline for Brev GPU instance.

.DESCRIPTION
    Runs all benchmarks in order and collects unified results.
    Set $DRY_RUN = $true to do a fast pipeline check before committing
    to full GPU runs.

    Requires: pwsh (PowerShell Core) — install on Linux with:
        sudo apt-get install -y powershell

.NOTES
    Estimated GPU time (A100 40 GB, full run):
      Step 2  (Vanilla PINN, 20k iter):  ~45 min
      Step 3  (PINN-LRA,     20k iter):  ~50 min
      Step 4  (RAR,           20k iter):  ~55 min
      Step 5  (FBPINN, 4 PDEs):           ~2 h total
      Step 6  (FNO,    5 PDEs):           ~1 h
      Step 7  (GNOT,   5 PDEs):           ~1 h
      Step 8  (DeepONet, 5 PDEs):         ~1 h
      Step 9  (PINO,   5 PDEs):           ~1.5 h
      Step 10 (Mamba-NO, 5 PDEs):         ~1 h
      Step 11 (HPIT,   5 PDEs, PINNacle): ~30 min
      Step 12 (collect results):          <1 min
      TOTAL ESTIMATE:                     ~10-11 hours
    Dry-run estimate: ~5-10 minutes total.

    Data source: ALL neural operators (Steps 6-11) use PINNacle COMSOL .dat
    reference files from ref/ — same ground truth for fair comparison.
#>

# ============================================================
# USER CONFIGURATION — set these before running
# ============================================================
$DRY_RUN = $false            # Set to $true for a fast pipeline sanity-check

# Absolute path to a trained HPIT checkpoint (.pt).
# Leave as $null to use randomly initialised weights (L2 results will be ~1).
$HPIT_CHECKPOINT = $null

# ============================================================
# HELPERS
# ============================================================

function Write-Step {
    param([string]$msg)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Invoke-Step {
    param([string]$cmd)
    Write-Host "> $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: step failed (exit $LASTEXITCODE): $cmd" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$ITER = if ($DRY_RUN) { "100" } else { "20000" }
$DRY_FLAG = if ($DRY_RUN) { "--dry-run" } else { "" }

Write-Host ""
if ($DRY_RUN) {
    Write-Host "*** DRY RUN MODE — tiny grids, 100 iterations ***" -ForegroundColor Magenta
} else {
    Write-Host "*** FULL BENCHMARK RUN — 20 000 iterations ***" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# STEP 1 — Install requirements
# (~2 min on a clean Brev instance)
# ============================================================
Write-Step "STEP 1: Install requirements"
Invoke-Step "pip install -r requirements.txt --quiet"
Invoke-Step "pip install scipy einops --quiet"
# mamba_ssm provides GPU-optimised SSM kernels (optional — pure-PyTorch fallback available)
# Uncomment once you confirm Brev CUDA version is compatible:
# Invoke-Step "pip install mamba-ssm causal-conv1d --quiet"

# ============================================================
# STEP 2 — PINNacle: Vanilla PINN (Adam)
# (~45 min full / <1 min dry)
# NOTE: benchmark.py does not accept --dry-run; use --iter for fast runs.
# ============================================================
Write-Step "STEP 2: PINNacle Vanilla PINN (Adam, $ITER iter)"
Invoke-Step "python benchmark.py --name vanilla_pinn --method adam --iter $ITER"

# ============================================================
# STEP 3 — PINNacle: PINN with Learning-Rate Annealing (LRA)
# (~50 min full / <1 min dry)
# ============================================================
Write-Step "STEP 3: PINNacle PINN-LRA ($ITER iter)"
Invoke-Step "python benchmark.py --name pinn_lra --method lra --iter $ITER"

# ============================================================
# STEP 4 — PINNacle: Residual-Adaptive Refinement (RAR)
# (~55 min full / <1 min dry)
# ============================================================
Write-Step "STEP 4: PINNacle RAR ($ITER iter)"
Invoke-Step "python benchmark.py --name rar --method rar --iter $ITER"

# ============================================================
# STEP 5 — FBPINN: one run per PDE
# (~30 min per PDE full)
# NOTE: FBPINN scripts must be run from fbpinns/ directory.
# NOTE: No dry-run support — skipped when $DRY_RUN is true.
# NOTE: FBPINN writes its own logs, not PINNacle result.csv format.
#       Results must be collected manually from fbpinns/runs/.
# PDE → script mapping:
#   Burgers1D           → run_burger.py
#   HeatComplexGeometry → run_heat.py
#   KuramotoSivashinsky → run_chaotic.py
#   NavierStokes2D      → run_ns.py
#   Burgers2D           → (no FBPINN script — skip)
# ============================================================
Write-Step "STEP 5: FBPINN benchmark"

$FBPINN_MAP = @{
    "Burgers1D"           = "run_burger.py"
    "HeatComplexGeometry" = "run_heat.py"
    "KuramotoSivashinsky" = "run_chaotic.py"
    "NavierStokes2D"      = "run_ns.py"
}

if ($DRY_RUN) {
    Write-Host "  Skipping FBPINN in dry-run mode (no --dry-run support)." -ForegroundColor Yellow
} else {
    Push-Location "fbpinns"
    foreach ($pde in $FBPINN_MAP.Keys) {
        $script = $FBPINN_MAP[$pde]
        if (Test-Path "runs/$script") {
            Write-Host "  Running FBPINN on $pde ($script)..."
            Invoke-Step "python runs/$script"
        } else {
            Write-Host "  WARNING: FBPINN script not found for ${pde}: runs/$script" -ForegroundColor Yellow
        }
    }
    Pop-Location
    Write-Host "  NOTE: Burgers2D has no FBPINN script — row will be '-' in results table." -ForegroundColor Yellow
}

# ============================================================
# STEP 6 — FNO benchmark (all 5 PDEs, PINNacle data)
# (~1 hr full / ~30 s dry)
# ============================================================
Write-Step "STEP 6: FNO PDE benchmark"
Invoke-Step "python hpit_benchmark/fno_benchmark.py $DRY_FLAG"

# ============================================================
# STEP 7 — GNOT benchmark (all 5 PDEs, PINNacle data)
# (~1 hr full / ~30 s dry)
# ============================================================
Write-Step "STEP 7: GNOT PDE benchmark"
Invoke-Step "python hpit_benchmark/gnot_benchmark.py $DRY_FLAG"

# ============================================================
# STEP 8 — DeepONet benchmark (all 5 PDEs, PINNacle data)
# (~1 hr full / ~30 s dry)
# ============================================================
Write-Step "STEP 8: DeepONet PDE benchmark"
Invoke-Step "python hpit_benchmark/deeponet_benchmark.py $DRY_FLAG"

# ============================================================
# STEP 9 — PINO benchmark (all 5 PDEs, PINNacle data)
# (~1.5 hr full / ~30 s dry)
# ============================================================
Write-Step "STEP 9: PINO PDE benchmark"
Invoke-Step "python hpit_benchmark/pino_benchmark.py $DRY_FLAG"

# ============================================================
# STEP 10 — Mamba-NO benchmark (all 5 PDEs, PINNacle data)
# (~1 hr full / ~30 s dry)
# ============================================================
Write-Step "STEP 10: Mamba-NO PDE benchmark"
Invoke-Step "python hpit_benchmark/mamba_no_benchmark.py $DRY_FLAG"

# ============================================================
# STEP 11 — HPIT benchmark (all 5 PDEs, PINNacle reference data)
# (~30 min full / ~30 s dry)
# NOTE: Full run uses PINNacle .dat ground truth (same as Steps 6-10).
#       Dry-run uses synthetic FD data (no .dat files needed).
# ============================================================
Write-Step "STEP 11: HPIT PDE benchmark"
$hpit_cmd = "python hpit_benchmark/hpit_pde_benchmark.py $DRY_FLAG"
if ($HPIT_CHECKPOINT -ne $null) {
    $hpit_cmd += " --checkpoint `"$HPIT_CHECKPOINT`""
} else {
    Write-Host "  WARNING: No HPIT checkpoint set (\$HPIT_CHECKPOINT is null)." -ForegroundColor Yellow
    Write-Host "  Set \$HPIT_CHECKPOINT to the .pt checkpoint path for real results."
}
Invoke-Step $hpit_cmd

# ============================================================
# STEP 12 — Collect and unify all results
# (<1 min)
# ============================================================
Write-Step "STEP 12: Collect and unify results"
Invoke-Step "python hpit_benchmark/collect_results.py"

# ============================================================
# Done
# ============================================================
Write-Host ""
Write-Host "All steps complete." -ForegroundColor Green
Write-Host "Unified results: hpit_benchmark/results/unified_results.md"
Write-Host "                 hpit_benchmark/results/unified_results.csv"
Write-Host ""
Write-Host "Models included in table:" -ForegroundColor Cyan
Write-Host "  PINN baselines:    Vanilla PINN, PINN-LRA, RAR"
Write-Host "  FBPINN:            results in fbpinns/runs/ (manual collection)"
Write-Host "  Neural operators:  FNO, GNOT, DeepONet, PINO, Mamba-NO"
Write-Host "  HPIT:              hpit_results.csv"
Write-Host ""
Write-Host "All neural operators trained and evaluated on PINNacle COMSOL .dat data." -ForegroundColor Green
