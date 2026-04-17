# HPIT PDE Benchmark

Tests HPIT generalizability on classical PDE benchmarks beyond its SWE training domain.

## Files

| File | Purpose |
|---|---|
| `pde_problems.py` | Ground truth PDE solvers + HPIT input formatters |
| `hpit_pde_benchmark.py` | HPIT benchmark runner — all 5 PDEs, dry-run passes |
| `fno_benchmark.py` | FNO benchmark — PINNacle .dat data, dry-run passes |
| `gnot_benchmark.py` | GNOT benchmark — dry-run passes |
| `deeponet_benchmark.py` | DeepONet benchmark — dry-run passes |
| `pino_benchmark.py` | PINO benchmark — dry-run passes |
| `collect_results.py` | Unified results table (all models + PINN baselines) |
| `BENCHMARK_PROVENANCE.md` | Data sources, citations, conceptual model summaries |
| `results/unified_results.md` | Current benchmark table |

## PDEs Covered

| PDE | Class | Difficulty |
|---|---|---|
| Burgers 1D | `Burgers1D` | Moderate (shock fronts) |
| Burgers 2D | `Burgers2D` | Hard (2D shocks) |
| Heat (complex geometry) | `HeatComplexGeometry` | Moderate (rectangle with circular holes) |
| Kuramoto-Sivashinsky | `KuramotoSivashinsky` | Very hard (chaotic) |

Note: Navier-Stokes is implemented as a lid-driven cavity (steady-state)
using PINNacle's `ref/lid_driven_a*.dat` reference data (7 Reynolds numbers).
All 5 PDEs now have complete implementations and passing dry-runs.

## Setup

```bash
# From repo root
pip install -r requirements.txt
pip install scipy  # needed for KS solver
```

## Running

```bash
# Step 1: Dry run first to verify the pipeline works (fast, ~1 min)
python benchmark/hpit_pde_benchmark.py --dry-run

# Step 2: Full benchmark with a trained checkpoint (run on Brev GPU)
python benchmark/hpit_pde_benchmark.py --checkpoint path/to/hpit.pt

# Step 3: Single PDE only
python benchmark/hpit_pde_benchmark.py --pde Burgers1D --checkpoint path/to/hpit.pt
```

## Output Format

`results/hpit_results.csv` columns match PINNacle's `result.csv`:

| Column | Description |
|---|---|
| `pde` | PDE name |
| `model` | Always "HPIT" |
| `l2rel` | L2 relative error (mean over seeds) |
| `l2rel_std` | Standard deviation over seeds |
| `run_time_seconds` | Inference time |
| `notes` | Flags (dry_run, no_checkpoint, errors) |

## How HPIT Inputs Are Constructed

HPIT expects `(batch, seq_len, features)` — meteorological time series.
For PDEs we adapt:
- **Sequence dimension** = time history of length `seq_len` (default 30)
- **Feature dimension** = [x, t, u] or [x, y, t, u, v] depending on PDE
- **Batch dimension** = spatial points × time steps

This treats each spatial location's time history as an independent sample,
mirroring how HPIT uses 30-day meteorological windows.

## Important Notes

1. **No HPIT code was modified.** This is a pure adapter approach.
2. **Results require a trained checkpoint.** Without `--checkpoint`, 
   the model uses random weights and results are meaningless.
3. **L2 relative error definition** follows PDEBench (Takamoto et al., 2022):
   `||u_pred - u_true||_2 / ||u_true||_2`
4. **Navier-Stokes** is implemented as lid-driven cavity using PINNacle
   `ref/lid_driven_a*.dat` (7 steady-state solutions at different Re values).
5. **HeatComplexGeometry** — HPIT's synthetic solver uses a simplified L-shaped
   domain for training data generation. PINNacle's reference is a rectangle with
   circular holes. For paper table fairness, HPIT must be evaluated on the
   PINNacle .dat reference (pending `--use-pinnacle-data` flag implementation).

## References

- PINNacle (Hao et al., NeurIPS 2024): https://arxiv.org/abs/2306.08827
- PDEBench (Takamoto et al., NeurIPS 2022): https://arxiv.org/abs/2210.07182
- ETDRK4 for KS (Cox & Matthews, 2002): J. Comput. Phys. 176(2), 430-455
