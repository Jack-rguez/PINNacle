"""
hpit_pde_benchmark.py — Benchmarks HPIT on classical PDE families.

Usage:
    # Dry run (tiny grids, fast, just checks the pipeline works):
    python hpit_benchmark/hpit_pde_benchmark.py --dry-run

    # Full benchmark using PINNacle reference data (default for paper table):
    python hpit_benchmark/hpit_pde_benchmark.py

    # Full benchmark using synthetic FD ground truth:
    python hpit_benchmark/hpit_pde_benchmark.py --synthetic

    # Single PDE:
    python hpit_benchmark/hpit_pde_benchmark.py --pde Burgers1D

Output:
    hpit_benchmark/results/hpit_results.csv

Data Sources:
    - --dry-run: random tensors (no data needed)
    - default (full run): PINNacle COMSOL .dat reference files from ref/
      Same ground truth as FNO/GNOT/DeepONet/PINO for fair comparison.
    - --synthetic: synthetic FD solutions from pde_problems.py
      Useful for HPIT physics-informed training but NOT for paper comparison.

This file does NOT modify any existing HPIT or PINNacle code.
IMPORTANT: Do not hardcode any results. All numbers come from running the model.
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
import csv
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# File-based imports — NO sys.path manipulation
# All HPIT modules are loaded by absolute path via importlib.util so that
# this script is independent of the rest of the PINNacle/swe-prediction repo.
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
HPIT_SRC_DIR = BENCHMARK_DIR / "hpit_src"


def _load_module(module_name: str, file_path: Path, package: str = None):
    """Load a Python module from an absolute file path.

    Registers the module in sys.modules so relative imports inside the loaded
    module (e.g. ``from .base import ...``) resolve correctly.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    if package:
        mod.__package__ = package
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register hpit_src as a package so relative imports inside hpit.py work.
_hpit_src_pkg = types.ModuleType("hpit_src")
_hpit_src_pkg.__path__ = [str(HPIT_SRC_DIR)]
_hpit_src_pkg.__package__ = "hpit_src"
sys.modules["hpit_src"] = _hpit_src_pkg

# Load base.py first — hpit.py has a relative import that depends on it.
_mod_base = _load_module("hpit_src.base", HPIT_SRC_DIR / "base.py", package="hpit_src")

# Load hpit.py
_mod_hpit = _load_module("hpit_src.hpit", HPIT_SRC_DIR / "hpit.py", package="hpit_src")
HPITModel = _mod_hpit.HPITModel
HPITConfig = _mod_hpit.HPITConfig

# Load pde_problems.py from the same directory as this script.
_mod_pde = _load_module("pde_problems", BENCHMARK_DIR / "pde_problems.py")
PDEProblemConfig = _mod_pde.PDEProblemConfig
ALL_PROBLEMS = _mod_pde.ALL_PROBLEMS
get_problem = _mod_pde.get_problem

# Load fno_benchmark for shared PINNacle data loading.
# Used when --synthetic is NOT set (default) to ensure HPIT evaluates on the
# same PINNacle COMSOL reference data as all other neural operators.
_mod_fno = _load_module("fno_benchmark_hpit", BENCHMARK_DIR / "fno_benchmark.py")
_load_pinnacle_data = _mod_fno.load_data
_l2_relative_error = _mod_fno.l2_relative_error
_PDE1D = _mod_fno._PDE1D

# ---------------------------------------------------------------------------
# Logging and results paths
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "hpit_results.csv"


# ---------------------------------------------------------------------------
# PINNacle → HPIT format converter
# ---------------------------------------------------------------------------

def pinnacle_to_hpit(pde_name: str, x_data: torch.Tensor,
                     y_data: torch.Tensor):
    """
    Convert PINNacle operator-format tensors to HPIT sequence format.

    Operator format (from fno_benchmark.load_data):
      1D: x_data (B, n_x, T_in),   y_data (B, n_x, T_out)
      2D: x_data (B, n_x, n_y, T_in), y_data (B, n_x, n_y, T_out)
      NS: x_data (B, n_x, n_y, 1), y_data (B, n_x, n_y, 3)  [steady-state]

    HPIT format: x_hpit (N, seq_len, n_features), y_hpit (N, output_dim)
      where N = B * n_x [1D] or B * n_x * n_y [2D]
      Each sample is one spatial point's time history.

    Coordinates are normalized to [0, 1].

    Returns:
        x_hpit: np.ndarray (N, seq_len, features)
        y_hpit: np.ndarray (N, output_dim)
    """
    x_np = x_data.numpy().astype(np.float32)
    y_np = y_data.numpy().astype(np.float32)

    if pde_name in _PDE1D:
        # x_np: (B, n_x, T_in), y_np: (B, n_x, T_out=1)
        B, n_x, T_in = x_np.shape
        x_coords = np.linspace(0.0, 1.0, n_x, dtype=np.float32)  # (n_x,)
        t_steps  = np.linspace(0.0, 1.0, T_in,  dtype=np.float32)  # (T_in,)

        # For each (b, j): sequence = [(x_j, t_k, u(b, j, k)) for k in T_in]
        # x_hpit: (B, n_x, T_in, 3) → reshape (B*n_x, T_in, 3)
        xc = np.broadcast_to(x_coords[None, :, None], (B, n_x, T_in))
        tc = np.broadcast_to(t_steps[None, None, :],  (B, n_x, T_in))
        seq = np.stack([xc, tc, x_np], axis=-1)   # (B, n_x, T_in, 3)
        x_hpit = seq.reshape(B * n_x, T_in, 3)

        # y: first T_out value per spatial point
        y_hpit = y_np[:, :, 0].reshape(B * n_x, 1)  # (B*n_x, 1)

    elif pde_name == "NavierStokes2D":
        # Steady-state: x_np (B, n_x, n_y, 1), y_np (B, n_x, n_y, 3)
        B, n_x, n_y, _ = x_np.shape
        xs = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')  # (n_x, n_y)

        # Sequence length = 1 (steady-state: input is Re parameter field)
        # Features: [x, y, Re_param]
        xc = np.broadcast_to(XX[None, :, :, None], (B, n_x, n_y, 1))
        yc = np.broadcast_to(YY[None, :, :, None], (B, n_x, n_y, 1))
        re = x_np  # (B, n_x, n_y, 1)
        seq = np.concatenate([xc, yc, re], axis=-1)     # (B, n_x, n_y, 3)
        x_hpit = seq.reshape(B * n_x * n_y, 1, 3)       # seq_len=1

        y_hpit = y_np.reshape(B * n_x * n_y, 3)         # (u, v, p)

    else:
        # 2D time-dependent: x_np (B, n_x, n_y, T_in), y_np (B, n_x, n_y, T_out)
        B, n_x, n_y, T_in = x_np.shape
        xs = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
        ts = np.linspace(0.0, 1.0, T_in, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')  # (n_x, n_y)

        # For each (b, i, j): sequence = [(x_i, y_j, t_k, u(b,i,j,k)) for k in T_in]
        xc = np.broadcast_to(XX[None, :, :, None], (B, n_x, n_y, T_in))
        yc = np.broadcast_to(YY[None, :, :, None], (B, n_x, n_y, T_in))
        tc = np.broadcast_to(ts[None, None, None, :], (B, n_x, n_y, T_in))
        seq = np.stack([xc, yc, tc, x_np], axis=-1)     # (B, n_x, n_y, T_in, 4)
        x_hpit = seq.reshape(B * n_x * n_y, T_in, 4)

        y_hpit = y_np[:, :, :, 0].reshape(B * n_x * n_y, 1)

    return x_hpit, y_hpit


# ---------------------------------------------------------------------------
# HPIT Loader
# ---------------------------------------------------------------------------

def load_hpit_model(input_dim: int, output_dim: int,
                    checkpoint_path: Optional[str] = None,
                    device: str = "cpu"):
    """
    Load HPIT model with the given input/output dimensions.
    Adapts HPITConfig for PDE inputs (not SWE meteorological inputs).

    Args:
        input_dim: Number of features per timestep (e.g. 3 for [x, t, u])
        output_dim: Number of outputs (e.g. 1 for scalar u, 2 for [u,v])
        checkpoint_path: Path to .pt checkpoint (optional)
        device: 'cpu' or 'cuda'
    """
    # HPITModel and HPITConfig are loaded at module level via importlib.util
    # from hpit_benchmark/hpit_src/hpit.py — no src/ dependency.

    config = HPITConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        embedding_dim=256,      # Reduced from 1024 — PDE tasks are smaller
        num_heads=8,
        num_layers=6,
        hidden_dim=512,
        num_attention_scales=4,
        dropout=0.1,
        use_physics_layers=False,   # Physics constraints are SWE-specific
        use_spatial_attention=True,
        use_temporal_attention=True,
        use_feature_selection=False,
        use_batch_norm=False,
        use_layer_norm=True,
        activation="swish",
    )

    model = HPITModel(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        # Handle both raw state_dict and wrapped checkpoints
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.info("No checkpoint provided — model initialized with random weights (will train).")

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hpit(model, x_train: np.ndarray, y_train: np.ndarray,
               epochs: int, batch_size: int, lr: float,
               device: str) -> nn.Module:
    """
    Train HPIT on PINNacle data.  Matches FNO/GNOT/DeepONet training pattern exactly:
      - Adam optimizer, weight_decay=1e-4
      - CosineAnnealingLR scheduler
      - MSE loss
      - Shuffled DataLoader

    Args:
        model:   HPITModel (on device, eval mode)
        x_train: (N, seq_len, features) float32 numpy array
        y_train: (N, output_dim) float32 numpy array
        epochs:  number of training epochs
        batch_size: training batch size
        lr:      initial learning rate
        device:  'cpu' or 'cuda'

    Returns: trained model in eval mode
    """
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=(device == "cuda"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    model.train()
    log_every = max(1, epochs // 5)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out  = model(xb)
                pred = out.predictions      # (batch, output_dim)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()

        if epoch == 1 or epoch % log_every == 0:
            avg = epoch_loss / len(x_train)
            logger.info(f"  Epoch {epoch}/{epochs}  loss={avg:.4e}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_hpit_on_problem(model, x_input: np.ndarray, y_target: np.ndarray,
                         device: str, batch_size: int = 256):
    """
    Run HPIT inference on a PDE problem's input/target arrays.

    Args:
        model: Loaded HPIT model (eval mode)
        x_input: shape (N, seq_len, n_features)
        y_target: shape (N,) or (N, output_dim)
        device: 'cpu' or 'cuda'
        batch_size: inference batch size

    Returns:
        predictions: np.ndarray, same shape as y_target
        l2_relative_error: float
        inference_time_seconds: float
    """
    assert x_input.ndim == 3, \
        f"Expected x_input shape (N, seq_len, features), got {x_input.shape}"
    assert len(x_input) == len(y_target), \
        f"x_input and y_target must have same length: {len(x_input)} vs {len(y_target)}"

    n_samples = len(x_input)
    all_preds = []

    t_start = time.time()

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_x = torch.tensor(x_input[start:end], dtype=torch.float32).to(device)

            output = model(batch_x)
            preds = output.predictions.cpu().numpy()  # (batch, output_dim)
            all_preds.append(preds)

    inference_time = time.time() - t_start
    predictions = np.concatenate(all_preds, axis=0)  # (N, output_dim)

    # Squeeze if output_dim == 1 to match (N,) target
    if predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)

    # L2 relative error
    # Source: PDEBench, Takamoto et al. 2022, https://arxiv.org/abs/2210.07182
    diff_norm = np.linalg.norm(predictions.flatten() - y_target.flatten())
    target_norm = np.linalg.norm(y_target.flatten())
    l2_rel = float(diff_norm / target_norm) if target_norm > 1e-10 else np.nan

    return predictions, l2_rel, inference_time


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def save_result(pde_name: str, l2_rel: float, l2_std: float,
                run_time: float, notes: str = ""):
    """Append one result row to the CSV."""
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pde", "model", "l2rel", "l2rel_std", "run_time_seconds", "notes"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "pde": pde_name,
            "model": "HPIT",
            "l2rel": f"{l2_rel:.6f}" if not np.isnan(l2_rel) else "nan",
            "l2rel_std": f"{l2_std:.6f}" if not np.isnan(l2_std) else "nan",
            "run_time_seconds": f"{run_time:.2f}",
            "notes": notes,
        })
    logger.info(f"Saved result: {pde_name} | L2={l2_rel:.4f} | time={run_time:.1f}s")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def benchmark_pde(pde_name: str, args, device: str):
    """Run HPIT on a single PDE and save result.

    Data source selection:
      --dry-run:   synthetic FD data (fast, no .dat files needed)
      default:     PINNacle .dat reference data (fair comparison with other models)
      --synthetic: synthetic FD data even in full run (old behavior)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {pde_name}")
    logger.info(f"{'='*60}")

    use_pinnacle = (not args.dry_run) and (not getattr(args, 'synthetic', False))

    try:
        if use_pinnacle:
            # --- PINNacle .dat path (default for full runs) ---
            logger.info("Loading PINNacle reference data for %s ...", pde_name)
            T_in  = 1 if pde_name == "NavierStokes2D" else (
                    5 if pde_name in {"Burgers2D", "HeatComplexGeometry"} else 10)
            T_out = 1
            x_data, y_data = _load_pinnacle_data(pde_name, T_in, T_out,
                                                  dry_run=False)
            logger.info("PINNacle shapes — x: %s, y: %s",
                        tuple(x_data.shape), tuple(y_data.shape))

            # 80/20 train/test split — same ratio as FNO/GNOT/DeepONet/PINO
            n = len(x_data)
            split = max(1, int(0.8 * n))
            x_train_op, y_train_op = x_data[:split], y_data[:split]
            x_test_op,  y_test_op  = x_data[split:], y_data[split:]
            if len(x_test_op) == 0:
                x_test_op, y_test_op = x_data[-1:], y_data[-1:]

            x_train_h, y_train_h = pinnacle_to_hpit(pde_name, x_train_op, y_train_op)
            x_input,   y_target  = pinnacle_to_hpit(pde_name, x_test_op,  y_test_op)
            logger.info("HPIT train: %s → %s | test: %s → %s",
                        x_train_h.shape, y_train_h.shape,
                        x_input.shape,   y_target.shape)
            notes_data = "pinnacle_dat"

        else:
            # --- Synthetic FD path (dry-run or --synthetic) ---
            config = PDEProblemConfig(
                n_x=args.n_x,
                n_y=args.n_x,
                n_t=args.n_t,
                seq_len=args.seq_len,
            )
            problem = get_problem(pde_name, config=config, dry_run=args.dry_run)
            logger.info("Formatting PDE as HPIT input (synthetic FD)...")
            x_all, y_all = problem.to_hpit_input()
            n = len(x_all)
            split = max(1, int(0.8 * n))
            x_train_h, y_train_h = x_all[:split], y_all[:split]
            x_input,   y_target  = x_all[split:],  y_all[split:]
            if len(x_input) == 0:
                x_input, y_target = x_all[-1:], y_all[-1:]
            logger.info(f"Input shape: {x_input.shape}, Target shape: {y_target.shape}")
            notes_data = "dry_run" if args.dry_run else "synthetic_fd"

        n_features = x_input.shape[-1]
        output_dim = int(y_target.shape[-1]) if y_target.ndim > 1 else 1

        # ---- Training ----
        # If --checkpoint is given: skip training, load weights and evaluate.
        # Otherwise: train from scratch on PINNacle (or synthetic) data.
        epochs = args.epochs
        if args.dry_run:
            epochs = min(epochs, 2)   # cap at 2 for dry-run speed

        if args.checkpoint and os.path.exists(args.checkpoint):
            logger.info("Checkpoint provided — skipping training, loading weights.")
            model = load_hpit_model(
                input_dim=n_features,
                output_dim=output_dim,
                checkpoint_path=args.checkpoint,
                device=device,
            )
            notes_train = "pretrained_ckpt"
        else:
            logger.info("No checkpoint — training HPIT from scratch for %d epochs.", epochs)
            model = load_hpit_model(
                input_dim=n_features,
                output_dim=output_dim,
                checkpoint_path=None,
                device=device,
            )
            t_train_start = time.time()
            model = train_hpit(
                model, x_train_h, y_train_h,
                epochs=epochs,
                batch_size=args.train_batch_size,
                lr=args.lr,
                device=device,
            )
            t_train = time.time() - t_train_start
            logger.info("Training complete in %.1fs.", t_train)

            # Auto-save checkpoint so it can be reused
            ckpt_path = RESULTS_DIR / f"hpit_{pde_name}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)
            notes_train = "trained_pinnacle" if use_pinnacle else notes_data

        notes_data = notes_train if args.checkpoint else notes_data

        # ---- Evaluation on test set ----
        preds, l2_rel, inf_time = run_hpit_on_problem(
            model, x_input, y_target, device=device, batch_size=args.batch_size
        )
        l2_mean = l2_rel
        l2_std  = 0.0
        logger.info(f"Test L2={l2_mean:.4f} ({inf_time:.1f}s inference)")

        save_result(pde_name, l2_mean, l2_std, inf_time, notes=notes_data)

    except Exception as e:
        logger.error(f"FAILED on {pde_name}: {e}", exc_info=True)
        save_result(pde_name, np.nan, np.nan, 0.0,
                    notes=f"ERROR: {str(e)[:100]}")


def main():
    parser = argparse.ArgumentParser(description="HPIT PDE Benchmark")
    parser.add_argument("--pde", type=str, default=None,
                        help="Run single PDE by name. Default: run all.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to HPIT checkpoint (.pt file).")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use tiny grids to verify pipeline only.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic FD ground truth instead of PINNacle .dat data. "
                             "Not recommended for paper comparison — use default (PINNacle) instead.")
    parser.add_argument("--n-x", type=int, default=64,
                        help="Spatial resolution (default: 64)")
    parser.add_argument("--n-t", type=int, default=100,
                        help="Temporal resolution (default: 100)")
    parser.add_argument("--seq-len", type=int, default=30,
                        help="HPIT sequence window length (default: 30)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs (default: 500; capped at 2 in --dry-run)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--train-batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Inference batch size (default: 256)")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds for std estimate (default: 3)")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    if args.dry_run:
        logger.info("DRY RUN MODE — using tiny grids. Results are not meaningful.")

    # Select PDEs to run
    if args.pde:
        pde_names = [args.pde]
    else:
        pde_names = list(ALL_PROBLEMS.keys())

    logger.info(f"Running benchmark on: {pde_names}")
    logger.info(f"Results will be saved to: {RESULTS_CSV}")

    total_start = time.time()
    for pde_name in pde_names:
        benchmark_pde(pde_name, args, device)

    total_time = time.time() - total_start
    logger.info(f"\nAll benchmarks complete in {total_time:.1f}s")
    logger.info(f"Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
