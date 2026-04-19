"""
pino_benchmark.py — Benchmarks PINO on PDE families using PINNacle reference data.

Usage:
    python hpit_benchmark/pino_benchmark.py --dry-run
    python hpit_benchmark/pino_benchmark.py --pde Burgers1D

Output:
    hpit_benchmark/results/pino_results.csv

Notes:
    - PINO = FNO backbone + physics residual loss (λ * ||F(u)||^2).
    - Same data, same evaluation metric as fno_benchmark.py.
    - Key difference from FNO: training uses BOTH data loss AND PDE residual.
    - If PINO outperforms FNO, it validates physics-informed operator learning.

Sources:
    PINO: Li et al., 2021, https://arxiv.org/abs/2111.08907
    FNO: Li et al., ICLR 2021, https://arxiv.org/abs/2010.08895
    PINNacle: Hao et al., NeurIPS 2024, https://arxiv.org/abs/2306.08827
"""

# Source: PINO, Li et al. 2021, https://arxiv.org/abs/2111.08907
# FNO backbone: Li et al. ICLR 2021, https://arxiv.org/abs/2010.08895

import argparse
import csv
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
PINO_SRC_DIR  = BENCHMARK_DIR / "pino_src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_pino = _load_module("pino_src.pino", PINO_SRC_DIR / "pino.py")
PINO1d = _mod_pino.PINO1d
PINO2d = _mod_pino.PINO2d

_mod_fno = _load_module("fno_benchmark_pino", BENCHMARK_DIR / "fno_benchmark.py")
load_data         = _mod_fno.load_data
l2_relative_error = _mod_fno.l2_relative_error
_PDE1D = _mod_fno._PDE1D
_PDE2D = _mod_fno._PDE2D

# ---------------------------------------------------------------------------
# Logging and results
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "pino_results.csv"

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(pde_name: str, T_in: int, T_out: int,
                modes: int, width: int, n_layers: int,
                physics_weight: float) -> nn.Module:
    if pde_name == "NavierStokes2D":
        return PINO2d(pde_name=pde_name, physics_weight=physics_weight,
                      modes1=modes, modes2=modes, width=width,
                      T_in=1, T_out=3, n_layers=n_layers)
    elif pde_name in _PDE1D:
        return PINO1d(pde_name=pde_name, physics_weight=physics_weight,
                      modes=modes, width=width, T_in=T_in, T_out=T_out,
                      n_layers=n_layers)
    else:
        return PINO2d(pde_name=pde_name, physics_weight=physics_weight,
                      modes1=modes, modes2=modes, width=width,
                      T_in=T_in, T_out=T_out, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Training — PINO loss = data loss + physics loss
# ---------------------------------------------------------------------------

def train_pino(model, x_train: torch.Tensor, y_train: torch.Tensor,
               epochs: int, lr: float, batch_size: int, device: str,
               patience: int = 50, checkpoint_path: Optional[str] = None):
    """Train PINO with Adam + data+physics loss + early stopping + checkpointing."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    mse = nn.MSELoss()

    dataset = TensorDataset(x_train, y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=(device == "cuda"))

    min_delta  = 1e-5
    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    log_every  = min(10, max(1, epochs // 10))

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            # PINO total loss: supervised + physics residual
            data_loss = mse(pred, yb)
            phys_loss = model.compute_physics_loss(pred)
            loss = data_loss + phys_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg = epoch_loss / len(x_train)
        scheduler.step(avg)

        # Early stopping
        if avg < best_loss - min_delta:
            best_loss  = avg
            no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save({"model_state_dict": best_state}, checkpoint_path)
        else:
            no_improve += 1

        if epoch == 1 or epoch % log_every == 0:
            logger.info("  Epoch %d/%d — loss=%.6f  best=%.6f  patience=%d/%d",
                        epoch, epochs, avg, best_loss, no_improve, patience)

        if no_improve >= patience:
            logger.info("  Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, patience)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("  Best weights restored (loss=%.6f)", best_loss)

    return model


def evaluate_pino(model, x_test: torch.Tensor, y_test: torch.Tensor,
                  device: str, batch_size: int = 32) -> float:
    model.eval()
    all_preds = []
    loader = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            all_preds.append(model(xb.to(device)).cpu())
    preds = torch.cat(all_preds, dim=0)
    return l2_relative_error(preds, y_test)


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def save_result(pde_name: str, l2: float, run_time: float, notes: str = ""):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pde", "model", "l2rel", "run_time_seconds", "notes"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "pde":              pde_name,
            "model":            "PINO",
            "l2rel":            f"{l2:.6f}" if not np.isnan(l2) else "nan",
            "run_time_seconds": f"{run_time:.2f}",
            "notes":            notes,
        })
    logger.info("Saved result: %s | L2=%.4f | time=%.1fs", pde_name, l2, run_time)


# ---------------------------------------------------------------------------
# Main per-PDE runner
# ---------------------------------------------------------------------------

def run_pino_on_pde(pde_name: str, args, device: str):
    logger.info("\n%s\nPINO benchmark: %s\n%s", "="*60, pde_name, "="*60)

    T_in  = 5  if args.dry_run else 10
    T_out = 1
    if pde_name == "NavierStokes2D":
        T_in = 1
    elif pde_name in {"Burgers2D", "HeatComplexGeometry"} and not args.dry_run:
        T_in = 5

    modes  = 4  if args.dry_run else args.modes
    width  = 8  if args.dry_run else args.width
    epochs = 2  if args.dry_run else args.epochs
    phys_w = args.physics_weight
    # KS 4th-order spectral residual inflates physics loss ~1000x vs other PDEs
    _PHYS_W_OVERRIDE = {"KuramotoSivashinsky": 1e-3}
    phys_w = _PHYS_W_OVERRIDE.get(pde_name, phys_w)

    try:
        x_data, y_data = load_data(pde_name, T_in, T_out, dry_run=args.dry_run)
        logger.info("Data shapes — x: %s, y: %s",
                    tuple(x_data.shape), tuple(y_data.shape))

        # Detect actual T_in/T_out from loaded data (Burgers2D y has 2 channels)
        T_in  = x_data.shape[-1]
        T_out = y_data.shape[-1]

        n     = len(x_data)
        split = max(1, int(0.8 * n))
        x_train, y_train = x_data[:split], y_data[:split]
        x_test,  y_test  = x_data[split:],  y_data[split:]
        if len(x_test) == 0:
            x_test, y_test = x_data[-1:], y_data[-1:]

        model = build_model(pde_name, T_in, T_out, modes, width,
                            args.n_layers, phys_w)
        logger.info("PINO parameters: %d",
                    sum(p.numel() for p in model.parameters()))

        ckpt_path = args.checkpoint
        t_start   = time.time()

        if ckpt_path and os.path.exists(ckpt_path):
            logger.info("Loading checkpoint: %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            model = model.to(device)
        else:
            if ckpt_path:
                logger.warning("Checkpoint not found: %s — training.", ckpt_path)
            logger.info("Training PINO for %d epochs, physics_weight=%.3f",
                        epochs, phys_w)
            ckpt_save = RESULTS_DIR / f"pino_{pde_name}_ckpt.pt"
            model = train_pino(model, x_train, y_train,
                               epochs=epochs, lr=args.lr,
                               batch_size=args.batch_size, device=device,
                               patience=args.patience,
                               checkpoint_path=str(ckpt_save))
            logger.info("Checkpoint saved to %s", ckpt_save)

        l2       = evaluate_pino(model, x_test, y_test, device=device,
                                 batch_size=args.batch_size)
        run_time = time.time() - t_start

        logger.info("PINO on %s: L2=%.4f (%.1fs)", pde_name, l2, run_time)
        notes = "dry_run" if args.dry_run else f"pinnacle_dat|phys_w={phys_w}"
        save_result(pde_name, l2, run_time, notes=notes)

    except Exception as e:
        logger.error("FAILED on %s: %s", pde_name, e, exc_info=True)
        save_result(pde_name, np.nan, 0.0, notes=f"ERROR: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_PDES = list(_PDE1D | _PDE2D)


def main():
    parser = argparse.ArgumentParser(description="PINO PDE Benchmark (PINNacle data)")
    parser.add_argument("--pde",            type=str,   default=None)
    parser.add_argument("--checkpoint",     type=str,   default=None)
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--epochs",         type=int,   default=500)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--patience",       type=int,   default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--modes",          type=int,   default=16)
    parser.add_argument("--width",          type=int,   default=64)
    parser.add_argument("--n-layers",       type=int,   default=4)
    parser.add_argument("--physics-weight", type=float, default=0.1,
                        help="Weight λ for the physics residual loss (default: 0.1)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    pdes = [args.pde] if args.pde else _ALL_PDES
    logger.info("PINO benchmark PDEs: %s", pdes)
    logger.info("Results → %s", RESULTS_CSV)

    for pde in pdes:
        run_pino_on_pde(pde, args, device)

    logger.info("All PINO benchmarks complete.")


if __name__ == "__main__":
    main()
