"""
mamba_no_benchmark.py — Benchmarks Mamba-NO on PDE families using PINNacle data.

Usage:
    python hpit_benchmark/mamba_no_benchmark.py --dry-run
    python hpit_benchmark/mamba_no_benchmark.py --pde Burgers1D

Output:
    hpit_benchmark/results/mamba_no_results.csv

Architecture:
    Mamba-NO = stack of Mamba SSM layers acting as a neural operator.
    Spatial dimension treated as sequence; temporal snapshots as channels.
    Pure-PyTorch selective scan — no CUDA kernels needed for dry-run.
    If mamba_ssm is installed (GPU), automatically uses the optimised fast path.

Sources:
    Mamba: Gu & Dao, ICLR 2024 — https://arxiv.org/abs/2312.00752
    state-spaces/mamba: https://github.com/state-spaces/mamba
    LaMO: M3RG-IITD, 2025 — https://github.com/M3RG-IITD/LaMO
    PINNacle: Hao et al., NeurIPS 2024 — https://arxiv.org/abs/2306.08827
"""

# Source: Mamba, Gu & Dao, ICLR 2024, https://arxiv.org/abs/2312.00752

import argparse
import csv
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

BENCHMARK_DIR    = Path(__file__).resolve().parent
MAMBA_SRC_DIR    = BENCHMARK_DIR / "mamba_no_src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_mamba = _load_module("mamba_no_src.mamba_no", MAMBA_SRC_DIR / "mamba_no.py")
MambaOperator1d = _mod_mamba.MambaOperator1d
MambaOperator2d = _mod_mamba.MambaOperator2d

_mod_fno = _load_module("fno_benchmark_mamba", BENCHMARK_DIR / "fno_benchmark.py")
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
RESULTS_CSV = RESULTS_DIR / "mamba_no_results.csv"

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(pde_name: str, T_in: int, T_out: int,
                d_model: int, d_state: int, n_layers: int) -> nn.Module:
    if pde_name == "NavierStokes2D":
        # Steady-state: treat as 2D with T_in=1, T_out=3
        return MambaOperator2d(T_in=1, T_out=3, d_model=d_model,
                                d_state=d_state, n_layers=n_layers)
    elif pde_name in _PDE1D:
        return MambaOperator1d(T_in=T_in, T_out=T_out, d_model=d_model,
                                d_state=d_state, n_layers=n_layers)
    else:
        return MambaOperator2d(T_in=T_in, T_out=T_out, d_model=d_model,
                                d_state=d_state, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mamba(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int, lr: float, batch_size: int, device: str) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()

    dataset = TensorDataset(x_train, y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=(device == "cuda"))

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            logger.info("  Epoch %d/%d — loss=%.6f", epoch, epochs,
                        epoch_loss / len(x_train))
    return model


def evaluate_mamba(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor,
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
            "model":            "Mamba-NO",
            "l2rel":            f"{l2:.6f}" if not np.isnan(l2) else "nan",
            "run_time_seconds": f"{run_time:.2f}",
            "notes":            notes,
        })
    logger.info("Saved result: %s | L2=%.4f | time=%.1fs", pde_name, l2, run_time)


# ---------------------------------------------------------------------------
# Main per-PDE runner
# ---------------------------------------------------------------------------

def run_mamba_on_pde(pde_name: str, args, device: str):
    logger.info("\n%s\nMamba-NO benchmark: %s\n%s", "="*60, pde_name, "="*60)

    T_in  = 5  if args.dry_run else 10
    T_out = 1
    if pde_name == "NavierStokes2D":
        T_in = 1
    elif pde_name in {"Burgers2D", "HeatComplexGeometry"} and not args.dry_run:
        T_in = 5

    d_model  = 16  if args.dry_run else args.d_model
    d_state  = 4   if args.dry_run else args.d_state
    n_layers = 2   if args.dry_run else args.n_layers
    epochs   = 2   if args.dry_run else args.epochs

    try:
        x_data, y_data = load_data(pde_name, T_in, T_out, dry_run=args.dry_run)
        logger.info("Data shapes — x: %s, y: %s",
                    tuple(x_data.shape), tuple(y_data.shape))

        n     = len(x_data)
        split = max(1, int(0.8 * n))
        x_train, y_train = x_data[:split], y_data[:split]
        x_test,  y_test  = x_data[split:],  y_data[split:]
        if len(x_test) == 0:
            x_test, y_test = x_data[-1:], y_data[-1:]

        model = build_model(pde_name, T_in, T_out, d_model, d_state, n_layers)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Mamba-NO parameters: %d", n_params)

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
            logger.info("Training Mamba-NO for %d epochs...", epochs)
            model = train_mamba(model, x_train, y_train,
                                epochs=epochs, lr=args.lr,
                                batch_size=args.batch_size, device=device)
            ckpt_save = RESULTS_DIR / f"mamba_no_{pde_name}_ckpt.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_save)
            logger.info("Checkpoint saved to %s", ckpt_save)

        l2       = evaluate_mamba(model, x_test, y_test, device=device,
                                   batch_size=args.batch_size)
        run_time = time.time() - t_start

        logger.info("Mamba-NO on %s: L2=%.4f (%.1fs)", pde_name, l2, run_time)
        notes = "dry_run" if args.dry_run else "pinnacle_dat"
        save_result(pde_name, l2, run_time, notes=notes)

    except Exception as e:
        logger.error("FAILED on %s: %s", pde_name, e, exc_info=True)
        save_result(pde_name, np.nan, 0.0, notes=f"ERROR: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_PDES = list(_PDE1D | _PDE2D)


def main():
    parser = argparse.ArgumentParser(description="Mamba-NO PDE Benchmark (PINNacle data)")
    parser.add_argument("--pde",        type=str,   default=None)
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--d-model",    type=int,   default=64,
                        help="Mamba model dimension (default: 64)")
    parser.add_argument("--d-state",    type=int,   default=16,
                        help="SSM state dimension N (default: 16)")
    parser.add_argument("--n-layers",   type=int,   default=4,
                        help="Number of Mamba layers (default: 4)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    pdes = [args.pde] if args.pde else _ALL_PDES
    logger.info("Mamba-NO benchmark PDEs: %s", pdes)
    logger.info("Results → %s", RESULTS_CSV)

    for pde in pdes:
        run_mamba_on_pde(pde, args, device)

    logger.info("All Mamba-NO benchmarks complete.")


if __name__ == "__main__":
    main()
