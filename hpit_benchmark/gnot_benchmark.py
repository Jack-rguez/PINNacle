"""
gnot_benchmark.py — Benchmarks GNOT on PDE families using PINNacle reference data.

Usage:
    # Dry run (random tensors, fast pipeline check):
    python hpit_benchmark/gnot_benchmark.py --dry-run

    # Real run (uses PINNacle ref/*.dat files):
    python hpit_benchmark/gnot_benchmark.py --pde Burgers1D

    # Load checkpoint:
    python hpit_benchmark/gnot_benchmark.py --pde Burgers1D --checkpoint path/to/ckpt.pt

Output:
    hpit_benchmark/results/gnot_results.csv

Notes:
    - Does NOT modify any original PINNacle or HPIT files.
    - GNOT uses the same PINNacle data and L2 metric as fno_benchmark.py.
    - Key difference from FNO: GNOT operates on unstructured point clouds
      via cross-attention, so it is naturally suited for the PINNacle .dat format.

Sources:
    GNOT: Hao et al., ICML 2023, https://arxiv.org/abs/2302.14376
    Code: https://github.com/thu-ml/GNOT
    PINNacle data: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
"""

# Source: GNOT, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
# Code reference: https://github.com/thu-ml/GNOT

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
# File-based import of gnot_src and shared data utilities from fno_benchmark
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
GNOT_SRC_DIR  = BENCHMARK_DIR / "gnot_src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_gnot = _load_module("gnot_src.gnot", GNOT_SRC_DIR / "gnot.py")
GNOT1d = _mod_gnot.GNOT1d
GNOT2d = _mod_gnot.GNOT2d

# Reuse all data loading from fno_benchmark (same PINNacle data, same format)
_mod_fno = _load_module("fno_benchmark_data", BENCHMARK_DIR / "fno_benchmark.py")
load_data              = _mod_fno.load_data
make_dry_run_data      = _mod_fno.make_dry_run_data
l2_relative_error      = _mod_fno.l2_relative_error

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
RESULTS_CSV = RESULTS_DIR / "gnot_results.csv"

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(pde_name: str, T_in: int, T_out: int,
                n_hidden: int, n_head: int, n_layers: int) -> nn.Module:
    """Build GNOT1d or GNOT2d depending on PDE spatial dimension."""
    if pde_name == "NavierStokes2D":
        # Steady-state: T_in=1 input channel (a-param), T_out=3 output (u,v,p)
        return GNOT2d(T_in=1, T_out=3, n_hidden=n_hidden,
                      n_head=n_head, n_layers=n_layers)
    elif pde_name in _PDE1D:
        return GNOT1d(T_in=T_in, T_out=T_out, n_hidden=n_hidden,
                      n_head=n_head, n_layers=n_layers)
    else:
        return GNOT2d(T_in=T_in, T_out=T_out, n_hidden=n_hidden,
                      n_head=n_head, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gnot(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
               epochs: int, lr: float, batch_size: int, device: str) -> nn.Module:
    """Train GNOT with Adam + relative L2 loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(x_train, y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            logger.info("  Epoch %d/%d — loss=%.6f", epoch, epochs,
                        epoch_loss / len(x_train))

    return model


def evaluate_gnot(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor,
                  device: str, batch_size: int = 32) -> float:
    """Evaluate GNOT using L2 relative error."""
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
            "pde":               pde_name,
            "model":             "GNOT",
            "l2rel":             f"{l2:.6f}" if not np.isnan(l2) else "nan",
            "run_time_seconds":  f"{run_time:.2f}",
            "notes":             notes,
        })
    logger.info("Saved result: %s | L2=%.4f | time=%.1fs", pde_name, l2, run_time)


# ---------------------------------------------------------------------------
# Main per-PDE runner
# ---------------------------------------------------------------------------

def run_gnot_on_pde(pde_name: str, args, device: str):
    logger.info("\n%s\nGNOT benchmark: %s\n%s", "="*60, pde_name, "="*60)

    T_in  = 5  if args.dry_run else 10
    T_out = 1
    if pde_name == "NavierStokes2D":
        T_in = 1
    elif pde_name in {"Burgers2D", "HeatComplexGeometry"} and not args.dry_run:
        T_in = 5

    n_hidden = 16 if args.dry_run else args.n_hidden
    n_head   = 1  if args.dry_run else args.n_head
    epochs   = 2  if args.dry_run else args.epochs

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

        model = build_model(pde_name, T_in, T_out, n_hidden, n_head, args.n_layers)
        logger.info("GNOT parameters: %d",
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
            logger.info("Training GNOT for %d epochs...", epochs)
            model = train_gnot(model, x_train, y_train,
                               epochs=epochs, lr=args.lr,
                               batch_size=args.batch_size, device=device)
            ckpt_save = RESULTS_DIR / f"gnot_{pde_name}_ckpt.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_save)
            logger.info("Checkpoint saved to %s", ckpt_save)

        l2       = evaluate_gnot(model, x_test, y_test, device=device,
                                 batch_size=args.batch_size)
        run_time = time.time() - t_start

        logger.info("GNOT on %s: L2=%.4f (%.1fs)", pde_name, l2, run_time)
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
    parser = argparse.ArgumentParser(description="GNOT PDE Benchmark (PINNacle data)")
    parser.add_argument("--pde", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Random tensors — fast pipeline check.")
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=16,
                        help="Smaller default than FNO due to GNOT attention cost.")
    parser.add_argument("--n-hidden",   type=int,   default=64,
                        help="GNOT hidden dimension (default: 64)")
    parser.add_argument("--n-head",     type=int,   default=1,
                        help="Number of attention heads (default: 1)")
    parser.add_argument("--n-layers",   type=int,   default=3,
                        help="Number of GNOT blocks (default: 3)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    pdes = [args.pde] if args.pde else _ALL_PDES
    logger.info("GNOT benchmark PDEs: %s", pdes)
    logger.info("Results → %s", RESULTS_CSV)

    for pde in pdes:
        run_gnot_on_pde(pde, args, device)

    logger.info("All GNOT benchmarks complete.")


if __name__ == "__main__":
    main()
