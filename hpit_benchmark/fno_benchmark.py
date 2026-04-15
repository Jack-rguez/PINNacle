"""
fno_benchmark.py — Benchmarks FNO on PDE families using PINNacle reference data.

Usage:
    # Dry run (random tensors, fast pipeline check):
    python hpit_benchmark/fno_benchmark.py --dry-run

    # Real run (uses PINNacle ref/*.dat files for training and evaluation):
    python hpit_benchmark/fno_benchmark.py --pde Burgers1D

    # Load checkpoint instead of training:
    python hpit_benchmark/fno_benchmark.py --pde Burgers1D --checkpoint path/to/ckpt.pt

Output:
    hpit_benchmark/results/fno_results.csv

Notes:
    - Does NOT modify any existing PINNacle or HPIT files.
    - Training uses PINNacle PDE parameters (same domain, nu, BCs as PINNacle PINN runs).
    - Evaluation uses PINNacle ref/*.dat COMSOL reference solutions as ground truth.
      This ensures direct comparability with PINNacle PINN benchmark results.
    - L2 relative error: ||u_pred - u_true||_2 / ||u_true||_2

Sources:
    FNO architecture: Li et al. ICLR 2021, https://arxiv.org/abs/2010.08895
    PINNacle benchmark data: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
"""

# Source: FNO, Li et al. ICLR 2021, https://arxiv.org/abs/2010.08895
# Adapted from https://github.com/neuraloperator/neuraloperator
# PINNacle data: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
# Reference .dat files are COMSOL FEM solutions included with PINNacle.

import argparse
import csv
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# File-based import of fno_src — no sys.path manipulation
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
FNO_SRC_DIR   = BENCHMARK_DIR / "fno_src"
PINNACLE_ROOT = BENCHMARK_DIR.parent
REF_DIR       = PINNACLE_ROOT / "ref"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_fno = _load_module("fno_src.fno", FNO_SRC_DIR / "fno.py")
FNO1d = _mod_fno.FNO1d
FNO2d = _mod_fno.FNO2d

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
RESULTS_CSV = RESULTS_DIR / "fno_results.csv"

# ---------------------------------------------------------------------------
# PDE metadata — matches PINNacle src/pde/ exactly
# ---------------------------------------------------------------------------

_PDE1D = {"Burgers1D", "KuramotoSivashinsky"}
_PDE2D = {"Burgers2D", "HeatComplexGeometry", "NavierStokes2D"}

# PINNacle ref file paths relative to PINNACLE_ROOT
_PINNACLE_REF = {
    "Burgers1D":            REF_DIR / "burgers1d.dat",
    "KuramotoSivashinsky":  REF_DIR / "Kuramoto_Sivashinsky.dat",
    "Burgers2D":            REF_DIR / "burgers2d_0.dat",
    "HeatComplexGeometry":  REF_DIR / "heat_complex.dat",
    "NavierStokes2D":       [REF_DIR / f"lid_driven_a{a}.dat"
                             for a in [2, 4, 6, 8, 10, 16, 32]],
}

# PDE physical parameters (from PINNacle src/pde/)
_BURGERS1D_NU  = 0.01 / np.pi
_BURGERS1D_X   = (-1.0, 1.0)
_BURGERS1D_T   = (0.0, 1.0)

_KS_X          = (0.0, 2 * np.pi)   # PINNacle KS bbox
_KS_T          = (0.0, 1.0)
_KS_ALPHA      = 100.0 / 16.0
_KS_BETA       = 100.0 / (16.0 ** 2)
_KS_GAMMA      = 100.0 / (16.0 ** 4)

_BURGERS2D_NU  = 0.001
_BURGERS2D_L   = 4.0
_BURGERS2D_T   = (0.0, 1.0)

_HEAT_BBOX     = (-8.0, 8.0, -12.0, 12.0, 0.0, 3.0)  # x0,x1,y0,y1,t0,t1
_NS_BBOX       = (0.0, 1.0, 0.0, 1.0)                  # x0,x1,y0,y1

# ---------------------------------------------------------------------------
# Utility: load PINNacle .dat files (COMSOL export, comment lines start with %)
# ---------------------------------------------------------------------------

def _load_dat(path: Path) -> np.ndarray:
    """Load a PINNacle COMSOL .dat file as a float32 array."""
    rows = []
    with open(str(path), "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# PDE-specific data builders (PINNacle data → FNO (x_data, y_data) format)
# ---------------------------------------------------------------------------

def _burgers1d_fd(n_x: int, n_t: int, ic: np.ndarray) -> Optional[np.ndarray]:
    """
    1D Burgers FD solver (upwind + central diffusion) with CFL stability.
    IC shape: (n_x,). Returns shape: (n_t, n_x), or None if unstable.

    Matches PINNacle Burgers1D exactly:
        u_t + u*u_x = nu*u_xx
        x in [-1,1], t in [0,1], u(-1)=u(1)=0, nu=0.01/pi
    """
    nu   = _BURGERS1D_NU
    x    = np.linspace(*_BURGERS1D_X, n_x)
    dx   = float(x[1] - x[0])
    # CFL-stable dt: advective limit + diffusive limit
    u_max = max(float(np.max(np.abs(ic))), 1e-8)
    dt_adv  = 0.4 * dx / u_max
    dt_diff = 0.4 * dx**2 / nu
    dt_internal = min(dt_adv, dt_diff)

    t_total = _BURGERS1D_T[1]
    n_steps = int(np.ceil(t_total / dt_internal))
    dt_internal = t_total / n_steps

    # Sample at n_t evenly spaced snapshots
    snap_every = max(1, n_steps // (n_t - 1))
    sol = np.zeros((n_t, n_x), dtype=np.float32)
    u = ic.astype(np.float64).copy()
    sol[0] = u.astype(np.float32)
    snap_idx = 1

    for i in range(1, n_steps + 1):
        adv = np.where(u[1:-1] > 0,
                       u[1:-1] * (u[1:-1] - u[:-2]) / dx,
                       u[1:-1] * (u[2:] - u[1:-1]) / dx)
        diff = nu * (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        u[1:-1] = u[1:-1] - dt_internal * adv + dt_internal * diff
        u[0] = u[-1] = 0.0
        if i % snap_every == 0 and snap_idx < n_t:
            sol[snap_idx] = u.astype(np.float32)
            snap_idx += 1
        if not np.isfinite(u).all():
            return None   # diverged

    # Fill any unfilled snapshots with last valid value
    for k in range(snap_idx, n_t):
        sol[k] = sol[snap_idx - 1]

    return sol


def _build_burgers1d_data(n_x: int = 128, n_t: int = 50, n_train: int = 200,
                           T_in: int = 10, T_out: int = 1,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Burgers1D training data and load PINNacle reference as test.

    Training ICs: perturbations around the PINNacle IC -sin(pi*x).
    Test: PINNacle COMSOL reference solution.

    Returns (x_all, y_all) merged — caller splits train/test.
    """
    ref_path = _PINNACLE_REF["Burgers1D"]
    x_grid = np.linspace(*_BURGERS1D_X, n_x)
    rng = np.random.default_rng(42)

    # Reference IC: -sin(pi*x) (PINNacle default)
    ic_ref = -np.sin(np.pi * x_grid).astype(np.float32)

    # ---- Test: PINNacle COMSOL reference -----------------------------------
    # burgers1d.dat: rows=101 spatial nodes, cols=[x, u@t=0..1 in 11 steps]
    dat = _load_dat(ref_path)           # (101, 12)
    x_dat = dat[:, 0]
    u_dat = dat[:, 1:]                  # (101, 11) — t=0,0.1,...,1.0
    # Interpolate onto our n_x × n_t grid
    t_dat = np.linspace(0, 1, 11)
    sol_ref = np.zeros((11, n_x), dtype=np.float32)
    for ti in range(11):
        sol_ref[ti] = np.interp(x_grid, x_dat, u_dat[:, ti])
    # Pad to n_t if needed using linear interpolation
    if n_t != 11:
        t_fine = np.linspace(0, 1, n_t)
        sol_full = np.zeros((n_t, n_x), dtype=np.float32)
        for xi in range(n_x):
            sol_full[:, xi] = np.interp(t_fine, t_dat, sol_ref[:, xi])
        sol_ref = sol_full

    # ---- Training: varied ICs -----------------------------------------------
    x_list, y_list = [], []

    def _windows(sol):
        """Extract sliding-window samples from a trajectory."""
        for t_start in range(0, n_t - T_in - T_out + 1):
            inp = sol[t_start:t_start + T_in].T        # (n_x, T_in)
            out = sol[t_start + T_in:t_start + T_in + T_out].T  # (n_x, T_out)
            x_list.append(inp)
            y_list.append(out)

    # Add windows from reference trajectory
    _windows(sol_ref)

    # Generate n_train synthetic trajectories with CFL-stable FD solver
    generated = 0
    attempts  = 0
    while generated < n_train and attempts < n_train * 3:
        attempts += 1
        # Random perturbation of the PINNacle IC: -sin(pi*x)
        eps   = rng.uniform(-0.3, 0.3)
        k     = rng.integers(1, 4)
        phase = rng.uniform(0, np.pi)
        ic = (-np.sin(np.pi * x_grid) +
              eps * np.sin(k * np.pi * x_grid + phase)).astype(np.float64)
        ic = np.clip(ic, -1.5, 1.5)
        sol = _burgers1d_fd(n_x, n_t, ic)
        if sol is None:
            continue   # skip unstable trajectory
        _windows(sol)
        generated += 1
    logger.info("Burgers1D: generated %d/%d training trajectories", generated, n_train)

    x_data = torch.tensor(np.array(x_list), dtype=torch.float32)
    y_data = torch.tensor(np.array(y_list), dtype=torch.float32)
    return x_data, y_data


def _build_ks_data(n_x: int = 512, T_in: int = 10, T_out: int = 1,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load KS data from PINNacle reference (512 spatial × 251 temporal).

    PINNacle KS: x∈[0,2π], t∈[0,1], alpha=100/16, beta=100/16^2, gamma=100/16^4
    Format: (x, t, u) columns.

    Returns sliding-window samples from the full trajectory.
    """
    ref_path = _PINNACLE_REF["KuramotoSivashinsky"]
    dat = _load_dat(ref_path)           # (~128512, 3) — (x, t, u)

    x_vals = np.unique(dat[:, 0])
    t_vals = np.unique(dat[:, 1])
    nx, nt = len(x_vals), len(t_vals)
    logger.info("KS grid: %d spatial × %d temporal", nx, nt)

    # Sort and reshape to (n_t, n_x)
    idx = np.lexsort((dat[:, 0], dat[:, 1]))
    dat_sorted = dat[idx]
    sol = dat_sorted[:, 2].reshape(nt, nx).astype(np.float32)

    # Interpolate if target n_x differs
    if n_x != nx:
        x_fine = np.linspace(_KS_X[0], _KS_X[1], n_x)
        x_coarse = np.linspace(_KS_X[0], _KS_X[1], nx)
        sol_new = np.zeros((nt, n_x), dtype=np.float32)
        for ti in range(nt):
            sol_new[ti] = np.interp(x_fine, x_coarse, sol[ti])
        sol = sol_new

    x_list, y_list = [], []
    for t_start in range(0, nt - T_in - T_out + 1):
        inp = sol[t_start:t_start + T_in].T           # (n_x, T_in)
        out = sol[t_start + T_in:t_start + T_in + T_out].T  # (n_x, T_out)
        x_list.append(inp)
        y_list.append(out)

    return (torch.tensor(np.array(x_list), dtype=torch.float32),
            torch.tensor(np.array(y_list), dtype=torch.float32))


def _interp_scattered_to_grid(x_pts, y_pts, values, x_grid, y_grid,
                               method: str = "linear") -> np.ndarray:
    """Interpolate scattered 2D points onto a regular grid."""
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    pts = np.stack([x_pts, y_pts], axis=1)
    return griddata(pts, values, (X, Y), method=method,
                    fill_value=float(np.nanmean(values))).astype(np.float32)


def _build_burgers2d_data(n_xy: int = 32, T_in: int = 5, T_out: int = 1,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Burgers2D data from PINNacle reference.

    ref/burgers2d_0.dat format (from PINNacle Burgers2D):
        cols: [x, y, u@t0, v@t0, u@t1, v@t1, ..., u@t10, v@t10]
        12657 irregular nodes, 11 timesteps, 2 velocity components.
        Domain: x,y in [0,4], t in [0,1].

    We interpolate to a regular (n_xy × n_xy) grid, take u only.
    """
    ref_path = _PINNACLE_REF["Burgers2D"]
    dat = _load_dat(ref_path)           # (12657, 24)
    x_pts, y_pts = dat[:, 0], dat[:, 1]
    n_t_ref = 11  # timesteps in file (t=0 to t=1 in 10 steps)

    x_grid = np.linspace(0, _BURGERS2D_L, n_xy)
    y_grid = np.linspace(0, _BURGERS2D_L, n_xy)

    # cols: u@t0, v@t0, u@t1, v@t1, ... (22 cols after x,y)
    # Take u component only (cols 2,4,6,8,...,22 — 0-indexed even offset from col 2)
    sol = np.zeros((n_t_ref, n_xy, n_xy), dtype=np.float32)
    for ti in range(n_t_ref):
        u_col = dat[:, 2 + ti * 2]     # u component at timestep ti
        sol[ti] = _interp_scattered_to_grid(x_pts, y_pts, u_col, x_grid, y_grid)

    x_list, y_list = [], []
    for t_start in range(0, n_t_ref - T_in - T_out + 1):
        inp = sol[t_start:t_start + T_in].transpose(1, 2, 0)  # (n_xy, n_xy, T_in)
        out = sol[t_start + T_in:t_start + T_in + T_out].transpose(1, 2, 0)
        x_list.append(inp)
        y_list.append(out)

    return (torch.tensor(np.array(x_list), dtype=torch.float32),
            torch.tensor(np.array(y_list), dtype=torch.float32))


def _build_heat_data(n_xy: int = 32, T_in: int = 5, T_out: int = 1,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heat2D_ComplexGeometry data from PINNacle reference.

    ref/heat_complex.dat:
        Domain: x∈[-8,8], y∈[-12,12], t∈[0,3], 13312 nodes.
        Cols: [x, y, u@t=0, u@t=t1, ..., u@t=t30]  (31 time snapshots)

    Interpolated to rectangular (n_xy × n_xy) bounding box grid.
    Note: FNO sees rectangular domain; complex geometry handled by near-zero
    boundary values from interpolation fill.
    """
    ref_path = _PINNACLE_REF["HeatComplexGeometry"]
    dat = _load_dat(ref_path)           # (13312, 33)
    x_pts, y_pts = dat[:, 0], dat[:, 1]
    n_t_ref = dat.shape[1] - 2         # 31 timesteps

    x0, x1, y0, y1 = _HEAT_BBOX[0], _HEAT_BBOX[1], _HEAT_BBOX[2], _HEAT_BBOX[3]
    x_grid = np.linspace(x0, x1, n_xy)
    y_grid = np.linspace(y0, y1, n_xy)

    sol = np.zeros((n_t_ref, n_xy, n_xy), dtype=np.float32)
    for ti in range(n_t_ref):
        sol[ti] = _interp_scattered_to_grid(x_pts, y_pts, dat[:, 2 + ti],
                                            x_grid, y_grid)

    x_list, y_list = [], []
    for t_start in range(0, n_t_ref - T_in - T_out + 1):
        inp = sol[t_start:t_start + T_in].transpose(1, 2, 0)
        out = sol[t_start + T_in:t_start + T_in + T_out].transpose(1, 2, 0)
        x_list.append(inp)
        y_list.append(out)

    return (torch.tensor(np.array(x_list), dtype=torch.float32),
            torch.tensor(np.array(y_list), dtype=torch.float32))


def _build_ns_data(n_xy: int = 64, ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NS2D LidDriven data from PINNacle reference files.

    Uses all available ref/lid_driven_a*.dat files (a=2,4,6,8,10,16,32).
    Steady-state operator: encode 'a' as a constant input field, predict (u,v,p).

    PINNacle NS2D_LidDriven: x,y∈[0,1], 7 different Re (a) values.
    Format: (x, y, u, v, p) per file, 10201 nodes ≈ 101×101 grid.

    FNO2d input:  (n_samples, n_xy, n_xy, 1)  — a-parameter field
    FNO2d output: (n_samples, n_xy, n_xy, 3)  — (u, v, p)
    """
    paths = _PINNACLE_REF["NavierStokes2D"]
    a_vals = [2, 4, 6, 8, 10, 16, 32]
    a_max  = float(max(a_vals))

    x_grid = np.linspace(*_NS_BBOX[:2], n_xy)
    y_grid = np.linspace(*_NS_BBOX[2:], n_xy)

    available = [(a, p) for a, p in zip(a_vals, paths) if p.exists()]
    if not available:
        raise FileNotFoundError(f"No lid_driven_a*.dat files found in {REF_DIR}")

    logger.info("NS LidDriven: using %d ref files", len(available))

    x_list, y_list = [], []
    for a, path in available:
        dat = _load_dat(path)           # (10201, 5): x, y, u, v, p
        x_pts, y_pts = dat[:, 0], dat[:, 1]

        # Encode 'a' normalised as a constant field
        a_field = np.full((n_xy, n_xy, 1), a / a_max, dtype=np.float32)
        x_list.append(a_field)

        # Interpolate u,v,p onto regular grid
        uvp = np.zeros((n_xy, n_xy, 3), dtype=np.float32)
        for ci, col in enumerate([dat[:, 2], dat[:, 3], dat[:, 4]]):
            uvp[:, :, ci] = _interp_scattered_to_grid(x_pts, y_pts, col,
                                                       x_grid, y_grid)
        y_list.append(uvp)

    return (torch.tensor(np.array(x_list), dtype=torch.float32),
            torch.tensor(np.array(y_list), dtype=torch.float32))


# ---------------------------------------------------------------------------
# Dry-run: random tensors (fast pipeline check, no real data needed)
# ---------------------------------------------------------------------------

def make_dry_run_data(pde_name: str, T_in: int, T_out: int,
                      n_samples: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate tiny random data for --dry-run pipeline testing."""
    n = 16
    if pde_name in _PDE1D:
        x = torch.randn(n_samples, n, T_in)
        y = torch.randn(n_samples, n, T_out)
    elif pde_name == "NavierStokes2D":
        # Steady-state: input = 1-channel param field, output = 3-channel uvp
        x = torch.randn(n_samples, n, n, 1)
        y = torch.randn(n_samples, n, n, 3)
    else:
        x = torch.randn(n_samples, n, n, T_in)
        y = torch.randn(n_samples, n, n, T_out)
    return x, y


# ---------------------------------------------------------------------------
# Unified data loader
# ---------------------------------------------------------------------------

def load_data(pde_name: str, T_in: int, T_out: int,
              dry_run: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load data for a given PDE.

    Dry-run: tiny random tensors (pipeline check only).
    Real run: PINNacle .dat reference data + generated training trajectories.
    """
    if dry_run:
        logger.info("DRY RUN: using random tensors for %s", pde_name)
        return make_dry_run_data(pde_name, T_in, T_out)

    logger.info("Loading PINNacle reference data for %s", pde_name)
    if pde_name == "Burgers1D":
        return _build_burgers1d_data(n_x=128, n_t=50,
                                     n_train=200, T_in=T_in, T_out=T_out)
    elif pde_name == "KuramotoSivashinsky":
        return _build_ks_data(n_x=512, T_in=T_in, T_out=T_out)
    elif pde_name == "Burgers2D":
        return _build_burgers2d_data(n_xy=32, T_in=min(T_in, 5), T_out=T_out)
    elif pde_name == "HeatComplexGeometry":
        return _build_heat_data(n_xy=32, T_in=min(T_in, 10), T_out=T_out)
    elif pde_name == "NavierStokes2D":
        return _build_ns_data(n_xy=64)
    else:
        raise ValueError(f"Unknown PDE: {pde_name}")


# ---------------------------------------------------------------------------
# L2 relative error (PINNacle / PDEBench definition)
# ---------------------------------------------------------------------------

def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    L2 relative error: ||pred - target||_F / ||target||_F

    Source: PDEBench, Takamoto et al. NeurIPS 2022, https://arxiv.org/abs/2210.07182
    """
    diff = (pred - target).view(pred.shape[0], -1).norm(dim=-1)
    norm = target.view(target.shape[0], -1).norm(dim=-1).clamp(min=1e-10)
    return float((diff / norm).mean().item())


# ---------------------------------------------------------------------------
# Model builder — handles NS steady-state (1 input channel, 3 output channels)
# ---------------------------------------------------------------------------

def build_model(pde_name: str, T_in: int, T_out: int,
                modes: int, width: int, n_layers: int) -> nn.Module:
    if pde_name == "NavierStokes2D":
        # Steady-state: 1-channel param → 3-channel (u,v,p)
        return FNO2d(modes1=modes, modes2=modes, width=width,
                     T_in=1, T_out=3, n_layers=n_layers)
    elif pde_name in _PDE1D:
        return FNO1d(modes=modes, width=width, T_in=T_in, T_out=T_out,
                     n_layers=n_layers)
    else:
        return FNO2d(modes1=modes, modes2=modes, width=width,
                     T_in=T_in, T_out=T_out, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fno(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
              epochs: int, lr: float, batch_size: int, device: str) -> nn.Module:
    """Train FNO with Adam + MSE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

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


def evaluate_fno(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor,
                 device: str, batch_size: int = 64) -> float:
    """Evaluate FNO using L2 relative error."""
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
            "model":             "FNO",
            "l2rel":             f"{l2:.6f}" if not np.isnan(l2) else "nan",
            "run_time_seconds":  f"{run_time:.2f}",
            "notes":             notes,
        })
    logger.info("Saved result: %s | L2=%.4f | time=%.1fs", pde_name, l2, run_time)


# ---------------------------------------------------------------------------
# Main per-PDE runner
# ---------------------------------------------------------------------------

def run_fno_on_pde(pde_name: str, args, device: str):
    logger.info("\n%s\nFNO benchmark: %s\n%s", "="*60, pde_name, "="*60)

    T_in  = 5  if args.dry_run else 10
    T_out = 1
    if pde_name == "NavierStokes2D":
        T_in = 1   # steady-state: single param input
    elif pde_name in {"Burgers2D", "HeatComplexGeometry"} and not args.dry_run:
        T_in = 5   # fewer timesteps in dat files → smaller T_in

    modes  = 4  if args.dry_run else args.modes
    width  = 8  if args.dry_run else args.width
    epochs = 2  if args.dry_run else args.epochs

    try:
        x_data, y_data = load_data(pde_name, T_in, T_out, dry_run=args.dry_run)
        logger.info("Data shapes — x: %s, y: %s", tuple(x_data.shape), tuple(y_data.shape))

        n = len(x_data)
        split = max(1, int(0.8 * n))
        x_train, y_train = x_data[:split], y_data[:split]
        x_test,  y_test  = x_data[split:], y_data[split:]

        # Ensure test set is non-empty
        if len(x_test) == 0:
            x_test, y_test = x_data[-1:], y_data[-1:]

        model = build_model(pde_name, T_in, T_out, modes, width, args.n_layers)
        logger.info("FNO parameters: %d", sum(p.numel() for p in model.parameters()))

        ckpt_path = args.checkpoint
        t_start   = time.time()

        if ckpt_path and os.path.exists(ckpt_path):
            logger.info("Loading checkpoint: %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(
                state.get("model_state_dict", state))
            model = model.to(device)
        else:
            if ckpt_path:
                logger.warning("Checkpoint not found: %s — training.", ckpt_path)
            logger.info("Training FNO for %d epochs...", epochs)
            model = train_fno(model, x_train, y_train,
                              epochs=epochs, lr=args.lr,
                              batch_size=args.batch_size, device=device)
            ckpt_save = RESULTS_DIR / f"fno_{pde_name}_ckpt.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_save)
            logger.info("Checkpoint saved to %s", ckpt_save)

        l2       = evaluate_fno(model, x_test, y_test, device=device,
                                batch_size=args.batch_size)
        run_time = time.time() - t_start

        logger.info("FNO on %s: L2=%.4f (%.1fs)", pde_name, l2, run_time)
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
    parser = argparse.ArgumentParser(description="FNO PDE Benchmark (PINNacle data)")
    parser.add_argument("--pde", type=str, default=None,
                        help="Single PDE to run. Default: run all.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="Random tensors only — fast pipeline check.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--modes",  type=int, default=16)
    parser.add_argument("--width",  type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    pdes = [args.pde] if args.pde else _ALL_PDES
    logger.info("FNO benchmark PDEs: %s", pdes)
    logger.info("Results → %s", RESULTS_CSV)

    for pde in pdes:
        run_fno_on_pde(pde, args, device)

    logger.info("All FNO benchmarks complete.")


if __name__ == "__main__":
    main()
