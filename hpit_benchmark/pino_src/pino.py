# Source: PINO — Physics-Informed Neural Operator
# Paper:  Li et al., "Physics-Informed Neural Operator for Learning Partial
#         Differential Equations", 2021.
#         https://arxiv.org/abs/2111.08907
# Code reference: https://github.com/neuraloperator/PINO
#
# PINO = FNO backbone + PDE residual physics loss terms.
#
# Key idea (from the paper, Section 3):
#   Total loss = L_data + λ_f * L_f
#   where:
#     L_data = ||u_pred - u_true||^2  (supervised data loss)
#     L_f    = ||F(u_pred)||^2        (PDE residual physics loss)
#
# The physics loss is computed on the FNO's predicted output field by
# automatic differentiation (autograd), enforcing the PDE equation at
# collocation points within the predicted trajectory.
#
# Architecture:
#   - FNO1d / FNO2d backbone (identical to fno_src/fno.py)
#   - Per-PDE physics residual loss computed at prediction time
#   - Configurable physics weight λ_f
#
# PINO1d and PINO2d expose the same tensor interface as FNO1d / FNO2d,
# differing only during training (they accept a physics_loss_fn argument).
# At inference time, they are identical to FNO.

import importlib.util
import sys
import math
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Re-use FNO backbone from fno_src (no duplication)
# ---------------------------------------------------------------------------

_FNO_SRC = Path(__file__).resolve().parents[1] / "fno_src" / "fno.py"
_spec = importlib.util.spec_from_file_location("fno_src.fno_pino", str(_FNO_SRC))
_fno_mod = importlib.util.module_from_spec(_spec)
sys.modules["fno_src.fno_pino"] = _fno_mod
_spec.loader.exec_module(_fno_mod)

FNO1d = _fno_mod.FNO1d
FNO2d = _fno_mod.FNO2d


# ---------------------------------------------------------------------------
# Physics residual functions — PINNacle-compatible PDE definitions
# ---------------------------------------------------------------------------
# Each function takes:
#   u_pred : (B, n_x, T_out)  or  (B, n_x, n_y, T_out)
#   x_grid : spatial coordinate tensor
#   dx      : spatial grid spacing
#   dt      : temporal grid spacing
# Returns scalar physics loss (mean squared PDE residual).

def _physics_loss_burgers1d(u_pred: torch.Tensor,
                             dx: float,
                             nu: float = 0.01 / math.pi) -> torch.Tensor:
    """
    1D Burgers spatial residual proxy:
        ||u * u_x - nu * u_xx||^2

    Full PDE is u_t + u*u_x - nu*u_xx = 0.  For single-step prediction
    (T_out=1) the time derivative is not available, so we penalize the
    spatial operators only.  This is the standard PINO approach for
    single-step problems.

    Boundary handling: zero-pad (Dirichlet u=0 at x=±1) via F.pad,
    which avoids the wrap-around artifacts of torch.roll.

    u_pred: (B, n_x, T_out)
    """
    u = u_pred                                           # (B, n_x, T_out)
    # Pad spatial dim with zeros on both sides (Dirichlet BCs)
    # F.pad pads last dims first; dims are (T_out_left, T_out_right, n_x_left, n_x_right)
    u_pad = F.pad(u, (0, 0, 1, 1), mode='constant', value=0.0)  # (B, n_x+2, T_out)
    u_xm = u_pad[:, :-2, :]   # u at x - dx
    u_xp = u_pad[:,  2:, :]   # u at x + dx

    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_xx = (u_xp - 2.0 * u + u_xm) / (dx ** 2)

    residual = u * u_x - nu * u_xx
    return (residual ** 2).mean()


def _physics_loss_ks1d(u_pred: torch.Tensor, dx: float) -> torch.Tensor:
    """
    KS PDE spatial residual proxy:
        ||u_xx + u_xxxx||^2

    PINNacle KS domain is [0, 2π] with periodic BCs, so torch.roll is
    correct here (no wrap-around error for periodic domains).

    u_pred: (B, n_x, T_out)
    """
    u = u_pred                                  # (B, n_x, T_out)
    # Periodic BCs — torch.roll is correct for periodic domains
    u_xp  = torch.roll(u, -1, dims=1)
    u_xm  = torch.roll(u,  1, dims=1)
    u_xpp = torch.roll(u, -2, dims=1)
    u_xmm = torch.roll(u,  2, dims=1)

    u_xx   = (u_xp - 2*u + u_xm) / (dx**2)
    u_xxxx = (u_xpp - 4*u_xp + 6*u - 4*u_xm + u_xmm) / (dx**4)

    residual = u_xx + u_xxxx
    return (residual ** 2).mean()


def _physics_loss_burgers2d(u_pred: torch.Tensor, dx: float,
                              nu: float = 0.001) -> torch.Tensor:
    """
    2D Burgers spatial residual proxy:
        ||u*u_x + u*u_y - nu*(u_xx + u_yy)||^2

    Zero-pad in x and y (Dirichlet BCs) via F.pad to avoid torch.roll
    wrap-around contamination at non-periodic boundaries.

    u_pred: (B, n_x, n_y, T_out)
    """
    u = u_pred                                  # (B, n_x, n_y, T_out)
    # Pad x-dimension (dim 1): pad=(T_left, T_right, y_left, y_right, x_left, x_right)
    # F.pad pads from last dim inward, so for (B,nx,ny,T) we need:
    # pad=(0,0, 0,0, 1,1) for x, then (0,0, 1,1, 0,0) for y
    u_xpad = F.pad(u, (0, 0, 0, 0, 1, 1), mode='constant', value=0.0)
    u_xm = u_xpad[:, :-2, :, :]
    u_xp = u_xpad[:,  2:, :, :]

    u_ypad = F.pad(u, (0, 0, 1, 1, 0, 0), mode='constant', value=0.0)
    u_ym = u_ypad[:, :, :-2, :]
    u_yp = u_ypad[:, :,  2:, :]

    u_x  = (u_xp - u_xm) / (2*dx)
    u_y  = (u_yp - u_ym) / (2*dx)
    u_xx = (u_xp - 2*u + u_xm) / (dx**2)
    u_yy = (u_yp - 2*u + u_ym) / (dx**2)

    residual = u*u_x + u*u_y - nu*(u_xx + u_yy)
    return (residual ** 2).mean()


def _physics_loss_heat2d(u_pred: torch.Tensor, dx: float,
                          alpha: float = 0.1) -> torch.Tensor:
    """
    Heat 2D spatial residual proxy:
        ||u_xx + u_yy||^2   (Laplacian of predicted field)

    Zero-pad in x and y via F.pad (Robin/Dirichlet BCs on complex geometry).

    u_pred: (B, n_x, n_y, T_out)
    """
    u = u_pred
    u_xpad = F.pad(u, (0, 0, 0, 0, 1, 1), mode='constant', value=0.0)
    u_xm = u_xpad[:, :-2, :, :]
    u_xp = u_xpad[:,  2:, :, :]

    u_ypad = F.pad(u, (0, 0, 1, 1, 0, 0), mode='constant', value=0.0)
    u_ym = u_ypad[:, :, :-2, :]
    u_yp = u_ypad[:, :,  2:, :]

    laplacian = (u_xp - 2*u + u_xm) / (dx**2) + (u_yp - 2*u + u_ym) / (dx**2)
    return (laplacian ** 2).mean()


def _physics_loss_ns2d(uvp_pred: torch.Tensor, dx: float,
                        nu: float = 0.01) -> torch.Tensor:
    """
    NS2D steady-state continuity residual:
        ||u_x + v_y||^2

    Zero-pad in x and y via F.pad (no-slip wall BCs for lid-driven cavity).

    uvp_pred: (B, n_x, n_y, 3) — (u, v, p) fields
    """
    u = uvp_pred[..., 0]   # (B, n_x, n_y)
    v = uvp_pred[..., 1]

    # Pad u in x-direction for u_x
    u_xpad = F.pad(u, (0, 0, 1, 1), mode='constant', value=0.0)  # pad dim 1
    u_xm = u_xpad[:, :-2, :]
    u_xp = u_xpad[:,  2:, :]

    # Pad v in y-direction for v_y
    v_ypad = F.pad(v, (1, 1, 0, 0), mode='constant', value=0.0)  # pad dim 2
    v_ym = v_ypad[:, :, :-2]
    v_yp = v_ypad[:, :,  2:]

    u_x = (u_xp - u_xm) / (2*dx)
    v_y = (v_yp - v_ym) / (2*dx)
    continuity = u_x + v_y
    return (continuity ** 2).mean()


# Map PDE name → physics loss function (legacy inline versions, kept for reference)
_PHYSICS_LOSS_FNS_LEGACY = {
    "Burgers1D":           lambda pred, dx: _physics_loss_burgers1d(pred, dx),
    "KuramotoSivashinsky": lambda pred, dx: _physics_loss_ks1d(pred, dx),
    "Burgers2D":           lambda pred, dx: _physics_loss_burgers2d(pred, dx),
    "HeatComplexGeometry": lambda pred, dx: _physics_loss_heat2d(pred, dx),
    "NavierStokes2D":      lambda pred, dx: _physics_loss_ns2d(pred, dx),
}

# Approximate grid spacings per PDE (from PINNacle parameters)
GRID_DX = {
    "Burgers1D":            2.0 / 128,          # x in [-1,1], n_x=128
    "KuramotoSivashinsky":  2*math.pi / 512,    # x in [0,2pi], n_x=512
    "Burgers2D":            4.0 / 32,           # x in [0,4], n_xy=32
    "HeatComplexGeometry":  16.0 / 32,          # x in [-8,8], n_xy=32
    "NavierStokes2D":       1.0 / 64,           # x in [0,1], n_xy=64
}

# --- Shared physics constraints (same as HPIT uses for comparability) ---
# Import from pde_physics_constraints.py in the benchmark directory.
_PHYS_CONSTRAINTS_PATH = Path(__file__).resolve().parents[1] / "pde_physics_constraints.py"
_phys_spec = importlib.util.spec_from_file_location("pde_physics_constraints", str(_PHYS_CONSTRAINTS_PATH))
_phys_mod = importlib.util.module_from_spec(_phys_spec)
sys.modules["pde_physics_constraints"] = _phys_mod
_phys_spec.loader.exec_module(_phys_mod)

# The shared physics loss classes: BurgersPhysicsLoss, HeatPhysicsLoss,
# NSPhysicsLoss, KSPhysicsLoss — used via get_physics_loss().
_get_physics_loss = _phys_mod.get_physics_loss

# Build PHYSICS_LOSS_FNS using the shared constraints
def _build_shared_phys_fn(pde_name):
    """Create a lambda(pred, dx) → scalar loss using shared constraints."""
    loss_mod, coords, params = _get_physics_loss(pde_name)
    if loss_mod is None:
        return None
    def _fn(pred, dx, _mod=loss_mod, _c=coords, _p=params):
        return _mod(pred, _c, _p)
    return _fn

PHYSICS_LOSS_FNS = {
    pde: _build_shared_phys_fn(pde)
    for pde in ["Burgers1D", "KuramotoSivashinsky", "Burgers2D",
                "HeatComplexGeometry", "NavierStokes2D"]
}
# Remove any None entries (shouldn't happen, all 5 are registered)
PHYSICS_LOSS_FNS = {k: v for k, v in PHYSICS_LOSS_FNS.items() if v is not None}


# ---------------------------------------------------------------------------
# PINO wrapper classes
# ---------------------------------------------------------------------------

class PINO1d(FNO1d):
    """
    PINO for 1D PDEs: FNO1d backbone + physics residual loss.

    Tensor interface: identical to FNO1d.
    Physics loss is applied ONLY during training (via compute_physics_loss).

    Source: Li et al. 2021, https://arxiv.org/abs/2111.08907
    """

    def __init__(self, pde_name: str, physics_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.pde_name       = pde_name
        self.physics_weight = physics_weight
        self._phys_fn       = PHYSICS_LOSS_FNS.get(pde_name)
        self._dx            = GRID_DX.get(pde_name, 0.02)

    def compute_physics_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual physics loss on the predicted field."""
        if self._phys_fn is None:
            return torch.tensor(0.0, device=u_pred.device)
        return self.physics_weight * self._phys_fn(u_pred, self._dx)


class PINO2d(FNO2d):
    """
    PINO for 2D PDEs: FNO2d backbone + physics residual loss.

    Source: Li et al. 2021, https://arxiv.org/abs/2111.08907
    """

    def __init__(self, pde_name: str, physics_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.pde_name       = pde_name
        self.physics_weight = physics_weight
        self._phys_fn       = PHYSICS_LOSS_FNS.get(pde_name)
        self._dx            = GRID_DX.get(pde_name, 0.02)

    def compute_physics_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        if self._phys_fn is None:
            return torch.tensor(0.0, device=u_pred.device)
        return self.physics_weight * self._phys_fn(u_pred, self._dx)
