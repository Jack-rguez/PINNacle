"""
PDE-specific physics layers and residual loss computation for HPIT.

Governing equations verified from PINNacle source (Hao et al., 2024):
  src/pde/burgers.py line 10,25:
      Burgers1D:  u_t + u·u_x - ν·u_xx = 0,   ν = 0.01/π,  x∈[-1,1], t∈[0,1]
  src/pde/heat.py line 147-152:
      Heat2D:     u_t - u_xx - u_yy   = 0,   (α=1 explicit), x∈[-8,8], y∈[-12,12], t∈[0,3]

Physics residual loss follows the PINN formulation of:
    Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019).
    Physics-informed neural networks: A deep learning framework for solving
    forward and inverse problems involving nonlinear partial differential equations.
    Journal of Computational Physics, 378, 686-707.
    https://doi.org/10.1016/j.jcp.2018.10.045

Architecture of per-PDE physics layers follows the physics-encoded operator approach:
    Wang, S., Wang, H., Perdikaris, P. (2022).
    Improved architectures and training algorithms for deep operator networks
    solving differential equations.
    Journal of Scientific Computing, 92(2), 35.
    https://doi.org/10.1007/s10915-022-01881-0
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Physical constants (from PINNacle src/pde/) ────────────────────────────────

# Burgers 1D: src/pde/burgers.py, line 10
BURGERS1D_NU = 0.01 / math.pi      # kinematic viscosity ≈ 3.183e-3

# Burgers 1D domain: src/pde/burgers.py, line 10
BURGERS1D_X0, BURGERS1D_X1 = -1.0, 1.0    # x ∈ [-1, 1]
BURGERS1D_T0, BURGERS1D_T1 =  0.0, 1.0    # t ∈ [0, 1]

# Heat 2D Complex: src/pde/heat.py, line 152: return [u_t - u_xx - u_yy]
HEAT2D_ALPHA = 1.0          # thermal diffusivity (no scaling coefficient in PDE)

# Heat 2D domain: src/pde/heat.py, line 124
HEAT2D_X0, HEAT2D_X1 = -8.0,   8.0    # x ∈ [-8, 8],  Lx = 16
HEAT2D_Y0, HEAT2D_Y1 = -12.0, 12.0    # y ∈ [-12, 12], Ly = 24
HEAT2D_T0, HEAT2D_T1 =   0.0,  3.0    # t ∈ [0, 3],   Lt = 3


# ── Helper: scaled PDE parameters in pinnacle_to_hpit normalized coords ────────
# pinnacle_to_hpit maps coordinates to [0,1] via linspace.
# Physical ↔ normalized: x_phys = x0 + (x1-x0)*x̂,  similarly for y, t.
# Derivative scaling:  ∂/∂x_phys = (1/Lx)·∂/∂x̂,  ∂²/∂x²_phys = (1/Lx²)·∂²/∂x̂²
#
# Burgers1D in normalized (x̂, t̂):
#   u_t̂/Lt' + u·(1/Lx)·u_x̂ - ν·(1/Lx²)·u_x̂x̂ = 0
#   where Lt' = physical time span of one input window ≈ (T_in-1)/(n_t-1) * Lt
#   For simplicity we absorb Lt' into an effective ν, using Lt'≈1 (unit window).
#   Result: u_t̂ + (Lt'/Lx)·u·u_x̂ - ν·(Lt'/Lx²)·u_x̂x̂ = 0
#
# With Lx=2, Lt'=1: u_t̂ + 0.5·u·u_x̂ - (ν/4)·u_x̂x̂ = 0
BURGERS1D_ADV_SCALE = 1.0 / (BURGERS1D_X1 - BURGERS1D_X0)           # 0.5
BURGERS1D_DIFF_SCALE = BURGERS1D_NU / (BURGERS1D_X1 - BURGERS1D_X0)**2  # ν/4

# Heat2D in normalized coords (x̂, ŷ, t̂):
#   u_t̂/Lt - (1/Lx²)·u_x̂x̂ - (1/Ly²)·u_ŷŷ = 0
#   Multiply by Lt: u_t̂ - (Lt/Lx²)·u_x̂x̂ - (Lt/Ly²)·u_ŷŷ = 0
_Lt = HEAT2D_T1 - HEAT2D_T0    # 3
_Lx = HEAT2D_X1 - HEAT2D_X0   # 16
_Ly = HEAT2D_Y1 - HEAT2D_Y0   # 24
HEAT2D_DIFF_X_SCALE = HEAT2D_ALPHA * _Lt / _Lx**2   # 3/256  ≈ 0.01172
HEAT2D_DIFF_Y_SCALE = HEAT2D_ALPHA * _Lt / _Ly**2   # 3/576  ≈ 0.00521


# ── PDE-specific physics layers ────────────────────────────────────────────────

class BurgersPhysicsLayer(nn.Module):
    """
    Physics-motivated feature layer encoding the structure of the Burgers equation.

    PDE: u_t + u·u_x - ν·u_xx = 0  (ν = 0.01/π)

    Three sub-networks encode the three terms of the Burgers equation:
      1. temporal_rate_net  — proxy for ∂u/∂t (temporal evolution)
      2. advection_net      — proxy for u·∂u/∂x (nonlinear convective term)
      3. diffusion_net      — proxy for ν·∂²u/∂x² (viscous dissipation)

    Spatial derivatives (u_x, u_xx) cannot be computed from a single-point
    temporal sequence alone; these sub-networks learn to approximate the
    relevant physics from the sequence embedding. Exact spatial residuals
    are enforced via the physics_loss in train_hpit (see compute_burgers1d_residual_loss).

    Input: high-dimensional sequence embedding (output of attention layers)
    Output: dict with 'physics_integrated' scalar correction per sample
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.nu = BURGERS1D_NU

        # Temporal rate of change: learns ∂u/∂t structure from embedding
        self.temporal_rate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Nonlinear advection: captures u·∂u/∂x — quadratic self-interaction
        # Uses a gated structure to model the u·(du/dx) product: a linear "u"
        # branch and a linear "du/dx" branch, fused multiplicatively.
        self.advection_gate   = nn.Linear(input_dim, hidden_dim)
        self.advection_signal = nn.Linear(input_dim, hidden_dim)
        self.advection_out    = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Viscous diffusion: captures ν·∂²u/∂x² — second-order smoothing
        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Burgers residual fusion: integrates all three physics terms
        # dim 3 = [u_t proxy, advection proxy, diffusion proxy]
        self.physics_fusion = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (N, input_dim) flattened sequence embeddings
        Returns:
            dict with 'physics_integrated': (N, 1)
        """
        temporal_rate = self.temporal_rate_net(x)

        # Gated product: gate * signal models u·(∂u/∂x) structure
        gate   = torch.sigmoid(self.advection_gate(x))
        signal = self.advection_signal(x)
        advection = self.advection_out(gate * signal)

        diffusion = self.diffusion_net(x)

        # Combine: Burgers residual proxy  u_t + u·u_x - ν·u_xx
        physics_components = torch.cat([temporal_rate, advection, diffusion], dim=-1)
        physics_integrated = self.physics_fusion(physics_components)

        return {
            "physics_integrated": physics_integrated,
            "temporal_rate":      temporal_rate,
            "advection":          advection,
            "diffusion":          diffusion,
        }


class HeatPhysicsLayer(nn.Module):
    """
    Physics-motivated feature layer encoding the structure of the 2D heat equation.

    PDE: u_t - u_xx - u_yy = 0  (α = 1, PINNacle Heat2D_ComplexGeometry)

    Two sub-networks encode the two terms of the heat equation:
      1. temporal_rate_net  — proxy for ∂u/∂t (temporal evolution)
      2. laplacian_net      — proxy for ∇²u = u_xx + u_yy (diffusive spreading)

    The heat equation is LINEAR (no nonlinear advection term), distinguishing
    it architecturally from BurgersPhysicsLayer which uses a gated-product
    structure for the nonlinear advection term.

    Input: high-dimensional sequence embedding (output of attention layers)
    Output: dict with 'physics_integrated' scalar correction per sample
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.alpha = HEAT2D_ALPHA

        # Temporal rate of change: ∂u/∂t
        self.temporal_rate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Laplacian proxy: ∇²u — captures diffusive spreading / spatial smoothing
        # Uses a PURELY LINEAR path followed by a small MLP to respect the
        # linearity of the heat equation (no multiplicative gating).
        self.laplacian_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Heat residual fusion: u_t - α·∇²u
        self.physics_fusion = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (N, input_dim) flattened sequence embeddings
        Returns:
            dict with 'physics_integrated': (N, 1)
        """
        temporal_rate = self.temporal_rate_net(x)
        laplacian     = self.laplacian_net(x)

        # Combine: heat residual proxy  u_t - α·∇²u
        physics_components = torch.cat([temporal_rate, laplacian], dim=-1)
        physics_integrated = self.physics_fusion(physics_components)

        return {
            "physics_integrated": physics_integrated,
            "temporal_rate":      temporal_rate,
            "laplacian":          laplacian,
        }


# ── Physics residual loss on reconstructed field ───────────────────────────────

def compute_burgers1d_residual_loss(
    model,
    x_windows: torch.Tensor,   # (B, n_x, T_in) — input u fields on regular grid
    y_windows: torch.Tensor,   # (B, n_x, 1)    — target u at next time step
    device: str,
    max_windows: int = 20,
    enable_grad: bool = False,
) -> torch.Tensor:
    """
    Compute the Burgers 1D physics residual on model predictions.

    For each spatial window b, runs the model on ALL n_x spatial sequences,
    reconstructs the predicted field, and evaluates the PDE residual:

        u_t + u·u_x - ν·u_xx = 0   (in normalized coordinates)

    Temporal derivative at the LAST input step uses central FD with the
    model prediction as the right neighbour:
        u_t[j, T-1] ≈ (u_pred[j] - u[j, T-3]) / (2·Δt̂)

    Spatial derivatives use 2nd-order central FD on the predicted field:
        u_x[j]  ≈ (u_pred[j+1] - u_pred[j-1]) / (2·Δx̂)
        u_xx[j] ≈ (u_pred[j+1] - 2·u_pred[j] + u_pred[j-1]) / Δx̂²

    Residual is evaluated only at INTERIOR spatial points (j ∈ [1, n_x-2]).

    References:
        Raissi et al. (2019) JCP 378, 686-707. — PINN residual formulation.
        PINNacle src/pde/burgers.py — exact equation and ν value.

    Args:
        model:       HPITModel in train mode
        x_windows:   (B, n_x, T_in) input u fields (NOT the HPIT format)
        y_windows:   (B, n_x, 1) target fields
        device:      'cuda' or 'cpu'
        max_windows: cap number of windows used per call (for speed)
    Returns:
        Scalar physics residual loss (mean squared residual at interior points)
    """
    B, n_x, T_in = x_windows.shape
    B = min(B, max_windows)

    # Grid spacings in normalized [0, 1] coordinates
    dx = 1.0 / (n_x - 1)
    dt = 1.0 / (T_in - 1)

    residuals = []

    for b in range(B):
        # Build HPIT-format sequences for all n_x spatial points of window b
        x_coords = torch.linspace(0.0, 1.0, n_x, device=device)
        t_coords = torch.linspace(0.0, 1.0, T_in, device=device)

        u_vals = x_windows[b].to(device)   # (n_x, T_in)

        # Feature matrix: (n_x, T_in, 3) = [x_coord, t_coord, u_value]
        xc = x_coords.unsqueeze(1).expand(n_x, T_in)  # (n_x, T_in)
        tc = t_coords.unsqueeze(0).expand(n_x, T_in)  # (n_x, T_in)
        seq = torch.stack([xc, tc, u_vals], dim=-1)    # (n_x, T_in, 3)

        ctx = torch.no_grad() if not enable_grad else torch.enable_grad()
        with ctx:
            out = model(seq)
            u_pred = out.predictions.squeeze(-1)   # (n_x,)

        # Central FD temporal derivative at step T_in-1 (last input step):
        #   u_t[j] = (u_pred[j] - u[j, T_in-3]) / (2·Δt̂)
        # Note: T_in-3 because central FD at T_in-1 uses T_in-2 as left point:
        #   left=T_in-2, center=T_in-1, right=T_in=pred
        # u[j, T_in-2] is at index T_in-2 in the input sequence (0-indexed).
        u_left = u_vals[:, T_in - 2]    # u at step T_in-2, shape (n_x,)
        u_t = (u_pred - u_left) / (2.0 * dt)     # (n_x,)

        # 2nd-order central FD spatial derivatives on the PREDICTED field
        u_pred_int = u_pred[1:-1]               # interior: (n_x-2,)
        u_xp = u_pred[2:]                       # j+1
        u_xm = u_pred[:-2]                      # j-1
        u_x   = (u_xp - u_xm)            / (2.0 * dx)    # (n_x-2,)
        u_xx  = (u_xp - 2*u_pred_int + u_xm) / (dx**2)   # (n_x-2,)
        u_t_int = u_t[1:-1]                     # (n_x-2,)

        # Burgers residual in normalized coords:
        #   u_t̂ + (Lt/Lx)·u·u_x̂ - ν·(Lt/Lx²)·u_x̂x̂ = 0
        # With Lx=2, Lt=1 (unit input window approximation):
        residual = (u_t_int
                    + BURGERS1D_ADV_SCALE  * u_pred_int * u_x
                    - BURGERS1D_DIFF_SCALE * u_xx)

        residuals.append(residual.pow(2).mean())

    return torch.stack(residuals).mean()


def compute_heat2d_residual_loss(
    model,
    x_windows: torch.Tensor,   # (B, n_x, n_y, T_in) — input u fields
    y_windows: torch.Tensor,   # (B, n_x, n_y, 1)    — target
    device: str,
    max_windows: int = 10,
    enable_grad: bool = False,
) -> torch.Tensor:
    """
    Compute the Heat 2D physics residual on model predictions.

    PDE: u_t - u_xx - u_yy = 0  (α=1, PINNacle Heat2D_ComplexGeometry)

    Same approach as compute_burgers1d_residual_loss but for 2D spatial domain.
    Temporal derivative uses central FD with model prediction as right neighbour.
    Spatial derivatives use 2nd-order central FD on the predicted 2D field.

    Residual evaluated at interior points (j,k) ∈ [1,n_x-2] × [1,n_y-2].

    References:
        Raissi et al. (2019) JCP 378, 686-707.
        PINNacle src/pde/heat.py — exact equation (α=1, no coefficient).

    Args:
        model:       HPITModel in train mode
        x_windows:   (B, n_x, n_y, T_in) input u fields
        y_windows:   (B, n_x, n_y, 1) target fields
        device:      'cuda' or 'cpu'
        max_windows: cap number of windows used per call
    Returns:
        Scalar physics residual loss
    """
    B, n_x, n_y, T_in = x_windows.shape
    B = min(B, max_windows)

    dx = 1.0 / (n_x - 1)
    dy = 1.0 / (n_y - 1)
    dt = 1.0 / (T_in - 1)

    residuals = []

    for b in range(B):
        u_vals = x_windows[b].to(device)   # (n_x, n_y, T_in)
        N = n_x * n_y

        xs = torch.linspace(0.0, 1.0, n_x, device=device)
        ys = torch.linspace(0.0, 1.0, n_y, device=device)
        ts = torch.linspace(0.0, 1.0, T_in, device=device)

        XX, YY = torch.meshgrid(xs, ys, indexing='ij')   # (n_x, n_y)
        xc = XX.unsqueeze(-1).expand(n_x, n_y, T_in)
        yc = YY.unsqueeze(-1).expand(n_x, n_y, T_in)
        tc = ts.unsqueeze(0).unsqueeze(0).expand(n_x, n_y, T_in)

        # Feature: (N, T_in, 4) = [x, y, t, u]
        seq = torch.stack([xc, yc, tc, u_vals], dim=-1)  # (n_x, n_y, T_in, 4)
        seq = seq.view(N, T_in, 4)

        ctx = torch.no_grad() if not enable_grad else torch.enable_grad()
        with ctx:
            out = model(seq)
            u_pred = out.predictions.squeeze(-1).view(n_x, n_y)  # (n_x, n_y)

        # Temporal derivative: central FD at last input step
        u_left = u_vals[:, :, T_in - 2]              # (n_x, n_y)
        u_t = (u_pred - u_left) / (2.0 * dt)

        # Spatial FD on predicted field — interior points only
        u_xx = (u_pred[2:, 1:-1] - 2*u_pred[1:-1, 1:-1] + u_pred[:-2, 1:-1]) / (dx**2)
        u_yy = (u_pred[1:-1, 2:] - 2*u_pred[1:-1, 1:-1] + u_pred[1:-1, :-2]) / (dy**2)
        u_t_int = u_t[1:-1, 1:-1]

        # Heat residual in normalized coords:
        #   u_t̂ - (Lt/Lx²)·u_x̂x̂ - (Lt/Ly²)·u_ŷŷ = 0
        residual = (u_t_int
                    - HEAT2D_DIFF_X_SCALE * u_xx
                    - HEAT2D_DIFF_Y_SCALE * u_yy)

        residuals.append(residual.pow(2).mean())

    return torch.stack(residuals).mean()


# ── Registry: maps pde_name → (PhysicsLayerClass, residual_loss_fn) ─────────────

PDE_PHYSICS_REGISTRY = {
    "Burgers1D": {
        "layer_class": BurgersPhysicsLayer,
        "residual_fn": compute_burgers1d_residual_loss,
        "description": "Burgers 1D: u_t + u*u_x - nu*u_xx = 0 (nu=0.01/pi)",
    },
    "HeatComplexGeometry": {
        "layer_class": HeatPhysicsLayer,
        "residual_fn": compute_heat2d_residual_loss,
        "description": "Heat 2D: u_t - u_xx - u_yy = 0 (alpha=1)",
    },
}
