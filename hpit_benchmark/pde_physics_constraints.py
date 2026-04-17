"""
pde_physics_constraints.py — Unified PDE residual loss for all benchmark PDEs.

This module contains EXACTLY ONE physics loss class per PDE family, each with a
uniform forward(u_pred, coords, pde_params) interface. Both HPIT and PINO use
these same classes for their physics loss terms. This is the critical
comparability requirement for the benchmark.

Does NOT import from or reference physics_constraints.py, losses.py, or any
SWE-domain code.

Governing equations are verified from PINNacle source (src/pde/):
  - Burgers1D:  src/pde/burgers.py line 10,25    — u_t + u*u_x - nu*u_xx = 0, nu=0.01/pi
  - Burgers2D:  src/pde/burgers.py line 55,66-80  — 2D system with nu=0.001
  - Heat2D:     src/pde/heat.py line 122,147-152  — u_t - u_xx - u_yy = 0 (alpha=1)
  - KS:         src/pde/chaotic.py line 63,77-83  — u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxx = 0
  - NS2D:       src/pde/ns.py line 130,141-161    — momentum + continuity (steady, nu=1/100)

References:
  [1] PINNacle: Hao et al., NeurIPS 2023. arXiv:2306.08827.
  [2] PDEBench: Takamoto et al., NeurIPS 2022. Table 1. arXiv:2210.07182.
  [3] PINO: Li et al., 2021. Section 3, Appendix B. arXiv:2111.03794.
  [4] Raissi et al., JCP 378, 686-707 (2019). — PINN residual formulation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── PDE parameters from PINNacle source ──────────────────────────────────────

# Burgers 1D: src/pde/burgers.py line 10
BURGERS1D_NU = 0.01 / math.pi   # ~3.183e-3

# Burgers 2D: src/pde/burgers.py line 55
BURGERS2D_NU = 0.001

# Heat 2D: src/pde/heat.py line 152: return [u_t - u_xx - u_yy]
# alpha = 1 (no explicit coefficient)

# KS: src/pde/chaotic.py line 65
KS_ALPHA = 100.0 / 16.0        # 6.25
KS_BETA  = 100.0 / (16.0**2)   # 0.390625
KS_GAMMA = 100.0 / (16.0**4)   # ~1.526e-3

# NS2D LidDriven: src/pde/ns.py line 132
NS2D_NU = 1.0 / 100.0          # Re=100


# ── Burgers Physics Loss (1D and 2D) ────────────────────────────────────────
# PDE: u_t + u*u_x - nu*u_xx = 0  (1D, nu=0.01/pi)
# PDE: u1_t + u1*u1_x + u2*u1_y - nu*(u1_xx+u1_yy) = 0, etc. (2D, nu=0.001)
# Cite: PINNacle src/pde/burgers.py; PDEBench Table 1 [2]

class BurgersPhysicsLoss(nn.Module):
    """
    Burgers equation physics loss for 1D and 2D.

    1D PDE: u_t + u*u_x - nu*u_xx = 0,  nu = 0.01/pi
        Source: PINNacle src/pde/burgers.py line 10, 21-25
        Domain: x in [-1,1], t in [0,1], IC: u(x,0) = -sin(pi*x), BC: u=0

    2D PDE: u1_t + u1*u1_x + u2*u1_y - nu*(u1_xx + u1_yy) = 0
            u2_t + u1*u2_x + u2*u2_y - nu*(u2_xx + u2_yy) = 0
        Source: PINNacle src/pde/burgers.py line 55, 66-80
        Domain: x,y in [0,4], t in [0,1], nu=0.001, periodic BCs

    Spatial derivatives computed via 2nd-order central finite differences
    on the output grid. For single-step predictions (T_out=1), the time
    derivative is not available from the output alone; the spatial residual
    proxy (u*u_x - nu*u_xx) is used instead (standard PINO approach [3]).

    For Dirichlet BCs (1D): zero-pad via F.pad.
    For periodic BCs (2D): torch.roll.
    """

    def forward(self, u_pred, coords, pde_params):
        """
        Args:
            u_pred: predicted field tensor
                1D: (B, n_x, T_out)
                2D: (B, n_x, n_y, T_out)  — u-component only; or (B,n_x,n_y,T_out*2) for u,v
            coords: dict with 'dx' (and optionally 'dy') grid spacings
            pde_params: dict with 'nu', 'dim' (1 or 2)
        Returns:
            Scalar physics loss (mean squared PDE residual)
        """
        nu  = pde_params.get('nu', BURGERS1D_NU)
        dim = pde_params.get('dim', 1)
        dx  = coords['dx']

        if dim == 1:
            return self._residual_1d(u_pred, dx, nu)
        else:
            return self._residual_2d(u_pred, dx, nu)

    def _residual_1d(self, u_pred, dx, nu):
        """
        1D Burgers spatial residual: u*u_x - nu*u_xx
        Dirichlet BCs: zero-pad edges.
        u_pred: (B, n_x, T_out)
        """
        # Pad spatial dim with zeros (Dirichlet u=0 at x=±1)
        # F.pad pads from last dim: (T_left, T_right, x_left, x_right)
        u = u_pred
        u_pad = F.pad(u, (0, 0, 1, 1), mode='constant', value=0.0)
        u_xm = u_pad[:, :-2, :]
        u_xp = u_pad[:,  2:, :]

        # Central FD: u_x = (u[j+1]-u[j-1]) / (2*dx)
        u_x  = (u_xp - u_xm) / (2.0 * dx)
        # Central FD: u_xx = (u[j+1]-2*u[j]+u[j-1]) / dx^2
        u_xx = (u_xp - 2.0*u + u_xm) / (dx**2)

        # Spatial residual proxy: u*u_x - nu*u_xx
        residual = u * u_x - nu * u_xx
        return (residual**2).mean()

    def _residual_2d(self, u_pred, dx, nu):
        """
        2D Burgers spatial residual.  Periodic BCs: torch.roll.

        The benchmark data (fno_benchmark._build_burgers2d_data) loads only the
        u-velocity component from the .dat file, so u_pred normally has a single
        output channel (C=1). In that case the v-advection term is approximated
        as v≈u (self-advection), yielding the scalar proxy:
            u*u_x + u*u_y - nu*(u_xx + u_yy)
        This is documented in benchmark_discrepancies.md.

        If u_pred has C=2 (both u1 and u2 available), both momentum residuals
        are computed correctly:
            Residual 1: u1*u1_x + u2*u1_y - nu*(u1_xx + u1_yy)
            Residual 2: u1*u2_x + u2*u2_y - nu*(u2_xx + u2_yy)

        u_pred: (B, n_x, n_y, C) where C=1 (u only) or C=2 (u, v)
        """
        def _fd(u):
            u_xp = torch.roll(u, -1, dims=1)
            u_xm = torch.roll(u,  1, dims=1)
            u_yp = torch.roll(u, -1, dims=2)
            u_ym = torch.roll(u,  1, dims=2)
            u_x  = (u_xp - u_xm) / (2*dx)
            u_y  = (u_yp - u_ym) / (2*dx)
            u_xx = (u_xp - 2*u + u_xm) / (dx**2)
            u_yy = (u_yp - 2*u + u_ym) / (dx**2)
            return u_x, u_y, u_xx, u_yy

        if u_pred.shape[-1] == 2:
            # Both velocity components: full vector residuals
            u1 = u_pred[..., 0:1]
            u2 = u_pred[..., 1:2]
            u1_x, u1_y, u1_xx, u1_yy = _fd(u1)
            u2_x, u2_y, u2_xx, u2_yy = _fd(u2)
            res1 = u1*u1_x + u2*u1_y - nu*(u1_xx + u1_yy)
            res2 = u1*u2_x + u2*u2_y - nu*(u2_xx + u2_yy)
            return (res1**2).mean() + (res2**2).mean()
        else:
            # Single-component: scalar proxy with v≈u approximation
            u = u_pred
            u_x, u_y, u_xx, u_yy = _fd(u)
            residual = u*u_x + u*u_y - nu*(u_xx + u_yy)
            return (residual**2).mean()


# ── Heat Complex Geometry Physics Loss ───────────────────────────────────────
# PDE: u_t - u_xx - u_yy = 0  (alpha=1, no scaling coefficient)
# Source: PINNacle src/pde/heat.py line 147-152
# Domain: [-8,8]x[-12,12]x[0,3], complex geometry (circles subtracted)
# BCs: Robin on circle boundaries, Robin on rectangle boundary
# Cite: PINNacle [1]; Raissi et al. 2019 [4]

class HeatPhysicsLoss(nn.Module):
    """
    Heat 2D complex geometry physics loss.

    PDE: u_t - u_xx - u_yy = 0
        # alpha=1 per PINNacle src/pde/heat.py Heat2D_ComplexGeometry — no scaling coefficient
        Source: PINNacle src/pde/heat.py line 152: return [u_t - u_xx - u_yy]
        Domain: rectangle [-8,8]x[-12,12] with 11 big (r=1) + 6 small (r=0.4) circular holes
        IC: u(x,y,0) = 0
        BCs: Robin on circles (big: 5-u, small: 1-u), Robin on outer (0.1-u)

    Spatial residual proxy: ||u_xx + u_yy||^2  (time derivative not available)

    DOMAIN MASK — Case B:
    The benchmark data is interpolated from 13,312 scattered COMSOL nodes to a
    regular n_xy×n_xy grid (fno_benchmark._build_heat_data via scipy.griddata).
    Grid points that fall inside any of the 17 circular holes receive physically
    meaningless interpolated fill values. The Laplacian at those points must be
    excluded from the physics loss. A precomputed boolean mask identifies valid
    (non-hole) grid points using the exact circle geometry from PINNacle's source.

    Circle geometry (PINNacle src/pde/heat.py Heat2D_ComplexGeometry):
      11 big circles, r=1:  (-4,-3),(4,-3),(-4,3),(4,3),(-4,-9),(4,-9),
                             (-4,9),(4,9),(0,0),(0,6),(0,-6)
       6 small circles, r=0.4: (-3.2,-6),(-3.2,6),(3.2,-6),(3.2,6),(-3.2,0),(3.2,0)
    """

    # Circle geometry from PINNacle src/pde/heat.py Heat2D_ComplexGeometry
    _BIG_CENTERS = [(-4.0,-3.0),(4.0,-3.0),(-4.0,3.0),(4.0,3.0),
                    (-4.0,-9.0),(4.0,-9.0),(-4.0,9.0),(4.0,9.0),
                    (0.0,0.0),(0.0,6.0),(0.0,-6.0)]
    _BIG_R2 = 1.0 ** 2
    _SMALL_CENTERS = [(-3.2,-6.0),(-3.2,6.0),(3.2,-6.0),(3.2,6.0),
                      (-3.2,0.0),(3.2,0.0)]
    _SMALL_R2 = 0.4 ** 2
    _X0, _X1 = -8.0, 8.0    # domain bounds (PINNacle Heat2D_ComplexGeometry)
    _Y0, _Y1 = -12.0, 12.0

    def __init__(self):
        super().__init__()
        self._mask_cache: dict = {}   # (n_x, n_y) → CPU bool tensor

    def _domain_mask(self, n_x: int, n_y: int, device) -> torch.Tensor:
        """
        Boolean mask (n_x, n_y) — True for grid points in valid domain
        (not inside any circular hole). Cached per grid shape.
        """
        key = (n_x, n_y)
        if key not in self._mask_cache:
            xs = torch.linspace(self._X0, self._X1, n_x)
            ys = torch.linspace(self._Y0, self._Y1, n_y)
            XX, YY = torch.meshgrid(xs, ys, indexing='ij')   # (n_x, n_y)
            valid = torch.ones(n_x, n_y, dtype=torch.bool)
            for cx, cy in self._BIG_CENTERS:
                valid &= ((XX - cx)**2 + (YY - cy)**2) > self._BIG_R2
            for cx, cy in self._SMALL_CENTERS:
                valid &= ((XX - cx)**2 + (YY - cy)**2) > self._SMALL_R2
            self._mask_cache[key] = valid   # stored on CPU, moved to device in forward
        return self._mask_cache[key].to(device)

    def forward(self, u_pred, coords, pde_params):
        """
        Args:
            u_pred: (B, n_x, n_y, T_out) predicted field
            coords: dict with 'dx', 'dy'
            pde_params: dict (alpha=1 implicit, no required params)
        Returns:
            Scalar physics loss (mean over valid domain points only)
        """
        dx = coords['dx']
        dy = coords.get('dy', dx)
        n_x, n_y = u_pred.shape[1], u_pred.shape[2]

        u = u_pred
        # Zero-pad x (dim 1) for u_xx
        # alpha=1 per PINNacle src/pde/heat.py Heat2D_ComplexGeometry — no scaling coefficient
        u_xpad = F.pad(u, (0, 0, 0, 0, 1, 1), mode='constant', value=0.0)
        u_xm = u_xpad[:, :-2, :, :]
        u_xp = u_xpad[:,  2:, :, :]

        # Zero-pad y (dim 2) for u_yy
        u_ypad = F.pad(u, (0, 0, 1, 1, 0, 0), mode='constant', value=0.0)
        u_ym = u_ypad[:, :, :-2, :]
        u_yp = u_ypad[:, :,  2:, :]

        u_xx = (u_xp - 2*u + u_xm) / (dx**2)
        u_yy = (u_yp - 2*u + u_ym) / (dy**2)

        # Heat residual proxy: Laplacian(u) = u_xx + u_yy (alpha=1, no scaling)
        laplacian = u_xx + u_yy

        # Apply domain mask — Case B: exclude grid points inside circular holes.
        # Points inside holes have physically meaningless griddata fill values.
        # Valid = not inside any of the 17 circles (11 big r=1, 6 small r=0.4).
        mask = self._domain_mask(n_x, n_y, u.device)  # (n_x, n_y)
        mask4 = mask.unsqueeze(0).unsqueeze(-1)         # (1, n_x, n_y, 1)
        n_valid = float(mask.sum().item())
        if n_valid == 0:
            return torch.tensor(0.0, device=u.device, requires_grad=True)
        # Mean over valid points × batch × time dims
        total_valid = n_valid * u_pred.shape[0] * u_pred.shape[-1]
        return (laplacian * mask4.float()).pow(2).sum() / total_valid


# ── Navier-Stokes 2D (Incompressible, Steady-State) ─────────────────────────
# PDE (steady, lid-driven cavity):
#   Momentum-x: u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
#   Momentum-y: u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
#   Continuity: u_x + v_y = 0
# Source: PINNacle src/pde/ns.py line 130-161 (NS2D_LidDriven)
# Domain: [0,1]^2, nu=1/100
# BCs: no-slip walls, lid u=a*x*(1-x) at top, p(0,0)=0
#
# NOTE: Brandon specified cylinder wake geometry (Flag 6). PINNacle's available
# NS data is lid-driven cavity and backstep flow, NOT cylinder wake.
# This implementation uses lid-driven cavity matching PINNacle's ns2d data.
# Cite: PINNacle [1]; Raissi et al. 2019 [4]

class NSPhysicsLoss(nn.Module):
    """
    Navier-Stokes 2D steady-state physics loss.

    Momentum residual: u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)  (x-component)
                       u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)  (y-component)
    Continuity: u_x + v_y = 0

    Combined: L_physics = L_momentum + lambda_c * L_continuity

    The continuity weight lambda_c is a LEARNED parameter initialized at 1.0.
    Incompressibility is a hard physical constraint, not a soft preference.

    Uses zero-pad (no-slip Dirichlet BCs) via F.pad.
    """

    def __init__(self, nu=NS2D_NU, lambda_c_init=1.0):
        super().__init__()
        self.nu = nu
        # Learned continuity weight (incompressibility enforcement)
        self.log_lambda_c = nn.Parameter(torch.tensor(math.log(lambda_c_init)))

    @property
    def lambda_c(self):
        return self.log_lambda_c.exp()

    def forward(self, u_pred, coords, pde_params):
        """
        Args:
            u_pred: (B, n_x, n_y, 3) — channels are (u, v, p)
            coords: dict with 'dx', 'dy'
            pde_params: dict (nu defaults to 1/100)
        Returns:
            Scalar physics loss (momentum + lambda_c * continuity)
        """
        nu = pde_params.get('nu', self.nu)
        dx = coords['dx']
        dy = coords.get('dy', dx)

        u_vel = u_pred[..., 0:1]  # (B, nx, ny, 1)
        v_vel = u_pred[..., 1:2]
        p     = u_pred[..., 2:3]

        # Derivatives of u in x and y
        u_xpad = F.pad(u_vel, (0,0, 0,0, 1,1), mode='constant', value=0.0)
        u_xm = u_xpad[:, :-2, :, :]; u_xp = u_xpad[:, 2:, :, :]
        u_ypad = F.pad(u_vel, (0,0, 1,1, 0,0), mode='constant', value=0.0)
        u_ym = u_ypad[:, :, :-2, :]; u_yp = u_ypad[:, :, 2:, :]
        u_x  = (u_xp - u_xm) / (2*dx)
        u_y  = (u_yp - u_ym) / (2*dy)
        u_xx = (u_xp - 2*u_vel + u_xm) / (dx**2)
        u_yy = (u_yp - 2*u_vel + u_ym) / (dy**2)

        # Derivatives of v in x and y
        v_xpad = F.pad(v_vel, (0,0, 0,0, 1,1), mode='constant', value=0.0)
        v_xm = v_xpad[:, :-2, :, :]; v_xp = v_xpad[:, 2:, :, :]
        v_ypad = F.pad(v_vel, (0,0, 1,1, 0,0), mode='constant', value=0.0)
        v_ym = v_ypad[:, :, :-2, :]; v_yp = v_ypad[:, :, 2:, :]
        v_x  = (v_xp - v_xm) / (2*dx)
        v_y  = (v_yp - v_ym) / (2*dy)
        v_xx = (v_xp - 2*v_vel + v_xm) / (dx**2)
        v_yy = (v_yp - 2*v_vel + v_ym) / (dy**2)

        # Pressure gradients
        p_xpad = F.pad(p, (0,0, 0,0, 1,1), mode='constant', value=0.0)
        p_xm = p_xpad[:, :-2, :, :]; p_xp = p_xpad[:, 2:, :, :]
        p_ypad = F.pad(p, (0,0, 1,1, 0,0), mode='constant', value=0.0)
        p_ym = p_ypad[:, :, :-2, :]; p_yp = p_ypad[:, :, 2:, :]
        p_x = (p_xp - p_xm) / (2*dx)
        p_y = (p_yp - p_ym) / (2*dy)

        # Momentum residuals (steady-state, no u_t/v_t)
        # PINNacle src/pde/ns.py line 156-157
        mom_x = u_vel*u_x + v_vel*u_y + p_x - nu*(u_xx + u_yy)
        mom_y = u_vel*v_x + v_vel*v_y + p_y - nu*(v_xx + v_yy)

        # Continuity: div(u) = 0
        # PINNacle src/pde/ns.py line 158
        continuity = u_x + v_y

        L_momentum   = (mom_x**2).mean() + (mom_y**2).mean()
        L_continuity = (continuity**2).mean()

        return L_momentum + self.lambda_c * L_continuity


# ── Kuramoto-Sivashinsky Physics Loss (Spectral Derivatives) ────────────────
# PDE: u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxx = 0
# Source: PINNacle src/pde/chaotic.py line 63, 77-83
# Domain: x in [0, 2*pi], t in [0,1], periodic BCs
# Parameters: alpha=100/16, beta=100/16^2, gamma=100/16^4
#
# The 2nd and 4th spatial derivatives are computed SPECTRALLY using FFT,
# following PINO (Li et al. 2021, Appendix B). Autograd for 4th-order
# derivatives is numerically unstable and ~4x slower.
#
# Spectral computation:
#   u_hat = FFT(u)
#   u_xx  = IFFT((ik)^2 * u_hat)
#   u_xxxx = IFFT((ik)^4 * u_hat)
# Cite: PINO Li et al. 2021 Appendix B [3]; PINNacle [1]

class KSPhysicsLoss(nn.Module):
    """
    Kuramoto-Sivashinsky equation physics loss with spectral derivatives.

    PDE: u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxx = 0
        Source: PINNacle src/pde/chaotic.py line 77-83
        alpha = 100/16, beta = 100/256, gamma = 100/65536
        x in [0, 2*pi], periodic BCs

    Spatial derivatives (u_x, u_xx, u_xxxx) computed spectrally via FFT.
    This follows PINO's implementation exactly (Li et al. 2021, Appendix B).

    For single-step prediction (T_out=1), time derivative is unavailable.
    Spatial residual proxy: alpha*u*u_x + beta*u_xx + gamma*u_xxxx
    """

    def __init__(self, alpha=KS_ALPHA, beta=KS_BETA, gamma=KS_GAMMA,
                 L=2*math.pi):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.L = L  # domain length for wavenumber computation

    def forward(self, u_pred, coords, pde_params):
        """
        Args:
            u_pred: (B, n_x, T_out) predicted field (periodic domain)
            coords: dict with 'dx' (not actually used — spectral needs L and n_x)
            pde_params: dict (alpha, beta, gamma default to KS values)
        Returns:
            Scalar physics loss (mean squared spatial residual)
        """
        alpha = pde_params.get('alpha', self.alpha)
        beta  = pde_params.get('beta',  self.beta)
        gamma = pde_params.get('gamma', self.gamma)

        u = u_pred  # (B, n_x, T_out)
        n_x = u.shape[1]

        # Wavenumbers for x in [0, L] with n_x periodic points
        # k = 2*pi/L * [0, 1, 2, ..., n_x/2, -n_x/2+1, ..., -1]
        k = torch.fft.fftfreq(n_x, d=self.L / (2*math.pi*n_x)).to(u.device)
        # Reshape for broadcasting: (1, n_x, 1)
        k = k.unsqueeze(0).unsqueeze(-1)
        ik = 1j * k

        # FFT along spatial dimension (dim=1)
        u_hat = torch.fft.fft(u, dim=1)

        # Spectral derivatives:
        #   u_x    = IFFT(ik * u_hat)
        #   u_xx   = IFFT((ik)^2 * u_hat) = IFFT(-k^2 * u_hat)
        #   u_xxxx = IFFT((ik)^4 * u_hat) = IFFT(k^4 * u_hat)
        u_x_hat    = ik * u_hat
        u_xx_hat   = (ik**2) * u_hat
        u_xxxx_hat = (ik**4) * u_hat

        u_x    = torch.fft.ifft(u_x_hat,    dim=1).real
        u_xx   = torch.fft.ifft(u_xx_hat,   dim=1).real
        u_xxxx = torch.fft.ifft(u_xxxx_hat, dim=1).real

        # KS spatial residual: alpha*u*u_x + beta*u_xx + gamma*u_xxxx
        # PINNacle src/pde/chaotic.py line 83:
        #   return u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx
        residual = alpha*u*u_x + beta*u_xx + gamma*u_xxxx
        return (residual**2).mean()


# ── Registry ─────────────────────────────────────────────────────────────────

# Grid spacings derived from PINNacle domain sizes and benchmark resolution.
# These match the data loaded by fno_benchmark.load_data().
GRID_DX = {
    "Burgers1D":            2.0 / 128,          # x in [-1,1], n_x=128
    "KuramotoSivashinsky":  2*math.pi / 512,    # x in [0,2pi], n_x=512
    "Burgers2D":            4.0 / 32,           # x in [0,4], n_xy=32
    "HeatComplexGeometry":  16.0 / 32,          # x in [-8,8], n_xy=32
    "NavierStokes2D":       1.0 / 64,           # x in [0,1], n_xy=64
}

GRID_DY = {
    "Burgers2D":            4.0 / 32,
    "HeatComplexGeometry":  24.0 / 32,          # y in [-12,12], n_xy=32
    "NavierStokes2D":       1.0 / 64,
}

# PDE_PHYSICS maps pde_name → (loss_class, default_pde_params)
PDE_PHYSICS = {
    "Burgers1D": {
        "class": BurgersPhysicsLoss,
        "pde_params": {"nu": BURGERS1D_NU, "dim": 1},
        "coords": lambda: {"dx": GRID_DX["Burgers1D"]},
        "description": "Burgers 1D: u_t + u*u_x - nu*u_xx = 0 (nu=0.01/pi)",
    },
    "Burgers2D": {
        "class": BurgersPhysicsLoss,
        "pde_params": {"nu": BURGERS2D_NU, "dim": 2},
        "coords": lambda: {"dx": GRID_DX["Burgers2D"]},
        "description": "Burgers 2D: vector system with nu=0.001, periodic BCs",
    },
    "HeatComplexGeometry": {
        "class": HeatPhysicsLoss,
        "pde_params": {},
        "coords": lambda: {"dx": GRID_DX["HeatComplexGeometry"],
                           "dy": GRID_DY["HeatComplexGeometry"]},
        "description": "Heat 2D: u_t - u_xx - u_yy = 0 (alpha=1, complex geometry)",
    },
    "KuramotoSivashinsky": {
        "class": KSPhysicsLoss,
        "pde_params": {"alpha": KS_ALPHA, "beta": KS_BETA, "gamma": KS_GAMMA},
        "coords": lambda: {"dx": GRID_DX["KuramotoSivashinsky"]},
        "description": "KS: u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxx = 0 (spectral)",
    },
    "NavierStokes2D": {
        "class": NSPhysicsLoss,
        "pde_params": {"nu": NS2D_NU},
        "coords": lambda: {"dx": GRID_DX["NavierStokes2D"],
                           "dy": GRID_DY["NavierStokes2D"]},
        "description": "NS 2D steady: momentum + continuity (lid-driven, nu=1/100)",
    },
}


def get_physics_loss(pde_name: str) -> tuple:
    """
    Get physics loss module and default parameters for a PDE.

    Returns:
        (loss_module, coords_dict, pde_params_dict)
        or (None, None, None) if PDE not registered.
    """
    entry = PDE_PHYSICS.get(pde_name)
    if entry is None:
        return None, None, None
    loss_module = entry["class"]()
    coords      = entry["coords"]()
    pde_params  = entry["pde_params"]
    return loss_module, coords, pde_params
