# Source: Mamba-NO — Mamba Neural Operator for PDE Benchmarking
#
# Architecture based on:
#   Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
#   ICLR 2024. https://arxiv.org/abs/2312.00752
#   Code: https://github.com/state-spaces/mamba
#
# Operator wrapper design inspired by:
#   LaMO (Latent Mamba Operator), M3RG-IITD, 2025
#   https://github.com/M3RG-IITD/LaMO
#
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# Implementation notes:
#   - Pure-PyTorch selective scan — no CUDA kernels required for CPU/dry-run.
#   - If `mamba_ssm` is installed (GPU environment), automatically uses the
#     optimized fast path via mamba_ssm.modules.mamba_simple.Mamba.
#   - Input/output interface identical to FNO: (B, n_x, T_in) → (B, n_x, T_out)
#     for 1D and (B, n_x, n_y, T_in) → (B, n_x, n_y, T_out) for 2D.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try to import compiled mamba_ssm (GPU fast path). Fall back to pure PyTorch.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.modules.mamba_simple import Mamba as _MambaFast
    _HAS_MAMBA_SSM = True
except ImportError:
    _MambaFast = None
    _HAS_MAMBA_SSM = False


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba layer (runs on CPU, no compiled kernels)
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """
    Single Mamba SSM layer.

    Input/output: (B, L, d_model)

    Implements selective state spaces with ZOH discretization.
    Pure-PyTorch sequential scan — correct but O(L) per layer.
    When mamba_ssm is installed, delegates to the optimised fast path.

    Reference: Gu & Dao 2024, Algorithm 2 (simplified scan).
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.d_inner  = int(expand * d_model)

        if _HAS_MAMBA_SSM:
            # Use the compiled fast path when available (GPU training)
            self._fast = _MambaFast(d_model=d_model, d_state=d_state,
                                     d_conv=d_conv, expand=expand)
        else:
            self._fast = None

        # Pure-PyTorch path
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                   kernel_size=d_conv, padding=d_conv - 1,
                                   groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # A: (d_inner, d_state) — log parameterisation for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A))

        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(self, u: torch.Tensor,
                         delta: torch.Tensor,
                         A: torch.Tensor,
                         B: torch.Tensor,
                         C: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan (sequential, pure PyTorch).

        Args:
            u:     (B, L, D)   — input sequence
            delta: (B, L, D)   — input-dependent time step
            A:     (D, N)      — state transition (negative for stability)
            B:     (B, L, N)   — input projection
            C:     (B, L, N)   — output projection

        Returns: y (B, L, D)
        """
        B_sz, L, D = u.shape
        N = A.shape[1]

        # Discretise A: dA = exp(delta * A)  [ZOH]
        # delta: (B, L, D), A: (D, N) → dA: (B, L, D, N)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,D,N)
        # dB = delta * B * u  expanded
        dB = (delta.unsqueeze(-1) * B.unsqueeze(2) *
              u.unsqueeze(-1))  # (B, L, D, N)

        h = torch.zeros(B_sz, D, N, device=u.device, dtype=u.dtype)
        ys = []
        for l in range(L):
            h = h * dA[:, l, :, :] + dB[:, l, :, :]   # (B, D, N)
            y = (h * C[:, l, None, :]).sum(-1)          # (B, D)
            ys.append(y)
        return torch.stack(ys, dim=1)  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        if self._fast is not None and x.is_cuda:
            return residual + self._fast(x)

        # Pure PyTorch path
        B_sz, L, _ = x.shape
        xz = self.in_proj(x)                            # (B, L, 2*d_inner)
        xi, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        # Causal depthwise conv
        xi_t = xi.transpose(1, 2)                       # (B, d_inner, L)
        xi_t = self.conv1d(xi_t)[:, :, :L]             # trim padding
        xi = F.silu(xi_t.transpose(1, 2))              # (B, L, d_inner)

        # SSM parameters
        x_dbc = self.x_proj(xi)                        # (B, L, N*2+1)
        dt_raw, B_ssm, C_ssm = x_dbc.split(
            [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))       # (B, L, d_inner)
        A = -torch.exp(self.log_A)                     # (d_inner, N)

        y = self._selective_scan(xi, delta, A, B_ssm, C_ssm)  # (B, L, d_inner)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + y


# ---------------------------------------------------------------------------
# Mamba Neural Operator — 1D
# ---------------------------------------------------------------------------

class MambaOperator1d(nn.Module):
    """
    Mamba Neural Operator for 1D time-dependent PDEs.

    Input:  (B, n_x, T_in)  — T_in spatial snapshots
    Output: (B, n_x, T_out) — T_out predicted snapshots

    Strategy: treat the spatial dimension n_x as the sequence length.
    For each batch, apply n_layers Mamba layers across spatial positions.
    Temporal features are embedded as channels.

    Inspired by the 1D sequence application in LaMO and FNO architecture.
    """

    def __init__(self, T_in: int, T_out: int,
                 d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 n_layers: int = 4):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        self.input_proj  = nn.Linear(T_in, d_model)
        self.layers      = nn.ModuleList([
            MambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, T_out)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_x, T_in) → (B, n_x, T_out)"""
        h = self.input_proj(x)              # (B, n_x, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.output_proj(h)          # (B, n_x, T_out)


# ---------------------------------------------------------------------------
# Mamba Neural Operator — 2D
# ---------------------------------------------------------------------------

class MambaOperator2d(nn.Module):
    """
    Mamba Neural Operator for 2D PDEs.

    Input:  (B, n_x, n_y, T_in)
    Output: (B, n_x, n_y, T_out)

    Strategy: flatten (n_x, n_y) into sequence length n_x*n_y,
    apply Mamba layers, reshape back.  Coordinates are implicit in position.

    For 2D operator learning this is analogous to treating spatial patches
    as tokens (cf. ViT, FNO2d).
    """

    def __init__(self, T_in: int, T_out: int,
                 d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 n_layers: int = 4):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        self.input_proj  = nn.Linear(T_in, d_model)
        self.layers      = nn.ModuleList([
            MambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, T_out)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_x, n_y, T_in) → (B, n_x, n_y, T_out)"""
        B, n_x, n_y, T_in = x.shape
        # Flatten spatial dims: (B, n_x*n_y, T_in)
        h = x.reshape(B, n_x * n_y, T_in)
        h = self.input_proj(h)              # (B, n_x*n_y, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        h = self.output_proj(h)             # (B, n_x*n_y, T_out)
        return h.reshape(B, n_x, n_y, self.T_out)
