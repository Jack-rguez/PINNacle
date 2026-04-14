# Source: GNOT, Hao et al., ICML 2023
# Paper: https://arxiv.org/abs/2302.14376
# Code:  https://github.com/thu-ml/GNOT
#
# This file is a DGL-free, einops-free adaptation of the GNOT architecture
# (cgpt.py from the official repository) for use with PDEBench-formatted data.
#
# What changed from the original:
#   - Removed DGL graph dependency: trunk (query) coordinates are passed as
#     dense tensors (B, N_q, d_coord) instead of DGL graph node features.
#   - Removed einops: replaced rearrange() with equivalent permute+reshape.
#   - Removed MultipleTensors: replaced with plain Python lists.
#   - Added GNOT1d / GNOT2d convenience wrappers that match the FNO1d / FNO2d
#     tensor interface used in fno_benchmark.py, enabling fair side-by-side
#     comparison on the same train/test splits.
#   - MLP helper class added (was imported from models.mlp in the original).
#
# Architectural fidelity preserved:
#   - LinearAttention (l1 normalization, same formula as original)
#   - LinearCrossAttention (multi-input cross-attention, same formula)
#   - CrossAttentionBlock (cross-attn → self-attn → MLP, same structure)
#   - CGPTNO trunk/branch MLP → stacked CrossAttentionBlocks → output MLP
#   - Fourier positional embedding (optional, disabled by default)
#
# Tensor convention (GNOT):
#   trunk x : (B, N_q, d_coord)   — query coordinate points
#   branch z : list of [(B, N_b, d_b), ...]  — input function evaluations
#   output   : (B, N_q, output_size)
#
# Tensor convention for 1D/2D wrappers:
#   GNOT1d : (B, n_x, T_in) → (B, n_x, T_out)   [same as FNO1d]
#   GNOT2d : (B, n_x, n_y, T_in) → (B, n_x, n_y, T_out)  [same as FNO2d]

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# MLP — simple feedforward net (originally in models/mlp.py)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable activation.
    Source: GNOT, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
    """

    _ACT = {
        "gelu":    nn.GELU,
        "relu":    nn.ReLU,
        "tanh":    nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 n_layers: int = 2, act: str = "gelu"):
        super().__init__()
        act_cls = self._ACT.get(act, nn.GELU)
        sizes = [in_size] + [hidden_size] * (n_layers - 1) + [out_size]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Fourier positional embedding (optional)
# Source: GNOT, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
# ---------------------------------------------------------------------------

def horizontal_fourier_embedding(X: torch.Tensor, n: int = 3) -> torch.Tensor:
    """
    Appends sinusoidal features at 2n+1 frequency scales to each coordinate.
    X: (B, T, C) → output: (B, T, C*(4n+3))
    """
    freqs = (2 ** torch.linspace(-n, n, 2 * n + 1, device=X.device))
    freqs = freqs[None, None, None, :]           # (1, 1, 1, 2n+1)
    X_ = X.unsqueeze(-1).expand(*X.shape, 2 * n + 1)
    X_cos = torch.cos(freqs * X_)
    X_sin = torch.sin(freqs * X_)
    return torch.cat([X.unsqueeze(-1), X_cos, X_sin], dim=-1).view(
        X.shape[0], X.shape[1], -1
    )


# ---------------------------------------------------------------------------
# LinearAttention — self-attention with L1-normalised keys/queries
# Source: GNOT cgpt.py, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """
    Linear (kernel) self-attention with three normalization modes:
      'l1'      — softmax keys and queries, L1 normalizer (default)
      'galerkin' — softmax keys and queries, 1/T normalizer
      'l2'      — L1-norm keys and queries, absolute L1 normalizer

    Forward: (B, T, C) → (B, T, C)
    """

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float = 0.0,
                 attn_type: str = "l1"):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embd // n_head
        self.attn_type = attn_type

        self.query = nn.Linear(n_embd, n_embd)
        self.key   = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj  = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)

    def forward(self, x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = x if y is None else y
        B, T1, C = x.shape
        T2 = y.shape[1]
        h, d = self.n_head, self.head_dim

        q = self.query(x).view(B, T1, h, d).transpose(1, 2)  # (B, h, T1, d)
        k = self.key(y).view(B, T2, h, d).transpose(1, 2)    # (B, h, T2, d)
        v = self.value(y).view(B, T2, h, d).transpose(1, 2)

        if self.attn_type == "l1":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_sum = k.sum(dim=-2, keepdim=True)                  # (B, h, 1, d)
            D_inv = 1.0 / (q * k_sum).sum(dim=-1, keepdim=True) # (B, h, T1, 1)
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1.0 / T2
        elif self.attn_type == "l2":
            q = q / q.norm(dim=-1, keepdim=True, p=1).clamp(min=1e-9)
            k = k / k.norm(dim=-1, keepdim=True, p=1).clamp(min=1e-9)
            k_sum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / (q * k_sum).abs().sum(dim=-1, keepdim=True).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown attn_type: {self.attn_type}")

        context = k.transpose(-2, -1) @ v                       # (B, h, d, d)
        out = self.attn_drop((q @ context) * D_inv + q)         # (B, h, T1, d)

        # (B, h, T1, d) → (B, T1, h*d) — equivalent to einops rearrange
        out = out.transpose(1, 2).contiguous().view(B, T1, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# LinearCrossAttention — multi-input cross-attention
# Source: GNOT cgpt.py, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
# ---------------------------------------------------------------------------

class LinearCrossAttention(nn.Module):
    """
    Cross-attention where a single query set attends to multiple input
    (branch) sequences.  Accumulates attention output additively over
    all branch inputs.

    Forward:
        x   : (B, T_q, C)  — trunk queries
        y   : list of (B, T_b_i, C) — branch key/value sequences
        → out: (B, T_q, C)
    """

    def __init__(self, n_embd: int, n_head: int, n_inputs: int,
                 attn_pdrop: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.n_inputs = n_inputs

        self.query  = nn.Linear(n_embd, n_embd)
        self.keys   = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_inputs)])
        self.values = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_inputs)])
        self.proj   = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)

    def forward(self, x: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:
        B, T1, C = x.shape
        h, d = self.n_head, self.head_dim

        q = self.query(x).view(B, T1, h, d).transpose(1, 2)   # (B, h, T1, d)
        q = q.softmax(dim=-1)
        out = q                                                  # accumulate

        for i in range(self.n_inputs):
            T2 = y[i].shape[1]
            k = self.keys[i](y[i]).view(B, T2, h, d).transpose(1, 2)
            v = self.values[i](y[i]).view(B, T2, h, d).transpose(1, 2)
            k = k.softmax(dim=-1)
            k_sum = k.sum(dim=-2, keepdim=True)
            D_inv = 1.0 / (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-9)
            out = out + self.attn_drop((q @ (k.transpose(-2, -1) @ v)) * D_inv)

        out = out.transpose(1, 2).contiguous().view(B, T1, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# Source: GNOT cgpt.py, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    One GNOT transformer block:
        1. Cross-attention: trunk queries attend to branch inputs
        2. FFN residual on trunk
        3. Self-attention on trunk
        4. FFN residual on trunk
    """

    def __init__(self, n_embd: int, n_head: int, n_inputs: int,
                 n_inner: int, act: str = "gelu",
                 ffn_dropout: float = 0.0, attn_dropout: float = 0.0,
                 attn_type: str = "l1"):
        super().__init__()

        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2_branches = nn.ModuleList(
            [nn.LayerNorm(n_embd) for _ in range(n_inputs)]
        )
        self.ln3  = nn.LayerNorm(n_embd)
        self.ln4  = nn.LayerNorm(n_embd)
        self.ln5  = nn.LayerNorm(n_embd)

        self.crossattn = LinearCrossAttention(
            n_embd, n_head, n_inputs, attn_pdrop=attn_dropout
        )
        self.selfattn = LinearAttention(
            n_embd, n_head, attn_pdrop=attn_dropout, attn_type=attn_type
        )

        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU,
                   "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}.get(act, nn.GELU)

        self.mlp1 = nn.Sequential(
            nn.Linear(n_embd, n_inner), act_cls(),
            nn.Linear(n_inner, n_embd),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(n_embd, n_inner), act_cls(),
            nn.Linear(n_inner, n_embd),
        )
        self.drop1 = nn.Dropout(ffn_dropout)
        self.drop2 = nn.Dropout(ffn_dropout)

    def forward(self, x: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:
        # y branches: apply per-branch layernorm
        y_normed = [self.ln2_branches[i](y[i]) for i in range(len(y))]

        x = x + self.drop1(self.crossattn(self.ln1(x), y_normed))
        x = x + self.mlp1(self.ln3(x))
        x = x + self.drop2(self.selfattn(self.ln4(x)))
        x = x + self.mlp2(self.ln5(x))
        return x


# ---------------------------------------------------------------------------
# Core GNOT operator (DGL-free)
# ---------------------------------------------------------------------------

class GNOTOperator(nn.Module):
    """
    Cross-attention GPT Neural Operator (CGPTNO) — DGL-free dense version.

    Source: GNOT, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376

    trunk_size   : dimension of query coordinate features (e.g. 1 for 1D x)
    branch_sizes : list of input function feature dimensions
    output_size  : number of predicted scalar fields
    """

    def __init__(self,
                 trunk_size: int = 1,
                 branch_sizes: List[int] = None,
                 output_size: int = 1,
                 n_layers: int = 3,
                 n_hidden: int = 64,
                 n_head: int = 1,
                 n_inner: int = 256,
                 mlp_layers: int = 2,
                 attn_type: str = "l1",
                 act: str = "gelu",
                 ffn_dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 horiz_fourier_dim: int = 0):
        super().__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        effective_trunk = (trunk_size * (4 * horiz_fourier_dim + 3)
                           if horiz_fourier_dim > 0 else trunk_size)
        self.branch_sizes = branch_sizes or []
        n_inputs = len(self.branch_sizes)

        self.trunk_mlp = MLP(effective_trunk, n_hidden, n_hidden,
                             n_layers=mlp_layers, act=act)
        self.branch_mlps = nn.ModuleList(
            [MLP(bs, n_hidden, n_hidden, n_layers=mlp_layers, act=act)
             for bs in self.branch_sizes]
        )

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(n_hidden, n_head, n_inputs, n_inner,
                                act=act, ffn_dropout=ffn_dropout,
                                attn_dropout=attn_dropout,
                                attn_type=attn_type)
            for _ in range(n_layers)
        ])

        self.out_mlp = MLP(n_hidden, n_hidden, output_size,
                           n_layers=mlp_layers, act=act)

    def forward(self,
                coords: torch.Tensor,
                branch_inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        coords        : (B, N_q, d_coord) — query coordinate points
        branch_inputs : list of (B, N_b_i, d_b_i) — input function samples
        returns       : (B, N_q, output_size)
        """
        x = coords
        if self.horiz_fourier_dim > 0:
            x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)

        x = self.trunk_mlp(x)                                 # (B, N_q, H)
        z = [self.branch_mlps[i](branch_inputs[i])
             for i in range(len(self.branch_sizes))]           # list of (B, N_b, H)

        for block in self.blocks:
            x = block(x, z)

        return self.out_mlp(x)                                 # (B, N_q, out)


# ---------------------------------------------------------------------------
# 1-D wrapper — same tensor interface as FNO1d
# ---------------------------------------------------------------------------

class GNOT1d(nn.Module):
    """
    GNOT adapter for 1D time-dependent PDEs.

    Tensor interface (same as FNO1d in fno_src/fno.py):
        x_in  : (B, n_x, T_in)   — T_in historical timesteps
        x_out : (B, n_x, T_out)  — T_out predicted timesteps

    Implementation:
        Trunk   = normalised spatial grid coords  (B, n_x, 1)
        Branch  = T_in input function evaluations (B, n_x, T_in)
        GNOT maps trunk coords × branch values → output predictions.

    Source: GNOT, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
    """

    def __init__(self, T_in: int = 10, T_out: int = 1,
                 n_hidden: int = 64, n_head: int = 1, n_layers: int = 3,
                 mlp_layers: int = 2, attn_type: str = "l1",
                 ffn_dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.T_out = T_out

        # trunk_size=1 (x coord), branch_sizes=[T_in] (input function)
        self.gnot = GNOTOperator(
            trunk_size=1,
            branch_sizes=[T_in],
            output_size=T_out,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            n_inner=4 * n_hidden,
            mlp_layers=mlp_layers,
            attn_type=attn_type,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_x, T_in)
        B, n_x, T_in = x.shape

        # Trunk: normalised [0, 1] spatial coordinates
        grid = torch.linspace(0, 1, n_x, device=x.device)
        coords = grid.view(1, n_x, 1).expand(B, -1, -1)     # (B, n_x, 1)

        # Branch: the T_in input timesteps as function values
        branch = [x]                                          # (B, n_x, T_in)

        out = self.gnot(coords, branch)                       # (B, n_x, T_out)
        return out


# ---------------------------------------------------------------------------
# 2-D wrapper — same tensor interface as FNO2d
# ---------------------------------------------------------------------------

class GNOT2d(nn.Module):
    """
    GNOT adapter for 2D time-dependent PDEs.

    Tensor interface (same as FNO2d in fno_src/fno.py):
        x_in  : (B, n_x, n_y, T_in)    — T_in historical timesteps
        x_out : (B, n_x, n_y, T_out)   — T_out predicted timesteps

    Implementation:
        Trunk   = normalised (x, y) grid coords   (B, n_x*n_y, 2)
        Branch  = T_in input function evaluations  (B, n_x*n_y, T_in)

    Source: GNOT, Hao et al. ICML 2023, https://arxiv.org/abs/2302.14376
    """

    def __init__(self, T_in: int = 10, T_out: int = 1,
                 n_hidden: int = 32, n_head: int = 1, n_layers: int = 3,
                 mlp_layers: int = 2, attn_type: str = "l1",
                 ffn_dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.T_out = T_out

        # trunk_size=2 (x,y coords), branch_sizes=[T_in] (input function)
        self.gnot = GNOTOperator(
            trunk_size=2,
            branch_sizes=[T_in],
            output_size=T_out,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            n_inner=4 * n_hidden,
            mlp_layers=mlp_layers,
            attn_type=attn_type,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
        )

    @staticmethod
    def _make_grid(B: int, n_x: int, n_y: int,
                   device: torch.device) -> torch.Tensor:
        """Return normalised (x, y) coordinates, shape (B, n_x*n_y, 2)."""
        gx = torch.linspace(0, 1, n_x, device=device)
        gy = torch.linspace(0, 1, n_y, device=device)
        gx, gy = torch.meshgrid(gx, gy, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).view(-1, 2)    # (n_x*n_y, 2)
        return grid.unsqueeze(0).expand(B, -1, -1)           # (B, n_x*n_y, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_x, n_y, T_in)
        B, n_x, n_y, T_in = x.shape
        N = n_x * n_y

        coords = self._make_grid(B, n_x, n_y, x.device)     # (B, N, 2)
        branch = [x.view(B, N, T_in)]                        # (B, N, T_in)

        out = self.gnot(coords, branch)                      # (B, N, T_out)
        return out.view(B, n_x, n_y, self.T_out)
