# Source: DeepONet — Deep Operator Network
# Paper:  Lu et al., "Learning nonlinear operators via DeepONet based on the
#         universal approximation theorem of operators", Nature Machine
#         Intelligence, 2021.
#         https://doi.org/10.1038/s42256-021-00302-5
# Code reference: https://github.com/lululululuuu/DeepONet
#
# This is a standalone PyTorch implementation that does NOT depend on DeepXDE,
# enabling clean comparison against FNO and GNOT on the same PINNacle data.
#
# Architecture (Cartesian product version for efficiency):
#   Branch net : maps u(x_s)_{s=1}^{S}  →  ℝ^p
#   Trunk  net : maps y (query coord)    →  ℝ^p
#   Output     : u(y) = Σ_{k=1}^p branch_k · trunk_k(y) + b
#                     = branch · trunk^T + b
#
# The Cartesian product version evaluates branch ONCE per training sample
# and trunk ONCE per query point, then multiplies them — this is O(Bp + Qp)
# instead of O(BQ) per sample.
#
# For multi-output (e.g. u,v in Burgers2D or u,v,p in NS):
#   Each scalar field has its own output head (shared trunk, separate branches).
#
# Wrappers:
#   DeepONet1d : (B, n_x, T_in) → (B, n_x, T_out)   same interface as FNO1d
#   DeepONet2d : (B, n_x, n_y, T_in) → (B, n_x, n_y, T_out)  same as FNO2d

import math
import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------------------------------------------------------
# Utility: fully-connected network
# ---------------------------------------------------------------------------

class FCNet(nn.Module):
    """Standard fully-connected network with configurable hidden layers."""

    def __init__(self, in_dim: int, hidden: List[int], out_dim: int,
                 act: str = "tanh"):
        super().__init__()
        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU,
                   "swish": nn.SiLU}
        act_cls = act_map.get(act, nn.Tanh)

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Core DeepONet (Cartesian product version, single output)
# ---------------------------------------------------------------------------

class DeepONetCore(nn.Module):
    """
    DeepONet (Cartesian product) for a single scalar output field.

    Lu et al., Nature Machine Intelligence 2021.
    https://doi.org/10.1038/s42256-021-00302-5

    Args:
        branch_in   : input dimension of the branch net (= n_sensors)
        trunk_in    : input dimension of the trunk net  (= coord_dim)
        p           : latent dimension (branch and trunk output size)
        branch_h    : hidden layer widths for branch net
        trunk_h     : hidden layer widths for trunk net

    Forward:
        branch_input : (B, branch_in) — input function values at sensor locations
        trunk_input  : (Q, trunk_in)  — Q query coordinates
        returns      : (B, Q)         — output field at Q query points
    """

    def __init__(self, branch_in: int, trunk_in: int, p: int = 128,
                 branch_h: Optional[List[int]] = None,
                 trunk_h:  Optional[List[int]] = None,
                 act: str = "tanh"):
        super().__init__()
        branch_h = branch_h or [128, 128, 128]
        trunk_h  = trunk_h  or [128, 128, 128]

        self.branch = FCNet(branch_in, branch_h, p, act=act)
        self.trunk  = FCNet(trunk_in,  trunk_h,  p, act=act)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_input)                   # (B, p)
        t = torch.tanh(self.trunk(trunk_input))         # (Q, p)
        # Cartesian product: each branch sample × each trunk query
        out = b @ t.T                                   # (B, Q)
        return out + self.bias


# ---------------------------------------------------------------------------
# 1-D wrapper — same tensor interface as FNO1d
# ---------------------------------------------------------------------------

class DeepONet1d(nn.Module):
    """
    DeepONet for 1D time-dependent PDEs.

    Tensor interface (same as FNO1d):
        x_in  : (B, n_x, T_in)   — T_in historical timesteps on 1D grid
        x_out : (B, n_x, T_out)  — T_out predicted timesteps

    Branch net: flattens input to (B, n_x * T_in) → encodes input function
    Trunk  net: normalised [0, 1] spatial coords (n_x, 1) → spatial basis

    Source: Lu et al., Nature Machine Intelligence 2021,
            https://doi.org/10.1038/s42256-021-00302-5
    """

    def __init__(self, n_x: int = 128, T_in: int = 10, T_out: int = 1,
                 p: int = 128,
                 branch_h: Optional[List[int]] = None,
                 trunk_h:  Optional[List[int]] = None):
        super().__init__()
        self.n_x   = n_x
        self.T_in  = T_in
        self.T_out = T_out

        # One DeepONet per output timestep (DeepONets share trunk)
        self.trunk_net = FCNet(1, trunk_h or [128, 128, 128], p, act="tanh")

        # Each output step has its own branch net
        self.branch_nets = nn.ModuleList([
            FCNet(n_x * T_in, branch_h or [128, 128, 128], p, act="tanh")
            for _ in range(T_out)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(T_out)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_x, T_in = x.shape
        assert n_x == self.n_x, f"Expected n_x={self.n_x}, got {n_x}"

        # Flatten input function: (B, n_x * T_in)
        branch_in = x.reshape(B, n_x * T_in)

        # Trunk: normalised [0, 1] spatial coordinates
        device = x.device
        grid   = torch.linspace(0, 1, n_x, device=device).view(n_x, 1)  # (n_x, 1)
        t_emb  = torch.tanh(self.trunk_net(grid))                         # (n_x, p)

        outputs = []
        for i in range(self.T_out):
            b_emb = self.branch_nets[i](branch_in)             # (B, p)
            out_i = b_emb @ t_emb.T + self.biases[i]           # (B, n_x)
            outputs.append(out_i.unsqueeze(-1))                 # (B, n_x, 1)

        return torch.cat(outputs, dim=-1)                       # (B, n_x, T_out)


# ---------------------------------------------------------------------------
# 2-D wrapper — same tensor interface as FNO2d
# ---------------------------------------------------------------------------

class DeepONet2d(nn.Module):
    """
    DeepONet for 2D time-dependent PDEs.

    Tensor interface (same as FNO2d):
        x_in  : (B, n_x, n_y, T_in)    — T_in historical timesteps on 2D grid
        x_out : (B, n_x, n_y, T_out)   — T_out predicted timesteps

    Branch net: flattens input to (B, n_x * n_y * T_in)
    Trunk  net: normalised (x, y) coordinates → (n_x * n_y, 2)

    Source: Lu et al., Nature Machine Intelligence 2021,
            https://doi.org/10.1038/s42256-021-00302-5
    """

    def __init__(self, n_x: int = 32, n_y: int = 32, T_in: int = 5,
                 T_out: int = 1, p: int = 64,
                 branch_h: Optional[List[int]] = None,
                 trunk_h:  Optional[List[int]] = None):
        super().__init__()
        self.n_x   = n_x
        self.n_y   = n_y
        self.T_in  = T_in
        self.T_out = T_out

        self.trunk_net = FCNet(2, trunk_h or [64, 64, 64], p, act="tanh")

        self.branch_nets = nn.ModuleList([
            FCNet(n_x * n_y * T_in, branch_h or [64, 64, 64], p, act="tanh")
            for _ in range(T_out)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(T_out)
        ])

    @staticmethod
    def _make_grid(n_x: int, n_y: int, device: torch.device) -> torch.Tensor:
        """Normalised (x, y) coordinates, shape (n_x*n_y, 2)."""
        gx = torch.linspace(0, 1, n_x, device=device)
        gy = torch.linspace(0, 1, n_y, device=device)
        gx, gy = torch.meshgrid(gx, gy, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (n_x*n_y, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n_x, n_y, T_in = x.shape
        N = n_x * n_y

        branch_in = x.reshape(B, N * T_in)                     # (B, N*T_in)
        grid      = self._make_grid(n_x, n_y, x.device)        # (N, 2)
        t_emb     = torch.tanh(self.trunk_net(grid))            # (N, p)

        outputs = []
        for i in range(self.T_out):
            b_emb = self.branch_nets[i](branch_in)             # (B, p)
            out_i = b_emb @ t_emb.T + self.biases[i]           # (B, N)
            outputs.append(out_i.view(B, n_x, n_y, 1))
        return torch.cat(outputs, dim=-1)                       # (B, n_x, n_y, T_out)


# ---------------------------------------------------------------------------
# Steady-state variant for NavierStokes2D (param → field)
# ---------------------------------------------------------------------------

class DeepONet2dSteady(nn.Module):
    """
    DeepONet for 2D steady-state PDEs (NavierStokes2D LidDriven).

    Tensor interface:
        x_in  : (B, n_x, n_y, 1)   — parameter field (normalised 'a')
        x_out : (B, n_x, n_y, 3)   — (u, v, p) fields

    Branch: (B, n_x*n_y) — the parameter field flattened
    Trunk:  (n_x*n_y, 2)  — (x, y) coordinates
    Output: 3 DeepONets for u, v, p

    Source: Lu et al., Nature Machine Intelligence 2021,
            https://doi.org/10.1038/s42256-021-00302-5
    """

    def __init__(self, n_x: int = 64, n_y: int = 64, n_out: int = 3,
                 p: int = 64,
                 branch_h: Optional[List[int]] = None,
                 trunk_h:  Optional[List[int]] = None):
        super().__init__()
        self.n_x   = n_x
        self.n_y   = n_y
        self.n_out = n_out

        self.trunk_net = FCNet(2, trunk_h or [64, 64, 64], p, act="tanh")
        self.branch_nets = nn.ModuleList([
            FCNet(n_x * n_y, branch_h or [64, 64, 64], p, act="tanh")
            for _ in range(n_out)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_out)
        ])

    @staticmethod
    def _make_grid(n_x: int, n_y: int, device: torch.device) -> torch.Tensor:
        gx = torch.linspace(0, 1, n_x, device=device)
        gy = torch.linspace(0, 1, n_y, device=device)
        gx, gy = torch.meshgrid(gx, gy, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_x, n_y, 1)
        B, n_x, n_y, _ = x.shape
        N = n_x * n_y

        branch_in = x.reshape(B, N)                             # (B, N)
        grid      = self._make_grid(n_x, n_y, x.device)        # (N, 2)
        t_emb     = torch.tanh(self.trunk_net(grid))            # (N, p)

        outputs = []
        for i in range(self.n_out):
            b_emb = self.branch_nets[i](branch_in)             # (B, p)
            out_i = b_emb @ t_emb.T + self.biases[i]           # (B, N)
            outputs.append(out_i.view(B, n_x, n_y, 1))
        return torch.cat(outputs, dim=-1)                       # (B, n_x, n_y, 3)
