# Source: PDEBench, Takamoto et al. NeurIPS 2022,
# https://arxiv.org/abs/2210.07182
# Adapted from https://github.com/pdebench/PDEBench/blob/main/pdebench/models/fno/fno.py
#
# Original FNO architecture:
# Source: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
# ICLR 2021, https://arxiv.org/abs/2010.08895
#
# This implementation follows the PDEBench convention for 1D and 2D problems:
#   1D: input (batch, x, T_in)       → output (batch, x, T_out)
#   2D: input (batch, x, y, T_in)    → output (batch, x, y, T_out)
#
# Key design choices that match PDEBench:
#   - SpectralConv with truncated modes (modes1, modes2)
#   - Bypass W branch using a pointwise Conv
#   - GELU activation (used in PDEBench's FNO1d / FNO2d)
#   - Input grid coordinates appended as extra channels

"""
Fourier Neural Operator (FNO) — 1D and 2D variants for PDE benchmarking.

Usage:
    from fno_src.fno import FNO1d, FNO2d

    # 1D: e.g. Burgers1D, KuramotoSivashinsky
    model = FNO1d(modes=16, width=64, T_in=10, T_out=1)
    y = model(x)   # x: (batch, n_x, T_in) → y: (batch, n_x, T_out)

    # 2D: e.g. NavierStokes2D
    model = FNO2d(modes1=12, modes2=12, width=32, T_in=10, T_out=1)
    y = model(x)   # x: (batch, n_x, n_y, T_in) → y: (batch, n_x, n_y, T_out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1-D Fourier layer
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """
    Spectral convolution in 1D Fourier space.
    Truncates to `modes` lowest-frequency modes.

    Source: Li et al. 2021, https://arxiv.org/abs/2010.08895
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def _compl_mul1d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (batch, in, modes), w: (in, out, modes) → (batch, out, modes)
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, n_x)
        bsz = x.shape[0]
        n_x = x.shape[-1]

        x_ft = torch.fft.rfft(x, dim=-1)                             # (..., n_x//2+1)
        out_ft = torch.zeros(bsz, self.out_channels, n_x // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self._compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=n_x, dim=-1)                # (batch, out, n_x)


class FNOBlock1d(nn.Module):
    """Single FNO block: SpectralConv + pointwise bypass + activation."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO1d(nn.Module):
    """
    Fourier Neural Operator for 1-D time-dependent PDEs.

    Inputs/outputs follow the PDEBench convention:
        x_in  : (batch, n_x, T_in)   — T_in historical timesteps
        x_out : (batch, n_x, T_out)  — T_out predicted timesteps

    An extra spatial coordinate channel (normalised ∈ [0,1]) is appended
    to the input before lifting, as in the original FNO paper.

    Source: PDEBench, Takamoto et al. NeurIPS 2022, https://arxiv.org/abs/2210.07182
    Source: Li et al. ICLR 2021, https://arxiv.org/abs/2010.08895
    """

    def __init__(self, modes: int = 16, width: int = 64,
                 T_in: int = 10, T_out: int = 1, n_layers: int = 4):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.width = width

        # +1 for the appended grid coordinate
        self.fc0 = nn.Linear(T_in + 1, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, T_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_x, T_in)
        bsz, n_x, _ = x.shape

        # Append normalised grid coordinate
        grid = torch.linspace(0, 1, n_x, device=x.device).view(1, n_x, 1).expand(bsz, -1, -1)
        x = torch.cat([x, grid], dim=-1)   # (batch, n_x, T_in+1)

        # Lift
        x = self.fc0(x)                    # (batch, n_x, width)
        x = x.permute(0, 2, 1)            # (batch, width, n_x)

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 1)            # (batch, n_x, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                    # (batch, n_x, T_out)
        return x


# ---------------------------------------------------------------------------
# 2-D Fourier layer
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """
    Spectral convolution in 2D Fourier space.

    Source: Li et al. 2021, https://arxiv.org/abs/2010.08895
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (batch, in, m1, m2), w: (in, out, m1, m2) → (batch, out, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, n_x, n_y)
        bsz, _, n_x, n_y = x.shape

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(bsz, self.out_channels, n_x, n_y // 2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # Lower-left and lower-right corners of spectrum
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(n_x, n_y), dim=(-2, -1))


class FNOBlock2d(nn.Module):
    """Single FNO block for 2D: SpectralConv2d + pointwise bypass + activation."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2-D time-dependent PDEs.

    Inputs/outputs follow the PDEBench convention:
        x_in  : (batch, n_x, n_y, T_in)    — T_in historical timesteps
        x_out : (batch, n_x, n_y, T_out)   — T_out predicted timesteps

    A 2-channel normalised (x, y) coordinate grid is appended to the input.

    Source: PDEBench, Takamoto et al. NeurIPS 2022, https://arxiv.org/abs/2210.07182
    Source: Li et al. ICLR 2021, https://arxiv.org/abs/2010.08895
    """

    def __init__(self, modes1: int = 12, modes2: int = 12,
                 width: int = 32, T_in: int = 10, T_out: int = 1,
                 n_layers: int = 4):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.width = width

        # +2 for appended (x, y) grid coordinates
        self.fc0 = nn.Linear(T_in + 2, width)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, T_out)

    @staticmethod
    def _get_grid(bsz: int, n_x: int, n_y: int, device) -> torch.Tensor:
        """Return normalised (x, y) coordinate grid of shape (batch, n_x, n_y, 2)."""
        gx = torch.linspace(0, 1, n_x, device=device)
        gy = torch.linspace(0, 1, n_y, device=device)
        gx, gy = torch.meshgrid(gx, gy, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)              # (n_x, n_y, 2)
        return grid.unsqueeze(0).expand(bsz, -1, -1, -1)  # (batch, n_x, n_y, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_x, n_y, T_in)
        bsz, n_x, n_y, _ = x.shape

        grid = self._get_grid(bsz, n_x, n_y, x.device)
        x = torch.cat([x, grid], dim=-1)   # (batch, n_x, n_y, T_in+2)

        # Lift
        x = self.fc0(x)                    # (batch, n_x, n_y, width)
        x = x.permute(0, 3, 1, 2)         # (batch, width, n_x, n_y)

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)         # (batch, n_x, n_y, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                    # (batch, n_x, n_y, T_out)
        return x
