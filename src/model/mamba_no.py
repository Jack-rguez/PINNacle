# Source: Mamba Neural Operator
# Paper: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
#        ICLR 2024. https://arxiv.org/abs/2312.00752
# Code reference: https://github.com/state-spaces/mamba
# LaMO operator design: https://github.com/M3RG-IITD/LaMO
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# Re-exports from hpit_benchmark/mamba_no_src/mamba_no.py.
# Pure-PyTorch implementation — no CUDA kernels required for dry-run.

import importlib.util
import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[2]
        / "hpit_benchmark" / "mamba_no_src" / "mamba_no.py")
_spec = importlib.util.spec_from_file_location("mamba_no_src.mamba_no", str(_SRC))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["mamba_no_src.mamba_no"] = _mod
_spec.loader.exec_module(_mod)

MambaOperator1d = _mod.MambaOperator1d
MambaOperator2d = _mod.MambaOperator2d
MambaLayer      = _mod.MambaLayer

__all__ = ["MambaOperator1d", "MambaOperator2d", "MambaLayer"]
