# Source: DeepONet, Lu et al., Nature Machine Intelligence, 2021
# Paper: https://doi.org/10.1038/s42256-021-00302-5
# Code reference: https://github.com/lululululuuu/DeepONet
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# Note: DeepXDE (deepxde/nn/pytorch/deeponet.py) also contains a DeepONet
# implementation tied to the DeepXDE training framework. This module provides
# a standalone version for the neural operator benchmark.
#
# Re-exports from hpit_benchmark/deeponet_src/deeponet.py.

import importlib.util
import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[2]
        / "hpit_benchmark" / "deeponet_src" / "deeponet.py")
_spec = importlib.util.spec_from_file_location("deeponet_src.deeponet", str(_SRC))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["deeponet_src.deeponet"] = _mod
_spec.loader.exec_module(_mod)

DeepONet1d       = _mod.DeepONet1d
DeepONet2d       = _mod.DeepONet2d
DeepONet2dSteady = _mod.DeepONet2dSteady
DeepONetCore     = _mod.DeepONetCore

__all__ = ["DeepONet1d", "DeepONet2d", "DeepONet2dSteady", "DeepONetCore"]
