# Source: PINO — Physics-Informed Neural Operator
# Paper: Li et al., "Physics-Informed Neural Operator for Learning Partial
#        Differential Equations", 2021.
#        https://arxiv.org/abs/2111.08907
# Code reference: https://github.com/neuraloperator/PINO
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# Re-exports from hpit_benchmark/pino_src/pino.py.

import importlib.util
import sys
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[2]
        / "hpit_benchmark" / "pino_src" / "pino.py")
_spec = importlib.util.spec_from_file_location("pino_src.pino", str(_SRC))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["pino_src.pino"] = _mod
_spec.loader.exec_module(_mod)

PINO1d = _mod.PINO1d
PINO2d = _mod.PINO2d

__all__ = ["PINO1d", "PINO2d"]
