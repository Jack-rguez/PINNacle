# Source: FNO, Li et al. ICLR 2021, https://arxiv.org/abs/2010.08895
# Code reference: https://github.com/neuraloperator/neuraloperator
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# This module re-exports the FNO1d and FNO2d classes from
# hpit_benchmark/fno_src/fno.py so they are accessible as src.model.fno.

import importlib.util
import sys
from pathlib import Path

_FNO_SRC = Path(__file__).resolve().parents[2] / "hpit_benchmark" / "fno_src" / "fno.py"
_spec = importlib.util.spec_from_file_location("fno_src.fno", str(_FNO_SRC))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["fno_src.fno"] = _mod
_spec.loader.exec_module(_mod)

FNO1d = _mod.FNO1d
FNO2d = _mod.FNO2d

__all__ = ["FNO1d", "FNO2d"]
