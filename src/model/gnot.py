# Source: GNOT, Hao et al., ICML 2023
# Paper: https://arxiv.org/abs/2302.14376
# Code reference: https://github.com/thu-ml/GNOT
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# Re-exports GNOT1d, GNOT2d, and GNOTOperator from
# hpit_benchmark/gnot_src/gnot.py for use as src.model.gnot.

import importlib.util
import sys
from pathlib import Path

_GNOT_SRC = (Path(__file__).resolve().parents[2]
             / "hpit_benchmark" / "gnot_src" / "gnot.py")
_spec = importlib.util.spec_from_file_location("gnot_src.gnot", str(_GNOT_SRC))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["gnot_src.gnot"] = _mod
_spec.loader.exec_module(_mod)

GNOT1d        = _mod.GNOT1d
GNOT2d        = _mod.GNOT2d
GNOTOperator  = _mod.GNOTOperator

__all__ = ["GNOT1d", "GNOT2d", "GNOTOperator"]
