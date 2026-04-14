"""
collect_fbpinn.py — Extract L2 results from FBPINN .npy loss files.

FBPINN (run from fbpinns/ dir) writes loss arrays to:
    fbpinns/results/models/<RUN_NAME>/loss_<step>.npy

Each .npy file is a 2D array with rows:
    [step, mstep, fstep, l2_component_0, ..., physics_loss, ...]

Column 3 (index 3) is the L2 relative error for the first output component.
We load the latest loss file for each run and take the last row.

PDE mapping (by script name → PINNacle PDE name):
    run_burger  → Burgers2D        (coupled 2D Burgers)
    run_heat    → HeatComplexGeometry
    run_chaotic → KuramotoSivashinsky
    run_ns      → NavierStokes2D

Output: hpit_benchmark/results/fbpinn_results.csv
"""

import csv
import glob
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).resolve().parents[1]
FBPINN_MODELS_DIR = REPO_ROOT / "fbpinns" / "results" / "models"
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
OUT_CSV      = RESULTS_DIR / "fbpinn_results.csv"

# Map partial run-name strings → PINNacle PDE column names
# FBPINN RUN strings look like "bench2_<grid>_CoupledBurgers_..." etc.
PDE_MAP = {
    "burger":  "Burgers2D",
    "Burger":  "Burgers2D",
    "heat":    "HeatComplexGeometry",
    "Heat":    "HeatComplexGeometry",
    "chaotic": "KuramotoSivashinsky",
    "Chaotic": "KuramotoSivashinsky",
    "KS":      "KuramotoSivashinsky",
    "ns":      "NavierStokes2D",
    "NS":      "NavierStokes2D",
    "Navier":  "NavierStokes2D",
}

def guess_pde(run_name: str) -> str:
    for key, pde in PDE_MAP.items():
        if key in run_name:
            return pde
    return run_name  # fallback: use raw run name


def collect() -> dict:
    """Returns {pde_name: l2_rel_str}"""
    if not FBPINN_MODELS_DIR.exists():
        logger.warning("FBPINN models dir not found: %s", FBPINN_MODELS_DIR)
        return {}

    results = {}
    run_dirs = [d for d in FBPINN_MODELS_DIR.iterdir() if d.is_dir()]

    if not run_dirs:
        logger.warning("No FBPINN run directories found in %s", FBPINN_MODELS_DIR)
        return {}

    for run_dir in sorted(run_dirs):
        run_name = run_dir.name
        loss_files = sorted(run_dir.glob("loss_*.npy"))
        if not loss_files:
            logger.warning("No loss files in %s — skipping", run_dir)
            continue

        # Latest loss file = highest step number
        latest = loss_files[-1]
        try:
            arr = np.load(str(latest))  # shape (n_steps_logged, n_cols)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            last_row = arr[-1]
            l2 = float(last_row[3])    # column 3 = first L2 component
        except Exception as e:
            logger.error("Failed to load %s: %s", latest, e)
            continue

        pde = guess_pde(run_name)
        logger.info("FBPINN run %-50s  →  %-30s  L2=%.4f", run_name, pde, l2)
        # Keep best (lowest) L2 if multiple runs map to same PDE
        if pde not in results or l2 < float(results[pde]):
            results[pde] = f"{l2:.6f}"

    return results


def write_csv(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pde", "model", "l2rel", "notes"])
        writer.writeheader()
        for pde, l2 in results.items():
            writer.writerow({"pde": pde, "model": "FBPINN", "l2rel": l2,
                             "notes": "fbpinn_npy"})
    logger.info("FBPINN results written to %s (%d PDEs)", OUT_CSV, len(results))


if __name__ == "__main__":
    results = collect()
    if results:
        write_csv(results)
    else:
        logger.warning("No FBPINN results found — CSV not written.")
