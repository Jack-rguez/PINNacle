"""
collect_results.py — Unifies HPIT and PINNacle benchmark results into one table.

Reads:
  hpit_benchmark/results/hpit_results.csv
  runs/*/result.csv  (PINNacle output format)

Produces:
  hpit_benchmark/results/unified_results.md
  hpit_benchmark/results/unified_results.csv

Table layout: rows = models, columns = PDEs, cells = L2 relative error.

Placeholder rows:
  - GNOT, Mamba-NO   → "see_paper"   (results from published papers)
  - FNO, DeepONet, PINO → "pending"  (awaiting benchmark runs)

Source:
  PINNacle (Hao et al., NeurIPS 2024): https://arxiv.org/abs/2306.08827
  PDEBench (Takamoto et al., NeurIPS 2022): https://arxiv.org/abs/2210.07182

Do not hardcode real numerical results — only placeholders allowed here.
"""

# Source: PINNacle, Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827

import csv
import glob
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
HPIT_CSV      = RESULTS_DIR / "hpit_results.csv"
FNO_CSV       = RESULTS_DIR / "fno_results.csv"
GNOT_CSV      = RESULTS_DIR / "gnot_results.csv"
DEEPONET_CSV  = RESULTS_DIR / "deeponet_results.csv"
PINO_CSV      = RESULTS_DIR / "pino_results.csv"
MAMBA_NO_CSV  = RESULTS_DIR / "mamba_no_results.csv"
FBPINN_CSV    = RESULTS_DIR / "fbpinn_results.csv"
PINNACLE_RUNS_DIR = BENCHMARK_DIR.parent / "runs"

OUT_MD = RESULTS_DIR / "unified_results.md"
OUT_CSV = RESULTS_DIR / "unified_results.csv"

# Canonical PDE column order
PDE_ORDER = [
    "Burgers1D",
    "Burgers2D",
    "HeatComplexGeometry",
    "KuramotoSivashinsky",
    "NavierStokes2D",
]

# Models with published-paper placeholders
PAPER_PLACEHOLDERS: Dict[str, str] = {}  # Mamba-NO now has its own implementation

# Models that need to be run but haven't been yet (all neural operators now implemented)
PENDING_PLACEHOLDERS: Dict[str, str] = {}  # empty — all 5 operators have result CSVs

# Result CSVs produced by each neural operator runner
_MODEL_CSVS = {
    "HPIT":     HPIT_CSV,
    "FNO":      FNO_CSV,
    "GNOT":     GNOT_CSV,
    "DeepONet": DEEPONET_CSV,
    "PINO":     PINO_CSV,
    "Mamba-NO": MAMBA_NO_CSV,
    "FBPINN":   FBPINN_CSV,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model_csv(csv_path: Path, default_model: str = "") -> Dict[str, Dict[str, str]]:
    """
    Generic loader for any benchmark results CSV with columns:
        pde, model, l2rel, ...

    Handles duplicate (pde, model) rows by keeping the LAST entry
    (most recent run wins).

    Returns: {model: {pde: l2rel_string}}
    """
    results: Dict[str, Dict[str, str]] = {}

    if not csv_path.exists():
        logger.debug("Results CSV not found: %s", csv_path)
        return results

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", default_model).strip()
            pde   = row.get("pde", "").strip()
            l2    = row.get("l2rel", "").strip()
            if model and pde:
                results.setdefault(model, {})[pde] = l2  # last row wins

    logger.info("Loaded %s: %d model(s)", csv_path.name,
                len(results))
    return results


def load_hpit_results() -> Dict[str, Dict[str, str]]:
    return load_model_csv(HPIT_CSV, default_model="HPIT")


def load_pinnacle_results() -> Dict[str, Dict[str, str]]:
    """
    Load PINNacle run result CSVs from runs/*/result.csv.

    Expected PINNacle result.csv format (two-column or multi-column):
        problem,l2rel     (minimal)
        or
        problem,model,l2rel,...

    Returns: {model: {pde: l2rel_string}}
    """
    results: Dict[str, Dict[str, str]] = {}

    pattern = str(PINNACLE_RUNS_DIR / "*" / "result.csv")
    files = glob.glob(pattern)
    if not files:
        logger.info("No PINNacle result.csv files found in %s", PINNACLE_RUNS_DIR)
        return results

    for filepath in sorted(files):
        run_name = Path(filepath).parent.name
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                pde = row.get("problem", row.get("pde", "")).strip()
                model = row.get("model", run_name).strip()
                # Try common column names
                l2 = (row.get("l2rel") or row.get("l2_rel") or
                      row.get("l2") or "").strip()
                if pde and l2:
                    results.setdefault(model, {})[pde] = l2

    logger.info("Loaded PINNacle results: %d model(s)", len(results))
    return results


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(all_results: Dict[str, Dict[str, str]],
                pde_columns: list) -> list:
    """
    Build list-of-dicts table with rows=models, columns=PDEs.
    Missing cells filled with '-'.
    """
    rows = []
    for model, pde_map in sorted(all_results.items()):
        row = {"model": model}
        for pde in pde_columns:
            row[pde] = pde_map.get(pde, "-")
        rows.append(row)

    # Add placeholder models at bottom
    for model, placeholder in {**PAPER_PLACEHOLDERS, **PENDING_PLACEHOLDERS}.items():
        if model not in all_results:
            row = {"model": model}
            for pde in pde_columns:
                row[pde] = placeholder
            rows.append(row)

    return rows


def format_cell(val: str) -> str:
    """Format a cell value for markdown — try to render as float."""
    if val in ("-", "nan", "pending", "see_paper", ""):
        return val
    try:
        f = float(val)
        return f"{f:.4f}"
    except ValueError:
        return val


def write_markdown(rows: list, pde_columns: list, output_path: Path):
    """Write unified results as a GitHub-flavored markdown table."""
    header = ["Model"] + pde_columns
    sep = ["---"] + ["---:"] * len(pde_columns)

    lines = []
    lines.append("# HPIT PDE Benchmark — Unified Results")
    lines.append("")
    lines.append("L2 relative error: `||u_pred - u_true||_2 / ||u_true||_2`")
    lines.append("Lower is better.  `nan` = no checkpoint (random weights).")
    lines.append("`see_paper` = value reported in the source paper.")
    lines.append("`pending` = benchmark run not yet complete.")
    lines.append("`-` = not tested.")
    lines.append("")

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for row in rows:
        cells = [row["model"]]
        for pde in pde_columns:
            cells.append(format_cell(row.get(pde, "-")))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Sources")
    lines.append("- PINNacle: Hao et al., NeurIPS 2024 — https://arxiv.org/abs/2306.08827")
    lines.append("- PDEBench: Takamoto et al., NeurIPS 2022 — https://arxiv.org/abs/2210.07182")
    lines.append("- FNO: Li et al., ICLR 2021 — https://arxiv.org/abs/2010.08895")
    lines.append("- GNOT: Hao et al., ICML 2023 — https://arxiv.org/abs/2302.14376")
    lines.append("- DeepONet: Lu et al., Nature Machine Intelligence, 2021 — https://doi.org/10.1038/s42256-021-00302-5")
    lines.append("- PINO: Li et al., 2021 — https://arxiv.org/abs/2111.08907")
    lines.append("- Mamba-NO: Gu & Dao, ICLR 2024 (Mamba SSM) — https://arxiv.org/abs/2312.00752")
    lines.append("  Operator wrapper inspired by LaMO (M3RG-IITD) — https://github.com/M3RG-IITD/LaMO")
    lines.append("  Pure-PyTorch selective scan; upgrades to mamba_ssm fast path on GPU.")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown written to %s", output_path)


def write_csv(rows: list, pde_columns: list, output_path: Path):
    """Write unified results as CSV."""
    fieldnames = ["model"] + pde_columns
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("CSV written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, str]] = {}

    # Load each model's results CSV
    for model_name, csv_path in _MODEL_CSVS.items():
        rows = load_model_csv(csv_path, default_model=model_name)
        for m, pde_map in rows.items():
            all_results.setdefault(m, {}).update(pde_map)

    # Load PINNacle PINN results (runs/*/result.csv)
    pinnacle = load_pinnacle_results()
    for model, pde_map in pinnacle.items():
        all_results.setdefault(model, {}).update(pde_map)

    if not all_results:
        logger.warning("No results found. Run benchmarks first.")

    table = build_table(all_results, PDE_ORDER)
    write_markdown(table, PDE_ORDER, OUT_MD)
    write_csv(table, PDE_ORDER, OUT_CSV)

    logger.info("Done. Unified results saved to %s and %s", OUT_MD, OUT_CSV)


if __name__ == "__main__":
    main()
