"""
show_ablations.py — Print a live summary of hpit_results_ablation.csv.

Usage:
    python hpit_benchmark/show_ablations.py
    python hpit_benchmark/show_ablations.py --watch   # refresh every 30s
"""
import argparse
import csv
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
ABLATION_CSV = RESULTS_DIR / "hpit_results_ablation.csv"
MAIN_CSV     = RESULTS_DIR / "hpit_results.csv"

EXPECTED = [
    # (pde, label)
    ("HeatComplexGeometry", "no_physics"),
    ("HeatComplexGeometry", "physics=mass"),
    ("HeatComplexGeometry", "physics=energy"),
    ("HeatComplexGeometry", "physics=elevation"),
    ("Burgers1D",           "no_physics bs=64"),
    ("Burgers1D",           "no_physics bs=512"),
]

BASELINE = {
    "HeatComplexGeometry": 0.014132,   # our best HPIT result (no physics)
    "Burgers1D":           None,       # pending
}


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise(rows):
    """Group rows by PDE and format for display."""
    results = {}
    for r in rows:
        pde   = r["pde"]
        notes = r["notes"]
        l2    = r["l2rel"]
        bs    = None

        # Skip dry runs and errors
        if "dry_run" in notes or "ERROR" in notes:
            continue

        # Parse variant label from notes
        if "ablation|physics=" in notes:
            # e.g. "ablation|physics=mass+energy|bs=64"
            parts = dict(p.split("=") for p in notes.split("|")[1:] if "=" in p)
            label = f"physics={parts.get('physics','?')}"
            bs    = parts.get("bs", "?")
        elif "ablation_no_physics" in notes:
            # e.g. "ablation_no_physics_bs64"
            bs_part = notes.split("_bs")[-1] if "_bs" in notes else "?"
            label = "no_physics"
            bs    = bs_part
        elif notes in ("trained_pinnacle", "pinnacle_dat"):
            label = "no_physics (main run)"
            bs    = "64"
        else:
            label = notes
            bs    = "-"

        key = (pde, label, bs)
        results[key] = l2
    return results


def print_table(results, baseline):
    pdas = sorted(set(k[0] for k in results))
    if not pdas:
        print("  (no results yet — runs still training)")
        return

    col_w = 32
    print(f"\n  {'PDE':<26} {'Variant':<{col_w}} {'bs':>6}  {'L2 rel':>10}  {'vs baseline':>12}")
    print("  " + "-" * 90)

    for (pde, label, bs), l2 in sorted(results.items()):
        base = baseline.get(pde)
        try:
            l2f = float(l2)
            l2_str = f"{l2f:.6f}"
            if base is not None and l2f > 0:
                ratio = l2f / base
                cmp = f"{ratio:.2f}x {'worse' if ratio > 1 else 'better'}"
            else:
                cmp = "-"
        except (ValueError, TypeError):
            l2_str = str(l2)
            cmp = "-"
        print(f"  {pde:<26} {label:<{col_w}} {str(bs):>6}  {l2_str:>10}  {cmp:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 30 seconds until Ctrl+C")
    args = parser.parse_args()

    def show():
        ablation_rows = read_csv(ABLATION_CSV)
        main_rows     = read_csv(MAIN_CSV)
        all_rows = ablation_rows + main_rows
        results  = summarise(all_rows)

        print(f"\n{'='*94}")
        print(f"  HPIT Ablation Results  —  {ABLATION_CSV.name}")
        print(f"  Baseline: HeatComplexGeometry no_physics = {BASELINE['HeatComplexGeometry']}")
        print(f"{'='*94}")
        print_table(results, BASELINE)

        # Show what's still pending
        done_keys = set(results.keys())
        pending = []
        for pde, label in EXPECTED:
            found = any(k[0] == pde and label in k[1] for k in done_keys)
            if not found:
                pending.append(f"{pde} [{label}]")
        if pending:
            print(f"\n  Still running / not started:")
            for p in pending:
                print(f"    • {p}")
        print()

    if args.watch:
        try:
            while True:
                show()
                print("  [refreshing in 30s — Ctrl+C to stop]\n")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show()


if __name__ == "__main__":
    main()
