#!/usr/bin/env bash
# launch_all_tmux.sh — Master tmux launcher for all benchmark models
#
# One tmux window per model. HPIT starts first with a 60s head start,
# then all other models launch in parallel. HPIT ablation starts after
# the main HPIT window (sequential on same GPU, 120s delay).
#
# Usage:
#   bash hpit_benchmark/launch_all_tmux.sh           # full run
#   bash hpit_benchmark/launch_all_tmux.sh --dry-run # pipeline check
#
# Windows created:
#   hpit         — HPIT + PDE physics  (→ hpit_results.csv)
#   hpit_ablation— HPIT no physics     (→ hpit_results_ablation.csv)
#   pino         — PINO
#   fno          — FNO
#   gnot         — GNOT
#   deeponet     — DeepONet
#   mambano      — Mamba-NO
#   pinnacle     — PINNacle PINNs (stub — see run_pinnacle.sh)
#
# To cancel a model: Ctrl-C in its window, or:
#   tmux kill-window -t benchmark:<window-name>
# To restart a single model later:
#   tmux new-window -t benchmark -n fno "bash hpit_benchmark/run_fno.sh; read"
# To restart a single HPIT PDE:
#   python3 hpit_benchmark/hpit_pde_benchmark.py --pde NavierStokes2D --pde-physics --epochs 500
# To restart a single ablation PDE:
#   python3 hpit_benchmark/hpit_ablation_runner.py --pde NavierStokes2D --epochs 500
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="benchmark"
EXTRA_ARGS="${*}"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Window 0: HPIT with physics (starts immediately)
tmux new-session -d -s "$SESSION" -n hpit \
    "bash hpit_benchmark/run_hpit.sh $EXTRA_ARGS; echo 'HPIT done. Press Enter.'; read"
echo "Started HPIT in tmux session '$SESSION' window 'hpit'"

# Window 1: HPIT ablation — starts 120s after physics run begins
# (runs sequentially on same GPU; 120s gives physics run time to grab GPU memory)
echo "Waiting 120s before starting HPIT ablation + parallel models..."
sleep 120

tmux new-window -t "$SESSION" -n hpit_ablation \
    "bash hpit_benchmark/run_hpit_ablation.sh $EXTRA_ARGS; echo 'HPIT ablation done. Press Enter.'; read"

tmux new-window -t "$SESSION" -n pino \
    "bash hpit_benchmark/run_pino.sh $EXTRA_ARGS; echo 'PINO done. Press Enter.'; read"

tmux new-window -t "$SESSION" -n fno \
    "bash hpit_benchmark/run_fno.sh $EXTRA_ARGS; echo 'FNO done. Press Enter.'; read"

tmux new-window -t "$SESSION" -n gnot \
    "bash hpit_benchmark/run_gnot.sh $EXTRA_ARGS; echo 'GNOT done. Press Enter.'; read"

tmux new-window -t "$SESSION" -n deeponet \
    "bash hpit_benchmark/run_deeponet.sh $EXTRA_ARGS; echo 'DeepONet done. Press Enter.'; read"

tmux new-window -t "$SESSION" -n mambano \
    "bash hpit_benchmark/run_mambano.sh $EXTRA_ARGS; echo 'Mamba-NO done. Press Enter.'; read"

tmux new-window -t "$SESSION" -n pinnacle \
    "bash hpit_benchmark/run_pinnacle.sh $EXTRA_ARGS; echo 'PINNacle done. Press Enter.'; read"

echo "All models launched in tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
echo ""
echo "Windows:"
echo "  hpit          HPIT + physics     (→ hpit_results.csv)"
echo "  hpit_ablation HPIT no physics    (→ hpit_results_ablation.csv)"
echo "  pino          PINO"
echo "  fno           FNO"
echo "  gnot          GNOT"
echo "  deeponet      DeepONet"
echo "  mambano       Mamba-NO"
echo "  pinnacle      PINNacle PINNs (manual steps required)"
