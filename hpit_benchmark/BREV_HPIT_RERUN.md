# Brev HPIT Rerun Instructions

Generated: 2026-04-15. Applies the speed fixes pushed in commit "HPIT speedups
(bs=512, bf16, GPU-preload, 50 epochs)". Expected wall-clock: **~2-3 hours total
for all 5 PDEs** on an A100 (down from projected 375+ hours).

---

## Before anything: kill the currently running HPIT

If HPIT is still running in tmux:

```bash
tmux attach -t bench          # reattach to the session
# Press Ctrl+C to stop the current python3 hpit_pde_benchmark.py
# Wait for the process to exit
```

If Ctrl+C won't reach it, from a separate Brev terminal:

```bash
pkill -f hpit_pde_benchmark.py
```

---

## Pull the fixes

```bash
cd ~/PINNacle                 # or wherever the repo lives on Brev
git pull
```

You should see updates to:
- `hpit_benchmark/hpit_pde_benchmark.py`
- `hpit_benchmark/results/hpit_results.csv` (cleared)
- `hpit_benchmark/BREV_HPIT_RERUN.md` (this file)

---

## Delete any stale HPIT checkpoints on Brev

The old 9-min/epoch run may have written partial `.pt` checkpoints. Remove them
so the rerun trains from scratch:

```bash
rm -f hpit_benchmark/results/hpit_Burgers1D.pt
rm -f hpit_benchmark/results/hpit_Burgers2D.pt
rm -f hpit_benchmark/results/hpit_HeatComplexGeometry.pt
rm -f hpit_benchmark/results/hpit_KuramotoSivashinsky.pt
rm -f hpit_benchmark/results/hpit_NavierStokes2D.pt
```

The benchmark only loads from a checkpoint when you pass `--checkpoint <path>`.
By default it always trains from scratch, so deleting these is belt-and-braces.

---

## Run HPIT (fresh, all 5 PDEs)

Inside your tmux session (new or reattached):

```bash
cd ~/PINNacle
python3 hpit_benchmark/hpit_pde_benchmark.py
```

That's it. New defaults baked into the script:

| Parameter | Old | New |
|---|---|---|
| `--epochs` | 500 | **50** |
| `--train-batch-size` | 32 | **512** |
| `--lr` | 1e-3 | **4e-3** (sqrt-scaled for bs=512) |
| Data path | DataLoader + num_workers=2 | **Preloaded to GPU, index slicing** |
| AMP | fp16 + GradScaler | **bf16 (no GradScaler)** |
| `use_gradient_checkpointing` | True (dead flag) | False |

If you'd rather override defaults on the CLI:

```bash
python3 hpit_benchmark/hpit_pde_benchmark.py \
  --epochs 50 --train-batch-size 512 --lr 4e-3
```

Run a single PDE for sanity first if you want:

```bash
python3 hpit_benchmark/hpit_pde_benchmark.py --pde Burgers1D
```

---

## Detach and wait

```
Ctrl+B  then  D     # detach from tmux
```

You can close the laptop. Reattach with `tmux attach -t bench` to check.

---

## After HPIT finishes

Collect final results:

```bash
python3 hpit_benchmark/collect_results.py
```

This aggregates `hpit_benchmark/results/*.csv` into a combined summary.

---

## Expected runtime budget

- Per-epoch (Burgers1D, A100): ~25-40 seconds (was ~540 seconds)
- 50 epochs × ~35 s ≈ **~30 min per PDE**
- 5 PDEs × 30 min ≈ **~2.5 hours total**

If this still feels too slow, additional levers (not applied yet) are in
`soft-growing-pearl.md` plan file — trajectory subsampling (#8) would give
another 4x.

---

## If anything fails

Check the first error in the terminal. Common issues:

1. **bf16 unsupported** — should not happen on A100. If it does, override:
   ```bash
   python3 hpit_benchmark/hpit_pde_benchmark.py --device cpu
   ```
2. **OOM with bs=512** — unlikely on 80 GB A100 for this model, but:
   ```bash
   python3 hpit_benchmark/hpit_pde_benchmark.py --train-batch-size 256 --lr 2.8e-3
   ```
3. **LR scaling unstable** — halve the LR:
   ```bash
   python3 hpit_benchmark/hpit_pde_benchmark.py --lr 2e-3
   ```
