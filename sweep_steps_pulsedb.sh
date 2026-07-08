#!/usr/bin/env bash
# Sampling-steps ablation (quality vs ODE steps): score ONE finetuned PulseDB
# checkpoint at several step counts. Pure inference, no retraining. Each step
# count runs the standard multi-seed eval (infer_pulsedb.sh) and gets its own
# per-task mean+/-std summary; a final per-task table collects waveform
# MAE/RMSE/Pearson across step counts (full clinical block -> steps_summary.json).
#
# Tasks: whatever the checkpoint trained on (infer's --task all default) -- a
# specialist checkpoint yields its one task, a unified one yields all of them.
# Pin a single direction by forwarding --task <name> (e.g. --task ecg_ppg2abp).
#
# For flow matching (euler) the number of function evaluations equals num_steps,
# so this is a quality-vs-compute curve. Defends the "few-step deterministic
# sampling" claim.
#
# Args: <finetuned_checkpoint> [--nproc <gpu|N>] [extra infer args...]
# Env:  STEPS="1 2 4 8 16"   ODE step counts to sweep (default)
#       SEEDS="0 1 2 3 4"    seeds per step count (forwarded to infer_pulsedb.sh)
#       OUTBASE=<dir>        sweep output root (default output/steps_sweep_<ts>)
# Examples:
#   bash sweep_steps_pulsedb.sh output/pdb_ft_<ts>_<dir>/checkpoint_best.pth
#   STEPS="4 8 16" SEEDS="0 1 2" bash sweep_steps_pulsedb.sh <ckpt> --nproc 4
#   pin one direction:  bash sweep_steps_pulsedb.sh <ckpt> --task ecg_ppg2abp
set -euo pipefail
cd "$(dirname "$0")"

CKPT="${1:?pass a finetuned checkpoint: bash sweep_steps_pulsedb.sh <path> [--nproc N]}"; shift
STEPS="${STEPS:-1 2 4 8 16}"
SWEEP="${OUTBASE:-output/steps_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SWEEP"
echo "[sweep_steps] ckpt=$CKPT  steps=[$STEPS]  -> $SWEEP"

for K in $STEPS; do
  echo "[sweep_steps] === num_steps=$K -> $SWEEP/steps_$K ==="
  OUTBASE="$SWEEP/steps_$K" bash infer_pulsedb.sh "$CKPT" \
    --num-steps "$K" "$@"
done

echo "[sweep_steps] collecting $SWEEP/steps_*/summary.json -> $SWEEP/steps_summary.json"
python3 - "$SWEEP" $STEPS <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
steps = [int(x) for x in sys.argv[2:]]
# Per-task metric suffixes pulled from each step's seed-aggregated summary.json.
# Its keys are "<task>/<metric...>", task == "_single" for a one-task checkpoint
# or the real task name(s) under --task all. Full clinical block so steps_summary.json
# mirrors the per-seed metrics: waveform + per-site AAMI (ME/SDE/pass) + BHS
# (<=5/10/15 mmHg). BHS letter grade is a per-seed string (not aggregated), so it
# is not collected. ->ABP tasks fill the clinical keys; ECG/PPG bridge tasks are
# waveform-only (their clinical keys resolve to null).
WAVE = ["waveform/MAE", "waveform/RMSE", "waveform/Pearson"]
SITES = ["SBP", "DBP", "MAP"]
CLIN = [f"{bp}/AAMI/{m}" for bp in SITES for m in ("ME", "SDE", "pass")] + \
       [f"{bp}/BHS/<={t}mmHg" for bp in SITES for t in (5, 10, 15)]
WANT = WAVE + CLIN

# steps_summary.json: {num_steps: {task: {metric_suffix: {mean,std,n,values}}}}.
table = {}
for K in steps:
    f = base / f"steps_{K}" / "summary.json"
    if not f.exists():
        print(f"  [warn] missing {f} -- skipping num_steps={K}")
        continue
    s = json.loads(f.read_text())
    tasks = sorted({k.split("/", 1)[0] for k in s})
    table[K] = {t: {w: s.get(f"{t}/{w}") for w in WANT} for t in tasks}

(base / "steps_summary.json").write_text(json.dumps(table, indent=2))

# compact console table per task (waveform quality only; full clinical block -> file)
def fmt(m, d=3):
    return f"{m['mean']:.{d}f}+/-{m['std']:.{d}f}" if m else "n/a"

all_tasks = sorted({t for K in steps if table.get(K) for t in table[K]})
print(f"\n=== steps sweep: {base.name} ===")
for t in all_tasks:
    label = "single-task" if t == "_single" else t
    print(f"\n[{label}]")
    hdr = f"{'steps(NFE)':>10} | {'MAE':>16} | {'RMSE':>16} | {'Pearson':>16}"
    print(hdr); print("-" * len(hdr))
    for K in steps:
        row = (table.get(K) or {}).get(t)
        if not row:
            continue
        print(f"{K:>10} | {fmt(row.get('waveform/MAE')):>16} | "
              f"{fmt(row.get('waveform/RMSE')):>16} | {fmt(row.get('waveform/Pearson'), 4):>16}")
print(f"\nfull metrics (per task: AAMI ME/SDE/pass + BHS per site) -> {base / 'steps_summary.json'}")
PY
echo "[sweep_steps] done -> $SWEEP/steps_summary.json"
