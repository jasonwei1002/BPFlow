#!/usr/bin/env bash
# Evaluate a finetuned PulseDB/CalFree checkpoint on the held-out test split over
# MULTIPLE seeds, then aggregate per-task per-metric mean+/-std. Config pulsedb_finetune.yaml
# (so the split matches finetune's fixed-seed 8:1:1). Single-node, multi-GPU via torchrun.
#
# Only the ODE initial noise depends on --seed (the test split is fixed by
# split_seed, the loader is shuffle=False), so every seed scores the SAME
# segments -> the spread across seeds is pure sampling variation, reported as mean+/-std.
#
# Per seed, --task all evaluates EVERY task the checkpoint trained on (metrics.json
# nested per task when >1, flat when 1):
#   ->ABP tasks (ecg_ppg2abp/ecg2abp/ppg2abp) -> clinical (AAMI/BHS) + waveform.
#   ECG/PPG translation tasks (ppg2ecg/ecg2ppg) -> waveform-only (normalized units).
#   Clinical TRUE is per-beat on the true ABP wave.
#
# Args: <checkpoint> [--nproc <gpu|N>] [extra args...]   (--nproc default 'gpu')
# Env:  SEEDS="0 1 2 3 4"   space-separated seeds to run (default below)
#       OUTBASE=<dir>       output root (default output/infer_ms_<timestamp>)
# Examples:
#   all visible GPUs (default):  bash infer_pulsedb.sh output/<ts>/checkpoint_best.pth
#   N GPUs:                      bash infer_pulsedb.sh output/<ts>/checkpoint_best.pth --nproc 4
#   pin one task:                bash infer_pulsedb.sh <path> --task ppg2abp
#   custom seeds / quick test:   SEEDS="1 2 3" bash infer_pulsedb.sh <path> --nproc 1 --num 100
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="bpflow/config/pulsedb_finetune.yaml"
NPROC=gpu
rest=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="${2:?--nproc needs a value (gpu or an integer)}"; shift 2;;
    --nproc=*) NPROC="${1#*=}"; shift;;
    *) rest+=("$1"); shift;;
  esac
done
set -- ${rest[@]+"${rest[@]}"}
CKPT="${1:?pass a checkpoint: bash infer_pulsedb.sh <path> [--nproc N]}"; shift

SEEDS="${SEEDS:-0 1 2 3 4}"
OUTBASE="${OUTBASE:-output/infer_ms_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTBASE"
echo "[infer_pulsedb.sh] multi-seed eval: seeds=[$SEEDS] ckpt=$CKPT -> $OUTBASE"

first=1
for S in $SEEDS; do
  out="$OUTBASE/seed_$S"
  # recon figures only on the first seed (the numbers, not figures, vary by seed)
  if [ "$first" -eq 1 ]; then plot=3; first=0; else plot=0; fi
  echo "[infer_pulsedb.sh] === seed $S -> $out ==="
  torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.infer \
    --config "$CONFIG" --ckpt "$CKPT" \
    --split test --num -1 --use-ema --seed "$S" --out "$out" --plot "$plot" "$@"
done

echo "[infer_pulsedb.sh] aggregating $OUTBASE/seed_*/metrics.json -> $OUTBASE/summary.json"
python3 - "$OUTBASE" <<'PY'
import json
import math
import sys
from pathlib import Path

base = Path(sys.argv[1])
files = sorted(base.glob("seed_*/metrics.json"))
if not files:
    sys.exit(f"no metrics.json found under {base}/seed_*/")


def normalize(report):
    """Wrap a flat single-task report ({waveform:...}) as {task: report} so the
    aggregator treats single- and multi-task metrics.json uniformly."""
    return {"_single": report} if "waveform" in report else report


def leaves(d, prefix=""):
    """Yield (dotted_path, float) for every numeric leaf (skips strings, e.g. BHS grade)."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{prefix}/{k}" if prefix else k)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix, float(d)


def mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # sample (n-1) std
    return m, math.sqrt(var)


runs = [normalize(json.loads(f.read_text())) for f in files]
n = len(runs)

acc = {}
for r in runs:
    for path, val in leaves(r):
        acc.setdefault(path, []).append(val)

summary = {}
for path, xs in acc.items():
    m, s = mean_std(xs)
    summary[path] = {"mean": m, "std": s, "n": len(xs), "values": xs}
(base / "summary.json").write_text(json.dumps(summary, indent=2))

# headline table (mean +/- std over seeds)
print(f"\n=== {n} seeds: {[f.parent.name for f in files]} ===")
KEY = [
    "waveform/MAE", "waveform/RMSE", "waveform/Pearson",
    "SBP/AAMI/SDE", "SBP/AAMI/pass", "SBP/BHS/<=5mmHg",
    "DBP/AAMI/SDE", "MAP/AAMI/SDE",
]
tasks = sorted({p.split("/", 1)[0] for p in summary})
for t in tasks:
    print(f"\n[{t}]")
    for k in KEY:
        full = f"{t}/{k}"
        if full in summary:
            v = summary[full]
            print(f"  {k:18} {v['mean']:.4f} +/- {v['std']:.4f}")
print(f"\nfull summary -> {base / 'summary.json'}")
PY
echo "[infer_pulsedb.sh] done."
