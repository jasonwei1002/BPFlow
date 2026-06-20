#!/usr/bin/env bash
# Evaluate a finetuned BPFlow checkpoint on the CalFree held-out test split
# (finetune.yaml: data.finetune true -> the fixed-seed 10% of CalFree the finetune
# never trained on). Single-node, multi-GPU via torchrun.
# Defaults: --task all evaluates EVERY task the checkpoint trained on (nested in
# metrics.json when >1, flat when 1):
#   ->ABP tasks (ecg_ppg2abp/ecg2abp/ppg2abp) -> clinical (AAMI/BHS) + waveform.
#   ECG/PPG translation tasks (ppg2ecg/ecg2ppg) -> waveform-only (normalized units).
#   --true-source both  : BOTH ABP clinical truths — per-beat on the true wave
#                         (true_waveform) AND the CSV cuff label (true_csv). Needs the
#                         sibling CSV (CalFree has it); --true-source waveform for wave-only.
#                         (Translation tasks ignore this — they have no cuff label.)
#   --plot 3            : recon figure with 3 example GT-vs-gen samples per task
#                         (infer_recon[_<task>].png); override with --plot N / --plot 0.
# Args: <checkpoint> [--nproc <gpu|N>] [extra args...]   (--nproc default 'gpu')
#   all visible GPUs (default):  bash infer.sh output/<ts>/checkpoint_best.pth
#   N GPUs:                      bash infer.sh output/<ts>/checkpoint_best.pth --nproc 4
#   pin one task:                bash infer.sh <path> --task ppg2abp
#   wave-truth only:             bash infer.sh <path> --true-source waveform
#   extra args:                  bash infer.sh <path> --nproc 1 --num 100
set -euo pipefail
cd "$(dirname "$0")"
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
CKPT="${1:?pass a checkpoint: bash infer.sh <path> [--nproc N]}"; shift
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.infer \
  --config bpflow/config/finetune.yaml --ckpt "$CKPT" \
  --split test --num -1 --use-ema --true-source both --plot 3 "$@"
