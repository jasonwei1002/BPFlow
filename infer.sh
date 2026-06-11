#!/usr/bin/env bash
# Evaluate a finetuned BPFlow checkpoint on the CalFree held-out test split
# (finetune.yaml: data.finetune true -> the fixed-seed 10% of CalFree the finetune
# never trained on). Single-node, multi-GPU via torchrun.
# Args: <checkpoint> [--nproc <gpu|N>] [extra args...]   (--nproc default 'gpu')
#   all visible GPUs (default):  bash infer.sh output/<ts>/checkpoint_best.pth
#   N GPUs:                      bash infer.sh output/<ts>/checkpoint_best.pth --nproc 4
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
  --split test --num -1 --use-ema "$@"
