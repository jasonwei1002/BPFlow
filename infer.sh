#!/usr/bin/env bash
# Evaluate BPFlow on the CalFree test set (single-node, multi-GPU via torchrun).
# Runs live under output/<timestamp>/, so pass the checkpoint explicitly:
#   all visible GPUs (default):  CKPT=output/<ts>/checkpoint_best.pth bash infer.sh
#   N GPUs:                      NPROC=4 CKPT=... bash infer.sh
#   single GPU:                  NPROC=1 CKPT=... bash infer.sh
set -euo pipefail
cd "$(dirname "$0")"
NPROC="${NPROC:-gpu}"   # 'gpu' = all visible GPUs, or an integer
CKPT="${CKPT:?set CKPT=output/<timestamp>/checkpoint_best.pth}"
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.infer \
  --config bpflow/config/gpu.yaml --ckpt "$CKPT" \
  --split test --num -1 --use-ema "$@"
