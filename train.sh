#!/usr/bin/env bash
# Train BPFlow (single-node, multi-GPU via torchrun).
#   all visible GPUs (default):  bash train.sh
#   N GPUs:                      NPROC=4 bash train.sh
#   single GPU:                  NPROC=1 bash train.sh
#   pick GPUs:                   CUDA_VISIBLE_DEVICES=0,1 bash train.sh
#   override config / args:      bash train.sh --config bpflow/config/other.yaml
set -euo pipefail
cd "$(dirname "$0")"
NPROC="${NPROC:-gpu}"   # 'gpu' = all visible GPUs, or an integer
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.train \
  --config bpflow/config/gpu.yaml "$@"
