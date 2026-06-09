#!/usr/bin/env bash
# Evaluate BPFlow on the CalFree test set.  Usage: bash infer.sh
set -euo pipefail
cd "$(dirname "$0")"
python -m bpflow.infer \
  --config bpflow/config/gpu.yaml \
  --ckpt output/bpflow_gpu_p10/checkpoint_best.pth \
  --split test --num -1 --use-ema "$@"
