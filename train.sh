#!/usr/bin/env bash
# Train BPFlow (single-node, multi-GPU via torchrun).
# Options:
#   --nproc <gpu|N>   number of processes ('gpu' = all visible GPUs, default)
#   all visible GPUs (default):  bash train.sh
#   N GPUs:                      bash train.sh --nproc 4
#   pick GPUs:                   CUDA_VISIBLE_DEVICES=0,1 bash train.sh --nproc 2
#   override config / args:      bash train.sh --config bpflow/config/other.yaml
#   override a config field:     bash train.sh training.lr=1e-4 model.use_demo=true
#   resume an interrupted run:   bash train.sh --resume output/<YYYYMMDD_HHMMSS>

# ECG单模态训练命令
# bash train.sh --config bpflow/config/gpu_ecg.yaml

# PPG单模态训练命令
# bash train.sh --config bpflow/config/gpu_ppg.yaml
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
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.train \
  --config bpflow/config/gpu.yaml "$@"
