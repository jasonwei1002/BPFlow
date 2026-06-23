#!/usr/bin/env bash
# Pretrain BPFlow on PulseDB (single-node, multi-GPU via torchrun), config gpu.yaml.
# Options:
#   --nproc <gpu|N>   number of processes ('gpu' = all visible GPUs, default)
#   all visible GPUs (default):  bash train_pulsedb.sh
#   N GPUs:                      bash train_pulsedb.sh --nproc 4
#   pick GPUs:                   CUDA_VISIBLE_DEVICES=0,1 bash train_pulsedb.sh --nproc 2
#   override config / args:      bash train_pulsedb.sh --config bpflow/config/other.yaml
#   override a config field:     bash train_pulsedb.sh training.lr=1e-4 'data.task_probs=[0.3,0.2,0.2,0.15,0.15]'
#   resume an interrupted run:   bash train_pulsedb.sh --resume output/<YYYYMMDD_HHMMSS>
#
# Per-direction specialists:
#   bash train_pulsedb.sh --config bpflow/config/gpu_ecg.yaml
#   bash train_pulsedb.sh --config bpflow/config/gpu_ppg.yaml
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="bpflow/config/gpu.yaml"
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
  --config "$CONFIG" "$@"
