#!/usr/bin/env bash
# Pretrain BPFlow on UCI (single-node, multi-GPU via torchrun), config uci.yaml
# (seq_len 1024, patch_size 8, UCI ABP constants). Trains on the UCI train fold
# (segment 80/20 train/val); the UCI test fold is reserved for finetune_uci.sh.
# Options:
#   --nproc <gpu|N>   number of processes ('gpu' = all visible GPUs, default)
#   all visible GPUs (default):  bash train_uci.sh
#   N GPUs:                      bash train_uci.sh --nproc 4
#   pick GPUs:                   CUDA_VISIBLE_DEVICES=0,1 bash train_uci.sh --nproc 2
#   override a config field:     bash train_uci.sh training.lr=1e-4
#   resume an interrupted run:   bash train_uci.sh --resume output/<YYYYMMDD_HHMMSS>
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="bpflow/config/uci.yaml"
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
