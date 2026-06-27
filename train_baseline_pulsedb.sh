#!/usr/bin/env bash
# Pretrain a baseline on PulseDB (Train_Subset; test = CalFree). Mirrors bpflow's
# train_pulsedb.sh. Produces weights only; finetune with finetune_baseline_pulsedb.sh,
# then score the held-out test split with infer_baseline_pulsedb.sh.
# base.yaml already defaults data.train_npy=Train_Subset.npy, so no data override here.
# Args: <model> <direction> [--nproc <gpu|N>] [overrides...]
#   bash train_baseline_pulsedb.sh nabnet ecg2abp                 # all GPUs
#   bash train_baseline_pulsedb.sh patchtst ppg2abp --nproc 4
#   bash train_baseline_pulsedb.sh wavenet ecg2ppg training.lr=5e-4   # dotted overrides
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?usage: bash train_baseline_pulsedb.sh <model> <direction> [--nproc N]}"; shift
DIRECTION="${1:?usage: bash train_baseline_pulsedb.sh <model> <direction> [--nproc N]}"; shift
NPROC=gpu
rest=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="${2:?--nproc needs a value}"; shift 2;;
    --nproc=*) NPROC="${1#*=}"; shift;;
    *) rest+=("$1"); shift;;
  esac
done
set -- ${rest[@]+"${rest[@]}"}
ARGS=(--model "$MODEL" --direction "$DIRECTION")
# Single process (nproc 1) skips torchrun (no DDP rendezvous); nproc>1 / 'gpu' = DDP.
if [ "$NPROC" = 1 ]; then
  python -m bpflow_baselines.train "${ARGS[@]}" "$@"
else
  torchrun --standalone --nproc_per_node="$NPROC" -m bpflow_baselines.train "${ARGS[@]}" "$@"
fi
