#!/usr/bin/env bash
# Finetune a pretrained baseline on the CalFree domain (8:1:1 stratified split),
# mirroring bpflow's finetune_pulsedb.sh. Produces weights only; score the held-out
# 10% test split with infer_baseline_pulsedb.sh. New output/baselines/<timestamp>/.
# Args: <model> <direction> <pretrained_ckpt> [--nproc <gpu|N>] [overrides...]
#   bash finetune_baseline_pulsedb.sh nabnet ecg2abp output/baselines/<pre>/checkpoint_best.pth
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?usage: bash finetune_baseline_pulsedb.sh <model> <direction> <ckpt> [--nproc N]}"; shift
DIRECTION="${1:?usage: bash finetune_baseline_pulsedb.sh <model> <direction> <ckpt> [--nproc N]}"; shift
CKPT="${1:?pass a pretrained checkpoint path}"; shift
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
# Finetune = repurpose CalFree test_npy into an 8:1:1 split (stratified per subject).
FTARGS=(--model "$MODEL" --direction "$DIRECTION" --init-ckpt "$CKPT"
        data.finetune=true data.train_npy=rawdata/pulsedb/CalFree_Test_Subset.npy
        data.finetune_split_mode=stratified)
# Single process (nproc 1) skips torchrun (no DDP rendezvous); nproc>1 / 'gpu' = DDP.
if [ "$NPROC" = 1 ]; then
  python -m bpflow_baselines.train "${FTARGS[@]}" "$@"
else
  torchrun --standalone --nproc_per_node="$NPROC" -m bpflow_baselines.train "${FTARGS[@]}" "$@"
fi
