#!/usr/bin/env bash
# Finetune a PulseDB-pretrained model on the CalFree domain (config finetune.yaml).
# CalFree is split 8:1:1 (per-segment) into train/val/test; this finetunes on the
# 80% train (val for early stop) and produces weights only -- run_test_after_train
# is OFF (it can deadlock under DDP). Score the held-out 10% test split separately
# with: bash infer_pulsedb.sh <ckpt>. Runs live under a NEW output/<timestamp>/.
# Args: <pretrained_checkpoint> [--nproc <gpu|N>] [extra args...]   (--nproc default 'gpu')
#   all visible GPUs (default):  bash finetune_pulsedb.sh output/<pretrain_ts>/checkpoint_best.pth
#   N GPUs:                      bash finetune_pulsedb.sh output/<pretrain_ts>/checkpoint_best.pth --nproc 4
#   override config / args:      bash finetune_pulsedb.sh <path> --nproc 4 --config bpflow/config/other.yaml
#   data-efficiency sweep:       bash finetune_pulsedb.sh <path> data.finetune_train_ratio=0.25 training.lr=3e-5
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="bpflow/config/finetune.yaml"
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
CKPT="${1:?pass a pretrained checkpoint: bash finetune_pulsedb.sh <path> [--nproc N]}"; shift
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.train \
  --config "$CONFIG" --init-ckpt "$CKPT" "$@"
