#!/usr/bin/env bash
# Finetune a UCI-pretrained model on the UCI test fold (config uci_finetune.yaml).
# The UCI test fold is split 8:1:1 (per-segment, segment mode -- UCI has no
# subject_id) into train/val/test; this finetunes on the 80% train (val for early
# stop) and produces weights only -- run_test_after_train is OFF. Score the
# held-out 10% test split separately with: bash infer_uci.sh <ckpt>.
# Runs live under a NEW output/<timestamp>/.
# Args: <pretrained_checkpoint> [--nproc <gpu|N>] [extra args...]   (--nproc default 'gpu')
#   all visible GPUs (default):  bash finetune_uci.sh output/<uci_pretrain_ts>/checkpoint_best.pth
#   N GPUs:                      bash finetune_uci.sh output/<uci_pretrain_ts>/checkpoint_best.pth --nproc 4
#   data-efficiency sweep:       bash finetune_uci.sh <path> data.finetune_train_ratio=0.25
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="bpflow/config/uci_finetune.yaml"
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
CKPT="${1:?pass a pretrained checkpoint: bash finetune_uci.sh <path> [--nproc N]}"; shift
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.train \
  --config "$CONFIG" --init-ckpt "$CKPT" "$@"
