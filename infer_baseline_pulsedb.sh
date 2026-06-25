#!/usr/bin/env bash
# Score a finetuned baseline on the held-out CalFree 10% test split (single process).
# Uses the SAME finetune split (seed 42, 8:1:1 stratified) so the test set matches
# the finetune run. Emits metrics.json (bpflow.evaluate -> directly comparable).
# Args: <model> <direction> <ckpt> [overrides...]
#   bash infer_baseline_pulsedb.sh nabnet ecg2abp output/baselines/<ft>/checkpoint_best.pth
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?usage: bash infer_baseline_pulsedb.sh <model> <direction> <ckpt>}"; shift
DIRECTION="${1:?usage: bash infer_baseline_pulsedb.sh <model> <direction> <ckpt>}"; shift
CKPT="${1:?pass a finetuned checkpoint path}"; shift
PYTHONPATH="$(pwd)" python -m bpflow_baselines.infer \
  --model "$MODEL" --direction "$DIRECTION" --ckpt "$CKPT" --split test \
  data.finetune=true \
  data.train_npy=rawdata/pulsedb/CalFree_Test_Subset.npy \
  data.finetune_split_mode=stratified "$@"
