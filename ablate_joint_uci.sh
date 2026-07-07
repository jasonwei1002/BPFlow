#!/usr/bin/env bash
# Ablation C1 (cross-modal joint attention): 3-stream JOINT attention  vs  LATE fusion.
#
# LATE = the three streams are encoded independently (block-diagonal self-only
# attention through the joint layers), then their per-token representations are
# concatenated and projected to the target rep, instead of attending jointly.
# This isolates the value of cross-stream attention over simple late fusion.
#
# Runs the flagship ECG+PPG->ABP SPECIALIST on UCI through the full
# pretrain -> finetune -> multi-seed infer pipeline, with a single train/finetune
# seed and 5-seed infer, so it is directly comparable to the joint baseline row
# (UCI / specialist / ecg_ppg2abp). Only model.stream_fusion is flipped.
#
# Baseline (joint) to compare against, if not already run:
#   bash run_uci.sh --tag base --pt "data.tasks=[ecg_ppg2abp] data.task_probs=[1.0]" --ft "data.tasks=[ecg_ppg2abp] data.task_probs=[1.0]"
#
# Usage: bash ablate_joint_uci.sh [--nproc N] [extra run_uci.sh args]
set -euo pipefail
cd "$(dirname "$0")"
bash run_uci.sh --tag abljoint_late \
  --pt "model.stream_fusion=late data.tasks=[ecg_ppg2abp] data.task_probs=[1.0]" \
  --ft "model.stream_fusion=late data.tasks=[ecg_ppg2abp] data.task_probs=[1.0]" \
  "$@"
