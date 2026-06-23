#!/usr/bin/env bash
# Optuna search over the unified model's modality_dropout_probs.
# Thin wrapper around tune_modality_probs.py — minimises the best val/MAE (the MEAN
# waveform MAE across ecg_ppg/ecg/ppg), so it optimises "all three directions good".
# Each trial = one full unified pretrain via train_pulsedb.sh (DDP/torchrun, all GPUs);
# trials run SEQUENTIALLY. ASHA prunes a losing prob mix after a few epochs.
#
# PREREQUISITE: run the ecg/ppg specialists FIRST (gpu_ecg.yaml / gpu_ppg.yaml) to
# learn each direction's ceiling. If ppg is info-limited (its specialist also fails
# AAMI), no prob mix recovers it — don't run this search blind.
#
# Use (all args pass through to tune_modality_probs.py):
#   bash tune.sh                                   # defaults: 8 trials, 150 epochs/trial, all GPUs
#   bash tune.sh --n-trials 5 --search-epochs 120  # cheaper: fewer trials, lower fidelity
#   bash tune.sh --nproc 4                          # 4 GPUs per trial
#   bash tune.sh --storage sqlite:///my_study.db --study-name run2
# Resumable: re-run the same command; the SQLite study continues.
# Cost: ~15h/trial at 150 epochs on 8 GPUs (pruned trials die in ~2-4h). Dial down
# --search-epochs / --n-trials to bound it; retrain the winner to full convergence.
set -euo pipefail
cd "$(dirname "$0")"

# Optuna is an optional dep, only needed for the search.
if ! python -c "import optuna" 2>/dev/null; then
  echo "optuna not installed. Install it first:  pip install optuna" >&2
  exit 1
fi

exec python tune_modality_probs.py "$@"
