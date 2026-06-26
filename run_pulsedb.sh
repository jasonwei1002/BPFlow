#!/usr/bin/env bash
# One-shot PulseDB pipeline: pretrain -> finetune -> multi-seed infer, chained so
# each stage auto-consumes the previous stage's best checkpoint (no timestamp guessing).
#
# Run dirs carry the active direction slug (uni5 for the unified model, or a single
# direction like ecg2abp) so stages and SwanLab runs are easy to tell apart:
#   Stage 1  pretrain (pulsedb.yaml) on Train_Subset            -> output/pdb_pt_<ts>_<dir>/
#            (run_name pinned so we know exactly where best lands)
#   Stage 2  finetune (pulsedb_finetune.yaml), init from stage-1 checkpoint_best.pth, on the
#            CalFree test_npy 8:1:1 split (weights only)     -> output/pdb_ft_<ts>_<dir>/
#   Stage 3  infer (delegates to infer_pulsedb.sh): multi-seed score of stage-2
#            checkpoint_best.pth on the held-out test split -> output/pdb_infer_<ts>/
#            (summary.json = per-task per-metric mean+/-std)
#
# Options:
#   --nproc <gpu|N>     processes for ALL stages ('gpu' = all visible GPUs, default)
#   --pt "<overrides>"  extra dotted key=value overrides for the PRETRAIN stage only
#   --ft "<overrides>"  extra dotted key=value overrides for the FINETUNE stage only
#   --in "<args>"       extra args forwarded to infer_pulsedb.sh (e.g. "--task ppg2abp --num 100")
#   --skip-infer        stop after finetune (produce weights only, no scoring)
# Env (consumed by the infer stage via infer_pulsedb.sh):
#   SEEDS="0 1 2 3 4"   space-separated seeds for the multi-seed eval
#   OUTBASE=<dir>       infer output root (default output/pdb_infer_<ts>)
# Examples:
#   bash run_pulsedb.sh
#   bash run_pulsedb.sh --nproc 4
#   bash run_pulsedb.sh --pt "training.lr=3e-4" --ft "data.finetune_train_ratio=0.25"
#   SEEDS="0 1 2" bash run_pulsedb.sh --in "--task ecg_ppg2abp"
#
# Note: this is the happy-path orchestrator. To resume an interrupted pretrain, use
# the standalone scripts: bash train_pulsedb.sh --resume output/<dir>, then
# bash finetune_pulsedb.sh <dir>/checkpoint_best.pth, then bash infer_pulsedb.sh <ft>/checkpoint_best.pth.
set -euo pipefail
cd "$(dirname "$0")"

NPROC=gpu
SKIP_INFER=0
PT_ARGS=()
FT_ARGS=()
IN_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="${2:?--nproc needs a value (gpu or an integer)}"; shift 2;;
    --nproc=*) NPROC="${1#*=}"; shift;;
    --pt) read -r -a PT_ARGS <<< "${2-}"; shift 2;;
    --ft) read -r -a FT_ARGS <<< "${2-}"; shift 2;;
    --in) read -r -a IN_ARGS <<< "${2-}"; shift 2;;
    --skip-infer) SKIP_INFER=1; shift;;
    *) echo "unknown arg: $1 (use --nproc / --pt / --ft / --in / --skip-infer)" >&2; exit 2;;
  esac
done

# Resolve each stage's active direction(s) into a short slug (uni5 / ecg2abp/...)
# from its config + overrides, so the pinned run dirs and SwanLab runs are
# self-describing. Same tasks_slug() the trainer uses for its auto run names.
dir_tag() {  # usage: dir_tag <config.yaml> [dotted key=value overrides...]
  python - "$@" <<'PY'
import sys
from bpflow.train import _overrides_from_extra
from bpflow.trainer_utils import load_config
from bpflow.data import tasks_slug
cfg = load_config(sys.argv[1], _overrides_from_extra(sys.argv[2:]) or None)
print(tasks_slug(cfg.data.tasks, cfg.data.task_probs))
PY
}

# Letter-prefixed names so OmegaConf's dotlist parser keeps them as strings; a
# pure-numeric "<ts>" would be read as an int (underscores = digit separators)
# and the dir we compute here would not match the trainer's actual exp_dir.
TS="$(date +%Y%m%d_%H%M%S)"
PT_TAG="$(dir_tag bpflow/config/pulsedb.yaml ${PT_ARGS[@]+"${PT_ARGS[@]}"})"
FT_TAG="$(dir_tag bpflow/config/pulsedb_finetune.yaml ${FT_ARGS[@]+"${FT_ARGS[@]}"})"
PRETRAIN_NAME="pdb_pt_${TS}_${PT_TAG}"
FINETUNE_NAME="pdb_ft_${TS}_${FT_TAG}"
PRETRAIN_DIR="output/$PRETRAIN_NAME"
FINETUNE_DIR="output/$FINETUNE_NAME"
# Shared SwanLab group: pretrain + finetune + infer log 3 separate runs grouped
# under this name on the dashboard.
GROUP="pdb_${TS}_${PT_TAG}"

echo "[1/3] pretrain (pulsedb.yaml) -> $PRETRAIN_DIR  (SwanLab group $GROUP)"
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.train \
  --config bpflow/config/pulsedb.yaml \
  training.run_name="$PRETRAIN_NAME" training.swanlab_group="$GROUP" \
  ${PT_ARGS[@]+"${PT_ARGS[@]}"}

PT_BEST="$PRETRAIN_DIR/checkpoint_best.pth"
if [ ! -f "$PT_BEST" ]; then
  echo "ERROR: pretrain finished but produced no $PT_BEST -- aborting before finetune." >&2
  exit 1
fi

echo "[2/3] finetune (pulsedb_finetune.yaml) <- $PT_BEST  ->  $FINETUNE_DIR"
torchrun --standalone --nproc_per_node="$NPROC" -m bpflow.train \
  --config bpflow/config/pulsedb_finetune.yaml \
  --init-ckpt "$PT_BEST" training.run_name="$FINETUNE_NAME" training.swanlab_group="$GROUP" \
  ${FT_ARGS[@]+"${FT_ARGS[@]}"}

FT_BEST="$FINETUNE_DIR/checkpoint_best.pth"
if [ ! -f "$FT_BEST" ]; then
  echo "ERROR: finetune finished but produced no $FT_BEST -- nothing to score." >&2
  exit 1
fi

if [ "$SKIP_INFER" -eq 1 ]; then
  echo "DONE (--skip-infer)."
  echo "  pretrain weights : $PT_BEST"
  echo "  finetune weights : $FT_BEST"
  echo "  score later with : bash infer_pulsedb.sh $FT_BEST"
  exit 0
fi

# Stage 3 delegates to infer_pulsedb.sh (reuses its multi-seed loop + aggregation).
# Pair the infer output dir with this run's <ts> unless the caller set OUTBASE.
# SEEDS (if set in the environment) propagates to infer_pulsedb.sh automatically.
export OUTBASE="${OUTBASE:-output/pdb_infer_$TS}"
echo "[3/3] infer (infer_pulsedb.sh) <- $FT_BEST  ->  $OUTBASE"
bash infer_pulsedb.sh "$FT_BEST" --nproc "$NPROC" ${IN_ARGS[@]+"${IN_ARGS[@]}"}

# Log the held-out test metrics as a 3rd SwanLab run in this pipeline's group.
# Online only; no-op if swanlab missing / offline. Never fails the pipeline.
python -m bpflow.infer_swanlab --summary "$OUTBASE/summary.json" --group "$GROUP" --ckpt "$FT_BEST" || true

echo "DONE."
echo "  pretrain weights : $PT_BEST"
echo "  finetune weights : $FT_BEST"
echo "  infer summary    : $OUTBASE/summary.json  (test/* also a SwanLab run in group $GROUP)"
