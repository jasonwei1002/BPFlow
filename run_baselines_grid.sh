#!/usr/bin/env bash
# Sweep the full baseline grid: {6 models} x {5 directions}, each running the
# bpflow protocol pretrain -> finetune -> infer. Deterministic run dirs (via
# training.run_name), resumable (--skip-existing), fault-tolerant (one failed
# cell does not abort the sweep). Single seed.
#
# Run dirs:   output/baselines/<model>_<direction>_pre   (pretrain)
#             output/baselines/<model>_<direction>_ft    (finetune)
#             output/baselines/<model>_<direction>_ft/infer_<direction>/metrics.json
# Logs:       output/baselines/logs/<model>_<direction>_<stage>.log
#
# Usage:
#   bash run_baselines_grid.sh                                  # full grid, all GPUs
#   bash run_baselines_grid.sh --nproc 4 --skip-existing        # 4 GPUs, resume
#   bash run_baselines_grid.sh --models "nabnet patchtst" --directions "ecg2abp ppg2abp"
#   bash run_baselines_grid.sh --stages "infer"                 # only (re)score existing finetunes
#   bash run_baselines_grid.sh --dry-run                        # print the plan, run nothing
#   bash run_baselines_grid.sh -- training.batch_size=64        # extra overrides after --
# no `set -u`: empty arrays (EXTRA/FAILED/...) trip it on macOS's bash 3.2
set -o pipefail
cd "$(dirname "$0")"

MODELS="mdvisco nabnet patchtst ppg2abp p2e_wgan wavenet"
DIRECTIONS="ecg2abp ppg2abp ecg_ppg2abp ecg2ppg ppg2ecg"
STAGES="pretrain finetune infer"
NPROC=gpu
SKIP=0
DRY=0
OUT="output/baselines"
EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --models) MODELS="${2:?}"; shift 2;;
    --directions) DIRECTIONS="${2:?}"; shift 2;;
    --stages) STAGES="${2:?}"; shift 2;;
    --nproc) NPROC="${2:?}"; shift 2;;
    --out) OUT="${2:?}"; shift 2;;
    --skip-existing) SKIP=1; shift;;
    --dry-run) DRY=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

LOGS="$OUT/logs"
mkdir -p "$LOGS"
has_stage() { echo " $STAGES " | grep -q " $1 "; }
ckpt_of() { [ -f "$OUT/$1/checkpoint_best.pth" ] && echo "$OUT/$1/checkpoint_best.pth" || echo "$OUT/$1/checkpoint_latest.pth"; }

declare -a DONE FAILED SKIPPED
run_cell() {  # model direction
  local m="$1" d="$2"
  local pre="${m}_${d}_pre" ft="${m}_${d}_ft"
  echo "==================== $m / $d ===================="

  if has_stage pretrain; then
    if [ "$SKIP" = 1 ] && [ -f "$OUT/$pre/checkpoint_best.pth" ]; then
      echo "  [pretrain] skip (exists)"; SKIPPED+=("$m/$d:pretrain")
    elif [ "$DRY" = 1 ]; then
      echo "  [pretrain] DRY: train_baseline_pulsedb.sh $m $d --nproc $NPROC training.run_name=$pre"
    else
      echo "  [pretrain] -> $LOGS/$pre.log"
      if ! bash train_baseline_pulsedb.sh "$m" "$d" --nproc "$NPROC" \
           "${EXTRA[@]}" training.output_dir="$OUT" training.run_name="$pre" >"$LOGS/$pre.log" 2>&1; then
        echo "  [pretrain] FAILED (see $LOGS/$pre.log)"; FAILED+=("$m/$d:pretrain"); return
      fi
    fi
  fi

  if has_stage finetune; then
    local init; init="$(ckpt_of "$pre")"
    if [ "$SKIP" = 1 ] && [ -f "$OUT/$ft/checkpoint_best.pth" ]; then
      echo "  [finetune] skip (exists)"; SKIPPED+=("$m/$d:finetune")
    elif [ "$DRY" = 1 ]; then
      echo "  [finetune] DRY: finetune_baseline_pulsedb.sh $m $d $init --nproc $NPROC training.run_name=$ft"
    elif [ ! -f "$init" ]; then
      echo "  [finetune] FAILED (no pretrain ckpt $init)"; FAILED+=("$m/$d:finetune"); return
    else
      echo "  [finetune] -> $LOGS/$ft.log"
      if ! bash finetune_baseline_pulsedb.sh "$m" "$d" "$init" --nproc "$NPROC" \
           "${EXTRA[@]}" training.output_dir="$OUT" training.run_name="$ft" >"$LOGS/$ft.log" 2>&1; then
        echo "  [finetune] FAILED (see $LOGS/$ft.log)"; FAILED+=("$m/$d:finetune"); return
      fi
    fi
  fi

  if has_stage infer; then
    local fck; fck="$(ckpt_of "$ft")"
    if [ "$SKIP" = 1 ] && [ -f "$OUT/$ft/infer_$d/metrics.json" ]; then
      echo "  [infer] skip (exists)"; SKIPPED+=("$m/$d:infer")
    elif [ "$DRY" = 1 ]; then
      echo "  [infer] DRY: infer_baseline_pulsedb.sh $m $d $fck"
    elif [ ! -f "$fck" ]; then
      echo "  [infer] FAILED (no finetune ckpt $fck)"; FAILED+=("$m/$d:infer"); return
    else
      echo "  [infer] -> $LOGS/${ft}_infer.log"
      if ! bash infer_baseline_pulsedb.sh "$m" "$d" "$fck" \
           "${EXTRA[@]}" >"$LOGS/${ft}_infer.log" 2>&1; then
        echo "  [infer] FAILED (see $LOGS/${ft}_infer.log)"; FAILED+=("$m/$d:infer"); return
      fi
    fi
  fi
  DONE+=("$m/$d")
}

for m in $MODELS; do
  for d in $DIRECTIONS; do
    run_cell "$m" "$d"
  done
done

echo ""
echo "==================== SWEEP SUMMARY ===================="
echo "completed cells : ${#DONE[@]}  [${DONE[*]:-}]"
echo "skipped stages  : ${#SKIPPED[@]}"
echo "failed stages   : ${#FAILED[@]}  [${FAILED[*]:-}]"
if [ "$DRY" = 0 ]; then
  echo ""
  echo "Aggregate results into a comparison table with:"
  echo "  PYTHONPATH=. python -m bpflow_baselines.summarize"
fi
[ "${#FAILED[@]}" -eq 0 ]
