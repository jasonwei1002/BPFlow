# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BPFlow generates arterial blood pressure (ABP) waveforms from synchronized ECG + PPG via **conditional flow matching**. It is Meta's WavFlow MMDiT adapted into a 3-stream joint-attention conditional DiT: the ABP latent, ECG, and PPG are three token streams that attend jointly. Trained/evaluated on PulseDB. The `wavflow/` directory is the original upstream reference (gitignored); `bpflow/` is the self-contained project — its WavFlow primitives are **vendored** under `bpflow/model/_vendor/`, so bpflow has no runtime dependency on `wavflow/`.

There is no `README.md`, `pyproject.toml`, or packaging — the project runs as a module (`python -m bpflow.*`). There is no `pytest`/`ruff`/`mypy` config checked in; the real correctness gate is the CPU smoke test below.

## Commands

```bash
# Smoke test — full pipeline on CPU, no GPU (data→patch→forward→loss→backward→sample).
# This IS the test suite: overfits a few real segments, asserts loss collapse + recon improvement.
python -m bpflow.smoke_test                       # uses bpflow/config/smoke.yaml
python -m bpflow.smoke_test --n-steps 600 --n-samples 8
python -m bpflow.smoke_test --config bpflow/config/smoke_calib.yaml   # exercises the K-shot calibration path

# Train (single-node, multi-GPU via torchrun). NPROC=gpu → all visible GPUs.
# gpu.yaml runs a full-CalFree test pass after training (best EMA) and logs test/* to SwanLab.
bash train.sh                                     # all GPUs, bpflow/config/gpu.yaml
NPROC=1 bash train.sh                             # single GPU/CPU
CUDA_VISIBLE_DEVICES=0,1 bash train.sh            # pick GPUs
bash train.sh --config bpflow/config/other.yaml   # override (args pass through to bpflow.train)
python -m bpflow.train --config bpflow/config/gpu.yaml   # no torchrun (single process)

# Inference / evaluation on the CalFree test set. CKPT is REQUIRED (runs live under output/<timestamp>/).
CKPT=output/<timestamp>/checkpoint_best.pth bash infer.sh
NPROC=4 CKPT=... bash infer.sh
python -m bpflow.infer --config bpflow/config/gpu.yaml --ckpt <path> --split test --num -1 --use-ema

# K-shot calibration sweep (needs use_calib + a calib-trained ckpt). Nested K, shared noise per batch;
# K=0 == calibration-free baseline. Writes kshot_sweep.json + a K-vs-error curve.
python -m bpflow.kshot_sweep --config bpflow/config/gpu.yaml --ckpt <best> --use-ema --ks 0,1,3,5,10 --num 2000 --plot

# Pull SwanLab metrics for the latest run (skill: swanlab-fetch)
python ~/.claude/skills/swanlab-fetch/scripts/fetch_swanlab.py --latest

# Force-sync local repo to remote (discards local TRACKED changes; keeps untracked/ignored files)
bash gitpull.sh
```

## Architecture (the data flow that spans files)

```
.npy (N,3,1250)               ch0=ECG, ch1=PPG, ch2=ABP (verified order)
  └─ PulseDBDataset           data/pulsedb_dataset.py  — 80/20 train/val split (seed 42, split_mode); CalFree = test
       ├─ standardize_abp     data/transforms.py       — clip[20,250] THEN global z-score (mean 81.94/std 24.43)
       ├─ patchify ABP        → abp_patches (N=125, P=10)
       ├─ build_cond_patches  → cond_patches (N, 2P) channel-major [ECG(P) | PPG(P)], ECG/PPG recentered −0.5
       ├─ standardize_demo    (use_demo) age/gender/height/weight/bmi from sibling CSV → demo_cont(5) + gender idx
       └─ _build_calib        (use_calib) K same-subject support segs → calib_cond(K,N,2P) + calib_bp(K,2 z-scored SBP/DBP) + calib_mask(K)
  └─ BPFlowModel              model/networks.py        — 3-stream joint DiT
       ├─ DemoEncoder         (use_demo) demo → global_c add-on; zero-init last layer → starts as a no-op
       ├─ CalibrationEncoder  (use_calib) K-shot cuff support → masked-mean → global_c add-on; zero-init → no-op start
       ├─ joint_blocks×8      model/blocks.py BPJointBlock — ABP+ECG+PPG attend jointly, shared RoPE grid
       ├─ fused_blocks×4      latent-only MMDitSingleBlock
       └─ final_layer         → predicts flow (B,N,P)
  └─ FlowMatching             model/flow_matching.py   — train: v-pred loss; infer: euler ODE, 16 steps
  └─ sample_abp               sampling.py              — noise→ODE→unpatchify→destandardize_abp → mmHg (B,L)
  └─ evaluate                 eval/metrics.py          — waveform MAE/RMSE/Pearson + clinical AAMI/BHS on SBP/DBP/MAP
```

Three glue modules tie it together and must stay consistent:
- **`sampling.py`** holds the single definition of `build_flow_matching` / `sample_abp` / `flow_matching_loss`, shared by `trainer.py`, `infer.py`, and `smoke_test.py`. Change the loss/sampling math here, not in three places.
- **`trainer_utils.py`** is the config schema (frozen-default dataclasses → OmegaConf structured) **and** the train helpers (`adjust_learning_rate`, `add_weight_decay`, `pick_device`, `is_main_process`). New config fields go on the dataclasses here first, or `load_config` drops them.
- **Factory/registry**: `model/__init__.py` (`build_model` via `MODEL_REGISTRY`, `@register_model`) and `data/__init__.py` (`build_dataset` via `DATASET_REGISTRY`). Add variants by registering a name; build from `cfg`.

## Config system

OmegaConf structured config with a `_base_:` include chain (resolved in `trainer_utils._merge_base`):
- **`base.yaml`** — architecture, on-disk shapes, normalization constants (ABP + demographics + `bp_*` for calibration), prediction_type. Anything here affects **checkpoint compatibility** (`infer.py` asserts `abp_mean`/`abp_std` match the checkpoint). Defaults to `split_mode: segment`, `use_demo: false`, `use_calib: false`.
- **`gpu.yaml`** (inherits `base.yaml`) — real training (epochs 200, per-GPU batch 128, lr 3e-4, plateau lr-decay + early stop, SwanLab cloud). Flips on **`split_mode: subject`**, **`use_demo: true`**, and **`use_calib: true`** — so real training needs the sibling CSV (see gotchas). Also `val_max_batches: -1` (full DDP-sharded val) and `run_test_after_train: true` (auto CalFree test after training).
- **`smoke.yaml`** (inherits `base.yaml`) — tiny CPU model for the smoke test (not a real model). Inherits segment split + no demo/calib, so the smoke test runs from the `.npy` alone (no CSV).
- **`smoke_calib.yaml`** (inherits `smoke.yaml`) — same tiny model with `use_calib: true` (k_max 4); exercises the calibration path on CPU. Needs the sibling `Train_Subset.csv` (subject_id + sbp/dbp).

## Conventions and gotchas

- **`batch_size` is per-GPU.** Under DDP the global batch is `batch_size × num_gpus` (e.g. 128 × 8 = 1024). `lr` in `gpu.yaml` is already sqrt-scaled for that global batch. Do not auto-divide.
- **Normalization is z-score, not min-max**, and uses **fixed global constants** (never per-sample — that would leak the target). `abp_raw` (unclipped mmHg) is carried through for evaluation; all metrics are computed after `destandardize_abp`.
- **Validation reports MAE, not a loss.** `Trainer.validate()` runs the ODE sampler on the val split and returns waveform **MAE in mmHg**; that MAE drives best-checkpoint, ReduceLROnPlateau, and early stopping. There is no `val/loss` logged (train logs `train/loss`, a v-prediction loss — not directly comparable to val MAE). Validation is **DDP-sharded across all ranks** and merged on rank 0 via `_distributed_eval`, which gathers **per-segment scalars** (BP values + waveform error summaries), not waveforms, so full-set val scales to many GPUs (aggregation is exact because every segment has equal length). `val_max_batches > 0` caps batches **per rank**; `<= 0` means full val (gpu.yaml default).
- **Run dirs are timestamped**: `output/<YYYYMMDD_HHMMSS>/`. rank 0 picks the name and broadcasts it under DDP. Resume reads `checkpoint_latest.pth` from that dir. SwanLab gets no `experiment_name` (auto-generates its run id; `run_name` stored in its config).
- **rank-0-only side effects** under DDP: terminal logging (other ranks at WARNING), tqdm bars, checkpoint saving, and metric logging. **Validation and the post-train test now run on all ranks** (sharded + gathered), but the report is assembled and decisions (`best_val`, `epochs_no_improve`, `lr_scale`, `should_stop`) are made on rank 0 and broadcast via `_sync_val_state`. ⚠️ Every rank must enter `_distributed_eval` together — its `all_gather` is a collective; an early per-rank return deadlocks DDP.
- **`empty_ecg` / `empty_ppg`** are learned CFG null conditions. `networks.py forward()` adds `0.0 * (empty_ecg.sum() + empty_ppg.sum())` so DDP sees them as used even when `label_drop_prob == 0` (avoids "parameters not used in producing loss"). Keep `find_unused_parameters=False`.
- **`torch.compile`** wraps the training-forward path (CUDA only, `use_compile`). Sampling/validation use the **eager** `model_raw`, so variable val batch sizes never trigger recompiles; `drop_last=True` keeps the train batch fixed.
- **EMA** weights are kept on `model_raw`; validation swaps in the current EMA (`_ema_swapped`). `infer.py --use-ema` and the post-train `run_test` instead load EMA weights from disk — `run_test` loads the **best-by-val checkpoint's** EMA (`_load_eval_weights`), not the last.
- **Flow-matching loss combos**: only `(v, v)` handles `min_sigma > 0` exactly; the other combos raise rather than train on a wrong target when `min_sigma != 0`.
- **Data paths live under `rawdata/`** (gitignored), e.g. `rawdata/pulsedb/Train_Subset.npy`. Subject-split and demographics additionally need the **sibling CSV** `<npy_basename>.csv` (e.g. `Train_Subset.csv`), row-aligned to the npy; the dataset raises if it's missing or row counts mismatch. The CSV's `sbp/dbp/map` columns are the **label** — never read them as input.
- **`split_mode`**: `subject` (gpu.yaml, preferred) splits train/val by the CSV's `subject_id` → subject-disjoint, matches the CalFree test setting. `segment` (base default) is a random per-segment split — the same subject lands in both train and val, so val MAE is optimistic. Either way, **CalFree (test) is the honest, subject-disjoint generalization gate.** Changing `split_mode` changes which segments are train vs val, so metrics aren't comparable across modes.
- **Demographics (`model.use_demo`) are a global prior, not a CFG-dropped condition.** Continuous `[age, height, weight, bmi, body_missing]` z-scored with the fixed `demo_*` config constants + a gender `nn.Embedding`, encoded by `DemoEncoder` and added to `global_c`. height/weight/bmi are ~48% jointly missing → `body_missing` flag set, NaN→0 (post-z-score mean). `label_drop_prob` CFG drop never touches demo (`get_empty_conditions` keeps `demo_emb`). The `DemoEncoder` last layer is zero-init so enabling demo starts as a no-op.
- **K-shot cuff calibration (`model.use_calib`) is a global prior like demographics** — the downstream story is "user does K cuff calibrations, then ECG/PPG-only inference". Each query carries K same-subject support segments (`_build_calib`): their ECG/PPG `cond_patches` + the cuff `[SBP, DBP]` z-scored by the fixed `bp_*` constants — but **NOT** their ABP waveform (a cuff gives only scalars), and **NEVER** the query's own SBP/DBP (that is the eval target → leakage). `CalibrationEncoder` masked-mean-pools the K into a `global_c` add-on (zero-init → no-op start; K=0 / all-masked → calibration-free). Train randomizes K∈[0, `calib_k_max`] per sample; val/test fix `calib_eval_k`. Needs `subject_id` from the CSV for **every** split (test included). Leakage guards: support excludes the query index and stays within its subject. Full K-shot curve via `bpflow.kshot_sweep`.
- **`run_test_after_train`** (gpu.yaml true): after training finishes (normal end or early stop — NOT a `max_steps` interrupt), `Trainer.run_test()` evaluates the **best-by-val EMA** model on the full CalFree test set (DDP-sharded, shared `_distributed_eval`), logs `test/*` to SwanLab, and writes `output/<ts>/test_metrics.json`. It runs inside `train()` before `close()`, so it lands in the same SwanLab run. `test_max_segments > 0` caps it. In-train test reports at the configured `calib_eval_k`; the K=0 baseline comes from `kshot_sweep`.

## Git

Conventional Commits. `git commit`/`push` are only run on explicit user authorization. `gitpull.sh` targets a gh-proxy mirror of `github.com/jasonwei1002/BPFlow`. Never commit `rawdata/`, `wavflow/`, `output/`, checkpoints, or secrets (all gitignored).
