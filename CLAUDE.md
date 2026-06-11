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

# Pretrain (single-node, multi-GPU via torchrun). --nproc <gpu|N> picks processes
# ('gpu' = all visible GPUs, default). Pretraining does NOT touch CalFree (no
# post-train test); CalFree is reserved for finetune.
bash train.sh                                     # all GPUs, bpflow/config/gpu.yaml
bash train.sh --nproc 1                           # single GPU/CPU
CUDA_VISIBLE_DEVICES=0,1 bash train.sh --nproc 2  # pick GPUs
bash train.sh --config bpflow/config/other.yaml   # override (args pass through to bpflow.train)
python -m bpflow.train --config bpflow/config/gpu.yaml   # no torchrun (single process)

# Finetune a pretrained model on the CalFree domain. CalFree (test_npy) is split 8:1:1
# (per-segment, fixed seed) into train/val/test; finetunes on 80%, then auto-tests on the
# held-out 10%. Args: <checkpoint> [--nproc N] [extra args]; new output/<ts>/.
bash finetune.sh output/<pretrain_ts>/checkpoint_best.pth            # all GPUs
bash finetune.sh output/<pretrain_ts>/checkpoint_best.pth --nproc 4  # 4 GPUs
python -m bpflow.train --config bpflow/config/finetune.yaml --init-ckpt <pretrained>

# Inference / evaluation on the CalFree test set. Args: <checkpoint> [--nproc N] [extra args].
bash infer.sh output/<timestamp>/checkpoint_best.pth                 # all GPUs
bash infer.sh output/<timestamp>/checkpoint_best.pth --nproc 4       # 4 GPUs
python -m bpflow.infer --config bpflow/config/gpu.yaml --ckpt <path> --split test --num -1 --use-ema

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
       └─ standardize_demo    (use_demo) age/gender/height/weight/bmi from sibling CSV → demo_cont(5) + gender idx
  └─ BPFlowModel              model/networks.py        — 3-stream joint DiT
       ├─ DemoEncoder         (use_demo) demo → global_c add-on; zero-init last layer → starts as a no-op
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
- **`base.yaml`** — architecture, on-disk shapes, normalization constants (ABP + demographics), prediction_type. Anything here affects **checkpoint compatibility** (`infer.py` asserts `abp_mean`/`abp_std` match the checkpoint). Defaults to `split_mode: segment`, `use_demo: false`.
- **`gpu.yaml`** (inherits `base.yaml`) — real **pretraining** (epochs 200, per-GPU batch 128, lr 3e-4, plateau lr-decay + early stop, SwanLab cloud). Uses `split_mode: segment` (random per-segment train/val) and flips on **`use_demo: true`** — so real training needs the sibling CSV for demographics (see gotchas). Also `val_max_batches: -1` (full DDP-sharded val) and **`run_test_after_train: false`** — pretraining never touches CalFree (reserved for finetune).
- **`finetune.yaml`** (inherits `gpu.yaml`) — domain adaptation on CalFree. Same architecture + hyperparams as gpu.yaml, but **`data.finetune: true`** splits the CalFree `test_npy` 8:1:1 (per-segment, fixed seed) into train/val/test, and **`run_test_after_train: true`** evaluates the held-out 10% test split at the end. Weights are initialized from a pretrained checkpoint via `--init-ckpt` (→ `training.init_from_ckpt`); optimizer/epoch/step reset.
- **`smoke.yaml`** (inherits `base.yaml`) — tiny CPU model for the smoke test (not a real model). Inherits segment split + no demo, so the smoke test runs from the `.npy` alone (no CSV).

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
- **`split_mode`**: `segment` (base + gpu.yaml default) is a random per-segment train/val split — the same subject lands in both train and val, so val MAE is optimistic. `subject` (optional) splits train/val by the CSV's `subject_id` → subject-disjoint, matches the CalFree test setting (needs the sibling CSV for the train/val split). Either way, **CalFree (test) is the honest, subject-disjoint generalization gate.** Changing `split_mode` changes which segments are train vs val, so metrics aren't comparable across modes.
- **Demographics (`model.use_demo`) are a global prior, not a CFG-dropped condition.** Continuous `[age, height, weight, bmi, body_missing]` z-scored with the fixed `demo_*` config constants + a gender `nn.Embedding`, encoded by `DemoEncoder` and added to `global_c`. height/weight/bmi are ~48% jointly missing → `body_missing` flag set, NaN→0 (post-z-score mean). `label_drop_prob` CFG drop never touches demo (`get_empty_conditions` keeps `demo_emb`). The `DemoEncoder` last layer is zero-init so enabling demo starts as a no-op.
- **`run_test_after_train`** (gpu.yaml **false**, finetune.yaml **true**): after training finishes (normal end or early stop — NOT a `max_steps` interrupt), `Trainer.run_test()` evaluates the **best-by-val EMA** model on the `test` split (DDP-sharded, shared `_distributed_eval`), logs `test/*` to SwanLab, and writes `output/<ts>/test_metrics.json`. It runs inside `train()` before `close()`, so it lands in the same SwanLab run. `test_max_segments > 0` caps it. For finetune the `test` split is the held-out 10% of CalFree; pretraining leaves it off so CalFree is untouched.
- **Finetune flow (`data.finetune: true`)** repurposes the CalFree `test_npy`: a fixed-seed **per-segment** 8:1:1 split (`finetune_val_fraction`/`finetune_test_fraction`, default 0.1/0.1) makes train/val/**test** all read CalFree, non-overlapping. `split_mode` is ignored (no subject split). The pretrained model is loaded via **`--init-ckpt`** → `training.init_from_ckpt` (`Trainer._maybe_init_from_ckpt`): it copies `model` (+ `model_ema`) weights only, then trains fresh (optimizer/epoch/step reset). Skipped when resuming an interrupted run. ⚠️ The split is per-segment, so the same subject can appear in both finetune-train and the final test → that test number is optimistic, not subject-disjoint.

## Git

Conventional Commits. `git commit`/`push` are only run on explicit user authorization. `gitpull.sh` targets a gh-proxy mirror of `github.com/jasonwei1002/BPFlow`. Never commit `rawdata/`, `wavflow/`, `output/`, checkpoints, or secrets (all gitignored).
