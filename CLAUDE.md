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
# (per-segment, fixed seed) into train/val/test; finetunes on 80% (val for early stop)
# and produces weights only — it does NOT auto-test. Score the held-out 10% test split
# separately with infer.sh below. Args: <checkpoint> [--nproc N] [extra args]; new output/<ts>/.
bash finetune.sh output/<pretrain_ts>/checkpoint_best.pth            # all GPUs
bash finetune.sh output/<pretrain_ts>/checkpoint_best.pth --nproc 4  # 4 GPUs
python -m bpflow.train --config bpflow/config/finetune.yaml --init-ckpt <pretrained>

# Evaluate a finetuned checkpoint on the CalFree held-out 10% test split. infer.sh
# loads finetune.yaml so the split matches finetune's fixed-seed 8:1:1 — this is the
# decoupled eval step finetune.sh no longer auto-runs. Args: <checkpoint> [--nproc N] [extra args].
bash infer.sh output/<finetune_ts>/checkpoint_best.pth              # all GPUs
bash infer.sh output/<finetune_ts>/checkpoint_best.pth --nproc 4    # 4 GPUs
python -m bpflow.infer --config bpflow/config/finetune.yaml --ckpt <path> --split test --num -1 --use-ema

# Pull SwanLab metrics for the latest run (project "bpflow"; skill: swanlab-skill)
swanlab api run list badwoman/bpflow --page_size 10        # newest run is first; copy its run id
swanlab api run summary badwoman/bpflow/<run_id>           # latest value + min/max of every metric

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
- **`gpu.yaml`** (inherits `base.yaml`) — real **pretraining** (epochs 200, per-GPU batch 128, lr 3e-4, plateau lr-decay + early stop, SwanLab cloud). Uses `split_mode: segment` (random per-segment train/val), `use_demo: false` (pure ECG/PPG → ABP this round). Also `val_eval_fraction: 0.1` (validate on a representative stride-sampled 10% of val) and **`run_test_after_train: false`** — pretraining never touches CalFree (reserved for finetune).
- **`finetune.yaml`** (inherits `gpu.yaml`) — domain adaptation on CalFree. Spells out its own `training` block (tune finetune hyperparams independently). **`data.finetune: true`** splits the CalFree `test_npy` 8:1:1 (per-segment, fixed seed) into train/val/test. Training produces weights only — **`run_test_after_train: false`** (eval is decoupled): the held-out 10% test split is scored separately via `bash infer.sh <ckpt>`, which also loads `finetune.yaml`. Weights init from a pretrained checkpoint via `--init-ckpt` (→ `training.init_from_ckpt`); optimizer/epoch/step reset.
- **`smoke.yaml`** (inherits `base.yaml`) — tiny CPU model for the smoke test (not a real model). Inherits segment split + no demo, so the smoke test runs from the `.npy` alone (no CSV).

## Conventions and gotchas

- **`batch_size` is per-GPU.** Under DDP the global batch is `batch_size × num_gpus` (e.g. 128 × 8 = 1024). `lr` in `gpu.yaml` is already sqrt-scaled for that global batch. Do not auto-divide.
- **Normalization is z-score, not min-max**, and uses **fixed global constants** (never per-sample — that would leak the target). `abp_raw` (unclipped mmHg) is carried through for evaluation; all metrics are computed after `destandardize_abp`.
- **Validation reports MAE, not a loss** (for decisions). `Trainer.validate()` runs the ODE sampler on the val split and returns waveform **MAE in mmHg**; that MAE — and only it — drives best-checkpoint, ReduceLROnPlateau, and early stopping. Validation is **DDP-sharded across all ranks** and merged on rank 0 via `_distributed_eval`, which gathers **per-segment scalars** (BP values + waveform error summaries), not waveforms, so full-set val scales to many GPUs (aggregation is exact because every segment has equal length). **`val_eval_fraction`** (≤1) stride-subsamples val to a representative fraction (covers all subjects, fixed across epochs → MAE trend comparable) **before** the DDP shard — the single val-subsampling knob (gpu.yaml `0.1`, finetune.yaml `1.0`).
- **Per-modality train/val losses are monitor-only.** The flow-matching (v-prediction) loss is logged separately for each `cond_mask` (`ecg_ppg`/`ecg`/`ppg`), to watch how a unified (`modality_dropout`) model fits each direction. Per batch the **same noise & timesteps** are reused across the three modalities → a paired comparison.
  - **Val** (`val/loss_{...}` via `_val_modality_losses`): on the val split, fixed per-batch seed + unshuffled loader → also **epoch-comparable**. Runs on **all ranks** (one `all_reduce`) inside the same EMA swap as the sampler, **before** the rank-0 early return. Only the **trained** directions are evaluated (`trained_modalities`: specialist → its 1 `cond_modality`; dropout → every modality with prob>0), so a specialist run logs just one `val/loss_*` and doesn't pay 3× for uninformative directions. `val_loss_max_batches`: `-1` = full val (gpu/finetune), `0` = off, `N>0` = cap per rank (base default 50). Absolute value depends on world_size, like `val_max_batches`.
  - **Train** (`train/loss_{...}` + `train/loss_epoch`, logged at **epoch end**): the **full-epoch** train loss, decomposed **for free** from the actual training loss — `flow_matching_loss(..., per_sample=True)` returns the `(B,)` per-sample loss, and `_train_step` buckets each sample by its live `modality_dropout` mask into epoch accumulators (`_accumulate_modality_loss`); `_log_epoch_loss` all-reduces across ranks and logs the mean. No extra forward. So `train/loss_{m}` = mean train loss of samples *actually assigned* modality m this epoch (a specialist run fills only its one modality), distinct from `val/loss_{m}` (every val sample under a *fixed* mask). `train/loss` (per-step instantaneous) is still logged every `log_freq`.
  - Neither influences best/plateau/early-stop (those still use val MAE).
- **Run dirs are timestamped**: `output/<YYYYMMDD_HHMMSS>/`. rank 0 picks the name and broadcasts it under DDP. **Resume is opt-in via `--resume <run_dir>`** (→ `training.resume_dir`): it reuses that existing dir as `exp_dir` (no new timestamp), loads its `checkpoint_latest.pth`, and **continues the same SwanLab run** (the run id is saved in the checkpoint as `swanlab_run_id` and replayed via `swanlab.init(id=..., resume="allow")`, online mode only). Without `--resume`, every launch is a fresh timestamped dir + fresh SwanLab run — a brand-new dir has no `checkpoint_latest.pth`, so resume never fires by accident. SwanLab otherwise gets no `experiment_name` (auto-generates its run id; `run_name` stored in its config).
- **rank-0-only side effects** under DDP: terminal logging (other ranks at WARNING), tqdm bars, checkpoint saving, and metric logging. **Validation and the post-train test now run on all ranks** (sharded + gathered), but the report is assembled and decisions (`best_val`, `epochs_no_improve`, `lr_scale`, `should_stop`) are made on rank 0 and broadcast via `_sync_val_state`. ⚠️ Every rank must enter `_distributed_eval` together — its `all_gather` is a collective; an early per-rank return deadlocks DDP.
- **Purely conditional** (no classifier-free guidance): the model always conditions on the real ECG/PPG — there are no learned null conditions and no `cfg_strength`/`label_drop_prob`. Every parameter is used in each forward, so DDP runs with `find_unused_parameters=False`.
- **Modality masking = `cond_mask` + learned null tokens (NOT zeroing).** The dataset always emits the full `(N, 2P)` ECG+PPG `cond_patches` plus a per-sample `cond_mask` `(2,)` = `[ecg_present, ppg_present]`. In the model, `_apply_null` replaces an absent stream's token embedding with a learned `null_ecg`/`null_ppg` parameter (`e*ecg_seq + (1-e)*null_ecg`), so "absent" is a dedicated, unambiguous code, not a flat-signal collision. The formula references the null params even when a stream is present (zero coefficient), so they always stay in the autograd graph → DDP `find_unused_parameters=False` still holds. `pooled` (global cond) is computed AFTER null replacement. `null_*` are new params: older checkpoints lack them → `load_model_state` tolerates them missing (kept at init), while still flagging real mismatches.
- **`data.cond_modality`** (input "direction"): `ecg_ppg` (default) | `ecg` | `ppg`. Maps to a fixed `cond_mask` (`ecg`→`[1,0]`, `ppg`→`[0,1]`); the model nulls the absent stream. Architecture/params/checkpoint shape are **identical** across directions (every model has the null params). Train a specialist per direction; train and infer must use the same value — baked into the saved `config`, and `infer.py` asserts `ckpt.config.data.cond_modality == cfg` (old checkpoints lack the field → `ecg_ppg`).
- **`data.modality_dropout`** (one **unified** model for all 3 directions): when true, each **train** sample randomly picks `ecg_ppg`/`ecg`/`ppg` → its `cond_mask` (per-sample, probs `modality_dropout_probs` = `[ecg_ppg, ecg, ppg]`, default `[0.34, 0.33, 0.33]` ~uniform); **val/test/infer keep the fixed `cond_modality`** so metrics stay comparable. Only the per-sample *choice* is randomized → modality-agnostic model instead of a specialist. The **trained set** (which modalities a checkpoint actually saw) is the shared `trained_modalities()` (`data/__init__.py`): dropout → every modality with **prob > 0** (so degenerate `[1,0,0]` restricts back to `ecg_ppg`, `[0.5,0.5,0]` allows only `{ecg_ppg, ecg}`); specialist → its single fixed `cond_modality`; old checkpoints → `{ecg_ppg}`. Both the `infer.py` guard (which infer direction is valid) and the per-modality val loss (which directions to log) use it; the modality↔prob order is `MODALITY_ORDER`. The per-sample dropout draw is seeded per worker from `cfg.training.seed` (via the train loader's `worker_init_fn=_seed_worker`), so it's reproducible at a fixed `num_workers` and tied to the run seed. No BatchNorm (AdaLN), so per-sample mixing within a batch is safe. Off by default → the per-direction specialist setup above.
- **`torch.compile`** wraps the training-forward path (CUDA only, `use_compile`). Sampling/validation use the **eager** `model_raw`, so variable val batch sizes never trigger recompiles; `drop_last=True` keeps the train batch fixed.
- **EMA** weights are kept on `model_raw`; validation swaps in the current EMA (`_ema_swapped`). `infer.py --use-ema` and the post-train `run_test` instead load EMA weights from disk — `run_test` loads the **best-by-val checkpoint's** EMA (`_load_eval_weights`), not the last.
- **Flow-matching loss combos**: only `(v, v)` handles `min_sigma > 0` exactly; the other combos raise rather than train on a wrong target when `min_sigma != 0`.
- **Data paths live under `rawdata/`** (gitignored), e.g. `rawdata/pulsedb/Train_Subset.npy`. Subject-split and demographics additionally need the **sibling CSV** `<npy_basename>.csv` (e.g. `Train_Subset.csv`), row-aligned to the npy; the dataset raises if it's missing or row counts mismatch. The CSV's `sbp/dbp/map` columns are the **label** — never read them as input.
- **`split_mode`**: `segment` (base + gpu.yaml default) is a random per-segment train/val split — the same subject lands in both train and val, so val MAE is optimistic. `subject` (optional) splits train/val by the CSV's `subject_id` → subject-disjoint, matches the CalFree test setting (needs the sibling CSV for the train/val split). Either way, **CalFree (test) is the honest, subject-disjoint generalization gate.** Changing `split_mode` changes which segments are train vs val, so metrics aren't comparable across modes.
- **Demographics (`model.use_demo`) are a global prior.** Continuous `[age, height, weight, bmi, body_missing]` z-scored with the fixed `demo_*` config constants + a gender `nn.Embedding`, encoded by `DemoEncoder` and added to `global_c`. height/weight/bmi are ~48% jointly missing → `body_missing` flag set, NaN→0 (post-z-score mean). The `DemoEncoder` last layer is zero-init so enabling demo starts as a no-op.
- **`run_test_after_train`** (**false** in gpu.yaml AND finetune.yaml — eval is decoupled from training): when **enabled**, after training finishes (normal end or early stop — NOT a `max_steps` interrupt), `Trainer.run_test()` evaluates the **best-by-val EMA** model on the `test` split (DDP-sharded, shared `_distributed_eval`), logs `test/*` to SwanLab, and writes `output/<ts>/test_metrics.json` inside `train()` before `close()` (same SwanLab run); `test_max_segments > 0` caps it. Both configs now leave it **off**: finetune produces weights only and the held-out 10% test is scored separately via `bash infer.sh <ckpt>`, and pretraining never touches CalFree.
- **Finetune flow (`data.finetune: true`)** repurposes the CalFree `test_npy`: a fixed-seed **per-segment** 8:1:1 split (`finetune_val_fraction`/`finetune_test_fraction`, default 0.1/0.1) makes train/val/**test** all read CalFree, non-overlapping. `split_mode` is ignored (no subject split). The pretrained model is loaded via **`--init-ckpt`** → `training.init_from_ckpt` (`Trainer._maybe_init_from_ckpt`): it copies `model` (+ `model_ema`) weights only, then trains fresh (optimizer/epoch/step reset). Skipped when resuming an interrupted run. ⚠️ The split is per-segment, so the same subject can appear in both finetune-train and the held-out test (scored separately via `infer.sh`) → that test number is optimistic, not subject-disjoint.

## Git

Conventional Commits. `git commit`/`push` are only run on explicit user authorization. `gitpull.sh` targets a gh-proxy mirror of `github.com/jasonwei1002/BPFlow`. Never commit `rawdata/`, `wavflow/`, `output/`, checkpoints, or secrets (all gitignored).
