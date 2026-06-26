# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BPFlow is a **symmetric multi-target** generator over three cardiovascular streams (ABP, ECG, PPG) via **conditional flow matching**. It is Meta's WavFlow MMDiT adapted into a 3-stream joint-attention conditional DiT: ABP, ECG, and PPG are three token streams that attend jointly, and a per-sample **task** picks which stream is the noised generation TARGET and which are the clean CONDITIONS (routed by an attention mask). The flagship task is ECG+PPG→ABP; the model also trains the single-modality →ABP directions (ecg2abp, ppg2abp) and the cross-modal bridge tasks ppg2ecg / ecg2ppg — all in one unified model. Trained/evaluated on PulseDB. The `wavflow/` directory is the original upstream reference (gitignored); `bpflow/` is the self-contained project — its WavFlow primitives are **vendored** under `bpflow/model/_vendor/`, so bpflow has no runtime dependency on `wavflow/`.

There is no `README.md`, `pyproject.toml`, or packaging — the project runs as a module (`python -m bpflow.*`). There is no `pytest`/`ruff`/`mypy` config checked in; the real correctness gate is the CPU smoke test below.

Comparison **baselines** live in a separate, self-contained package `bpflow_baselines/` (six reproduced models — MD-ViSCo, NABNet, PatchTST, PPG2ABP, P2E-WGAN, WaveNet — reusing bpflow's data/splits/metrics over the same five directions). It has **no runtime dependency on the upstream `MD-ViSCo/`** (gitignored) and its own README + CPU smoke test; see `bpflow_baselines/README.md`.

## Commands

```bash
# Smoke test — full pipeline on CPU, no GPU (data→patch→forward→loss→backward→sample).
# This IS the test suite: overfits a few real segments, asserts loss collapse + recon improvement.
python -m bpflow.smoke_test                       # uses bpflow/config/smoke.yaml
python -m bpflow.smoke_test --n-steps 600 --n-samples 8

# Scripts come in a PulseDB set (*_pulsedb.sh) and a UCI set (*_uci.sh); the suffix
# picks the dataset/config. PulseDB uses pulsedb.yaml/pulsedb_finetune.yaml; UCI uses
# uci.yaml/uci_finetune.yaml (seq_len 1024, patch_size 8). Examples below show
# PulseDB; swap _pulsedb -> _uci for the UCI variant.

# Pretrain (single-node, multi-GPU via torchrun). --nproc <gpu|N> picks processes
# ('gpu' = all visible GPUs, default). Pretraining does NOT touch the test domain
# (no post-train test); it is reserved for finetune.
bash train_pulsedb.sh                                     # all GPUs, bpflow/config/pulsedb.yaml
bash train_pulsedb.sh --nproc 1                           # single GPU/CPU
CUDA_VISIBLE_DEVICES=0,1 bash train_pulsedb.sh --nproc 2  # pick GPUs
bash train_pulsedb.sh --config bpflow/config/other.yaml   # override (args pass through to bpflow.train)
bash train_pulsedb.sh training.lr=1e-4 data.task_probs=[0.3,0.2,0.2,0.15,0.15]  # dotted key=value CLI config overrides (any field)
bash train_uci.sh                                         # UCI pretrain (uci.yaml)
python -m bpflow.train --config bpflow/config/pulsedb.yaml   # no torchrun (single process)

# Finetune a pretrained model on the test domain. test_npy is split 8:1:1 (fixed
# seed) into train/val/test; finetunes on 80% (val for early stop) and produces
# WEIGHTS ONLY — run_test_after_train is OFF (it deadlocks under DDP). Score the
# held-out 10% test split separately with infer_<dataset>.sh (below).
# Args: <checkpoint> [--nproc N] [extra args]; new output/<ts>/.
bash finetune_pulsedb.sh output/<pretrain_ts>/checkpoint_best.pth            # all GPUs (CalFree)
bash finetune_pulsedb.sh output/<pretrain_ts>/checkpoint_best.pth --nproc 4  # 4 GPUs
bash finetune_pulsedb.sh <ckpt> data.finetune_train_ratio=0.25              # data-efficiency sweep (CLI override)
bash finetune_uci.sh output/<uci_pretrain_ts>/checkpoint_best.pth           # UCI finetune (uci_finetune.yaml)
python -m bpflow.train --config bpflow/config/pulsedb_finetune.yaml --init-ckpt <pretrained>

# Score a finetuned checkpoint on the held-out 10% test split over multiple seeds
# (mean±std). infer_<dataset>.sh loads that dataset's finetune config so the split
# matches finetune's fixed-seed 8:1:1. Defaults to --task all (every trained task)
# and emits metrics.json + recon plots. Args: <checkpoint> [--nproc N] [extra args].
bash infer_pulsedb.sh output/<finetune_ts>/checkpoint_best.pth              # all GPUs (CalFree test split)
bash infer_pulsedb.sh output/<finetune_ts>/checkpoint_best.pth --nproc 4    # 4 GPUs
bash infer_uci.sh output/<uci_finetune_ts>/checkpoint_best.pth             # UCI test split
python -m bpflow.infer --config bpflow/config/pulsedb_finetune.yaml --ckpt <path> --split test --num -1 --use-ema

# One-shot pipeline: pretrain -> finetune -> multi-seed infer in a SINGLE command, each
# stage auto-loading the previous stage's checkpoint_best.pth (run names pinned, so no
# timestamp guessing). run_<dataset>.sh = the three scripts above chained; dirs are
# output/<set>_pt|ft_<ts>_<dir>/ + output/<set>_infer_<ts>/. --pt/--ft/--in pass
# stage-specific overrides (pretrain/finetune/infer); --skip-infer stops after finetune.
bash run_pulsedb.sh                                       # pulsedb.yaml -> pulsedb_finetune.yaml -> infer_pulsedb.sh
bash run_uci.sh --nproc 4                                 # UCI variant (uci.yaml -> uci_finetune.yaml)
bash run_pulsedb.sh --pt "training.lr=3e-4" --ft "data.finetune_train_ratio=0.25" --in "--task ecg_ppg2abp"

# Comparison baselines (separate package; see bpflow_baselines/README.md). 6 models x 5
# directions on the SAME bpflow data/splits/metrics, single seed, per-paper hyperparams.
python -m bpflow_baselines.smoke_test                     # CPU correctness gate (this IS the baselines test)
bash run_baselines_grid.sh                                # full 6x5 grid: pretrain->finetune->infer per cell
bash train_baseline_pulsedb.sh nabnet ecg2abp            # one model x one direction (pretrain only)
PYTHONPATH=. python -m bpflow_baselines.summarize        # aggregate infer metrics.json -> comparison table

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
       ├─ patchify ×3         → abp_patches / ecg_patches / ppg_patches, each (N=125, P=10); ECG/PPG recentered −0.5
       └─ task draw           per-sample task (TASK_SPEC) → target_idx (0=ABP/1=ECG/2=PPG) + cond_present (3,)
  └─ BPFlowModel              model/networks.py        — 3-stream joint DiT, task-routed
       ├─ embed streams       noised_in×3 (target←noised) / cond_in{ecg,ppg} (←clean) / absent_token×3
       ├─ joint_blocks×8      model/blocks.py BPJointBlock — ABP+ECG+PPG attend jointly w/ task attn_mask, shared RoPE
       ├─ gather target       → fused_blocks×4 refine the target stream only (MMDitSingleBlock)
       └─ heads×3 → gather    → predicts flow of the target modality (B,N,P)
  └─ FlowMatching             model/flow_matching.py   — train: v-pred loss; infer: euler ODE, 16 steps
  └─ sample_target            sampling.py              — noise→ODE→unpatchify→denorm by target (ABP z-score | ECG/PPG +0.5) (B,L)
  └─ evaluate                 eval/metrics.py          — waveform MAE/RMSE/Pearson + clinical AAMI/BHS on SBP/DBP/MAP (→ABP)
```

Three glue modules tie it together and must stay consistent:
- **`sampling.py`** holds the single definition of `build_flow_matching` / `sample_target` / `flow_matching_loss`, shared by `trainer.py`, `infer.py`, and `smoke_test.py`. All three take `(target_idx, cond_present)` to pick the task; the loss noises only the target modality's patches (`_select_target` gathers them). Change the loss/sampling math here, not in three places.
- **`trainer_utils.py`** is the config schema (frozen-default dataclasses → OmegaConf structured) **and** the train helpers (`adjust_learning_rate`, `add_weight_decay`, `pick_device`, `is_main_process`). New config fields go on the dataclasses here first, or `load_config` drops them.
- **Factory/registry**: `model/__init__.py` (`build_model` via `MODEL_REGISTRY`, `@register_model`) and `data/__init__.py` (`build_dataset` via `DATASET_REGISTRY`). Add variants by registering a name; build from `cfg`.

## Config system

OmegaConf structured config with a `_base_:` include chain (resolved in `trainer_utils._merge_base`). Any field is overridable on the CLI as a dotted `key=value` (e.g. `bash train_pulsedb.sh training.lr=1e-4 'data.task_probs=[0.3,0.2,0.2,0.15,0.15]'`), parsed via `OmegaConf.from_dotlist` in `train.py` into `load_config(..., overrides=)`; a typo or type clash fails loudly at the struct-mode merge.
- **`base.yaml`** — only params SHARED across datasets: model width (hidden_dim/heads/depth), channel order, `prediction_type`, sampling + training defaults, and the task/split system. **Dataset-specific shapes/paths/normalization** (`patch_size`, `seq_len`, `train_npy`, `test_npy`, `abp_mean`, `abp_std`) live in the per-dataset configs (`pulsedb.yaml` / `uci.yaml`), with the `trainer_utils.py` dataclass defaults (= PulseDB) as the backstop. These still drive **checkpoint compatibility** (`infer.py` asserts `abp_mean`/`abp_std` match the checkpoint) — they just come from the per-dataset config now, not `base.yaml`. Defaults to `split_mode: segment` and the **multi-target task set** `tasks: [ecg_ppg2abp, ecg2abp, ppg2abp, ppg2ecg, ecg2ppg]` + `task_probs: [0.30, 0.20, 0.20, 0.15, 0.15]` (→ABP-heavy) + `eval_task: ""` — the UNIFIED model (all five tasks in one) is the default; inherited by `pulsedb.yaml`/`uci.yaml`. A per-direction **specialist** is a single-element `tasks` list, set via CLI override (e.g. `bash train_pulsedb.sh 'data.tasks=[ecg2abp]'`); the dedicated `gpu_ecg.yaml`/`gpu_ppg.yaml` configs were removed.
- **`pulsedb.yaml`** (inherits `base.yaml`) — real **pretraining** (epochs 1000 as a ceiling; early-stop + plateau decay end runs sooner; per-GPU batch 128, lr 3e-4, SwanLab cloud). Uses `split_mode: segment` (random per-segment train/val). Also `val_eval_fraction: 0.1` (validate on a representative stride-sampled 10% of val) and **`run_test_after_train: false`** — pretraining never touches CalFree (reserved for finetune).
- **`pulsedb_finetune.yaml`** (inherits `pulsedb.yaml`) — domain adaptation on CalFree. Spells out its own `training` block (tune finetune hyperparams independently; epochs 1000 ceiling). **`data.finetune: true`** splits the CalFree `test_npy` 8:1:1 (fixed seed) into train/val/test; `finetune_split_mode: stratified` (default) splits each subject's segments 8:1:1 (segment-balanced per subject; `segment` = per-segment random), and `finetune_train_ratio` (default 1.0) sub-samples the finetune-train split for data-efficiency studies (val/test untouched). **`run_test_after_train: false`** (it deadlocks under DDP): finetune produces weights only — score the held-out 10% with `bash infer_pulsedb.sh <ckpt>`. Weights init from a pretrained checkpoint via `--init-ckpt` (→ `training.init_from_ckpt`); optimizer/epoch/step reset.
- **`smoke.yaml`** (inherits `base.yaml`) — tiny CPU model for the smoke test (not a real model). Pins **`tasks: [ecg_ppg2abp, ppg2ecg]`, `task_probs: [0.5, 0.5]`** (one →ABP recon task + one target≠ABP bridge task, to exercise multi-target routing), so the smoke test runs from the `.npy` alone (no CSV). `smoke_test.py` itself trains BOTH tasks deterministically on every overfit segment (the batch is duplicated, one copy per task) so the ABP recon gate stays stable.

## Conventions and gotchas

- **`batch_size` is per-GPU.** Under DDP the global batch is `batch_size × num_gpus` (e.g. 128 × 8 = 1024). `lr` in `pulsedb.yaml` is already sqrt-scaled for that global batch. Do not auto-divide.
- **Normalization is z-score, not min-max**, and uses **fixed global constants** (never per-sample — that would leak the target). `abp_raw` (unclipped mmHg) is carried through for evaluation; all metrics are computed after `destandardize_abp`.
- **Validation reports MAE, not a loss** (for decisions). `Trainer.validate()` runs the ODE sampler on the val split for **every trained task** (`_active_tasks` = `trained_tasks(cfg.data.tasks, task_probs)`: specialist → its 1 task; unified → every task with prob>0), each forced onto the **same** val segments via `_distributed_eval(..., task_override=t)`. The **decision metric is the mean waveform MAE in mmHg over the trained →ABP tasks only** (`_active_abp_tasks` = the `ABP_TASKS` subset; bridge tasks ppg2ecg/ecg2ppg live in [0,1] normalized units, not mmHg-comparable, so they never drive selection); that mean — and only it — drives best-checkpoint, ReduceLROnPlateau, and early stopping. Logged: the full per-task report as **flat** keys `val/<metric>_<task>` (clean ids, e.g. `val/MAE_ppg2abp`, `val/MAE_ppg2ecg`, `val/SBP_AAMI_pass_ecg2abp`) plus `val/loss_<task>`, and top-level `val/MAE|RMSE|Pearson` = the across-→ABP-task means. →ABP tasks report the full clinical block; bridge tasks report waveform-only (`_assemble_report` drops BP cols when `sbp_p` is absent). The rank-0 log line `"[val] MAE mean=<float> mmHg over [...]"` is a **contract** parsed by `tune_modality_probs.py` (ASHA) — don't change its format. Cost: the sampler runs once per trained task (up to 5× for the full unified model). Validation is **DDP-sharded across all ranks** and merged on rank 0 via `_distributed_eval`, which gathers **per-segment scalars** (BP values + waveform error summaries), not waveforms, so full-set val scales to many GPUs (aggregation is exact because every segment has equal length). **`val_eval_fraction`** (≤1) stride-subsamples val to a representative fraction (covers all subjects, fixed across epochs → MAE trend comparable) **before** the DDP shard — the single val-subsampling knob (pulsedb.yaml `0.1`, pulsedb_finetune.yaml `1.0`).
- **Per-task train/val losses are monitor-only.** The flow-matching (v-prediction) loss is logged separately per **task** (`train/loss_<task>` / `val/loss_<task>`), to watch how the unified model fits each direction.
  - **Val** (`val/loss_<task>` via `_val_task_losses`): on the val split, fixed per-batch seed + unshuffled loader → **epoch-comparable**. Runs on **all ranks** (one `all_reduce`) inside the same EMA swap as the sampler, **before** the rank-0 early return. For each `_active_tasks()` task it forces that task's `(target_idx, cond_present)` over the batch and noises the corresponding target modality, so a specialist run logs just its one `val/loss_*`. `val_loss_max_batches`: `-1` = full val (pulsedb/pulsedb_finetune), `0` = off, `N>0` = cap per rank (base default 50). Absolute value depends on world_size.
  - **Train** (`train/loss_<task>` + `train/loss_epoch`, logged at **epoch end**): the **full-epoch** train loss, decomposed **for free** from the actual training loss — `flow_matching_loss(..., per_sample=True)` returns the `(B,)` per-sample loss, and `_train_step` buckets each sample by its drawn task (`_accumulate_task_loss` matches its live `(target_idx, cond_present)` against `TASK_SPEC`) into epoch accumulators; `_log_epoch_loss` all-reduces across ranks and logs the mean. No extra forward. So `train/loss_<task>` = mean train loss of samples *actually drawn* into that task this epoch (a specialist fills only its one task), distinct from `val/loss_<task>` (every val sample under that *fixed* task). `train/loss` (per-step instantaneous) is still logged every `log_freq`.
  - Neither influences best/plateau/early-stop (those still use the →ABP val MAE mean).
- **Run dirs are timestamped + direction-tagged**: `output/<YYYYMMDD_HHMMSS>_<slug>/`, where `<slug>` = `tasks_slug(cfg.data.tasks, cfg.data.task_probs)` (`uni5` for the full unified model, a single direction like `ecg2abp` for a specialist, `uniN` for any other subset) — so the dir and SwanLab run are self-describing. The auto name is built in `_make_run_name` (rank 0 picks the timestamp and broadcasts it under DDP; the slug is pure cfg → identical on every rank). The `run_<dataset>.sh` orchestrators instead **pin** this name explicitly via `training.run_name=<set>_pt|ft_<ts>_<slug>` so each stage can locate the previous stage's checkpoint deterministically (a pure-numeric `run_name` would be mis-parsed as an int by OmegaConf's dotlist → always letter-prefix it). **Resume is opt-in via `--resume <run_dir>`** (→ `training.resume_dir`): it reuses that existing dir as `exp_dir` (no new timestamp), loads its `checkpoint_latest.pth`, and **continues the same SwanLab run** (the run id is saved in the checkpoint as `swanlab_run_id` and replayed via `swanlab.init(id=..., resume="allow")`, online mode only). Without `--resume`, every launch is a fresh timestamped dir + fresh SwanLab run — a brand-new dir has no `checkpoint_latest.pth`, so resume never fires by accident. SwanLab otherwise gets no `experiment_name` (auto-generates its run id; `run_name` stored in its config).
- **Warnings are silenced at package import**: `bpflow/__init__.py` calls `configure_warnings()` (`bpflow/_warnings.py`) **before** any submodule imports torch/swanlab, filtering the deprecation-family noise (`DeprecationWarning`/`FutureWarning`/`PendingDeprecationWarning`, NOT `UserWarning`) for every entrypoint (train/infer/smoke_test). If a warning you expect is missing while debugging, loosen `_NOISE_CATEGORIES` there.
- **rank-0-only side effects** under DDP: terminal logging (other ranks at WARNING), tqdm bars, checkpoint saving, and metric logging. **Validation and the post-train test now run on all ranks** (sharded + gathered), but the report is assembled and decisions (`best_val`, `epochs_no_improve`, `lr_scale`, `should_stop`) are made on rank 0 and broadcast via `_sync_val_state`. ⚠️ Every rank must enter `_distributed_eval` together — its `all_gather` is a collective; an early per-rank return deadlocks DDP.
- **Purely conditional** (no classifier-free guidance): the model conditions on whichever clean streams the task marks present — no learned null conditions, no `cfg_strength`/`label_drop_prob`. Every parameter is exercised in each forward (all `noised_in`/`absent_token`/`heads` are gated/gathered, never skipped), so DDP runs with `find_unused_parameters=False`.
- **Task routing = attention mask + per-stream roles (NOT zeroing / null tokens).** The dataset emits all three modalities' clean `*_patches` plus per-sample `target_idx` (0/1/2) and `cond_present` `(3,)` over `[abp, ecg, ppg]`. In the model (`preprocess_conditions`) each stream takes one of three roles: **target** (embedded from the *noised* latent via `noised_in[s]`), **condition** (embedded from the *clean* signal via `cond_in[ecg|ppg]`), or **absent** (a learned `absent_token[s]`). `build_task_mask` (`model/attention_mask.py`) then builds an additive `(B,1,3N,3N)` mask routing attention UniCardio-style: a **target** query attends {its own stream + present conditions}; a **condition** query attends {present conditions only — NOT the noised target, so clean conditions never absorb target noise}; an **absent** query attends itself only (output discarded). The diagonal is always open → no all-`-inf` row → no softmax NaN. **No-leak is structural**: the clean signal of whichever modality is the target is never fed as a condition (mask + role gating), verified by grad checks. **All embedders/heads stay in the graph** every forward — `noised_in[s]` and every `heads[s]` run for all `s` (gated/gathered; non-selected get 0-grad, not None), so DDP `find_unused_parameters=False` holds. `absent_token` are new params: `load_model_state` tolerates them missing on old checkpoints (kept at init) while still flagging real mismatches.
- **`data.tasks` / `data.task_probs` / `data.eval_task`** (the task system; replaces the old `cond_modality`/`modality_dropout`). `tasks` is a subset of `TASK_SPEC` = `{ecg_ppg2abp, ecg2abp, ppg2abp, ppg2ecg, ecg2ppg}` (names use `2` for `->`, so they're clean identifiers in YAML/metric keys/filenames; `[]` = all five). `task_probs` is the per-sample draw distribution aligned to it (`[]` = uniform). `eval_task` is the single fixed task val/test/infer default to (`""` = first →ABP task in the set). A **single-element `tasks` list = a specialist** (set via CLI override, e.g. `data.tasks=[ecg2abp]`); the full five-task list = the unified model. Architecture/params/checkpoint shape are **identical** across any task set (every model has all three `noised_in`/`absent_token`/`heads`). The chosen `tasks`/`task_probs` are baked into the saved `config`; the **trained set** is `trained_tasks()` (`data/__init__.py`) = every task with **prob > 0** (so `[0.5,0.5,0,0,0]` allows only the first two; old checkpoints lacking the fields → `{ecg_ppg2abp}`). Both the `infer.py` guard (which task is valid) and the per-task val loss (which to log) use it. `infer.py` defaults to **`--task all`** (evaluates every task in `trained_tasks`; >1 → `metrics.json` nested per task, 1 → flat/back-compat); a pinned `--task <name>` must be in that trained set.
- **Per-sample task draw (the unified model).** With a multi-task `tasks` list, each **train** sample independently draws a task from `task_probs` (per-sample, **not** per-batch → gradient diversity within a batch), setting its `target_idx` + `cond_present`; val/test/infer use the fixed `eval_task` (the trainer overrides it per direction via `task_override`). Only the per-sample *choice* is randomized → one modality-agnostic model instead of N specialists. The draw is seeded per worker from `cfg.training.seed` (train loader `worker_init_fn=_seed_worker`), reproducible at a fixed `num_workers`. No BatchNorm (AdaLN), so per-sample task mixing within a batch is safe; the tiny per-target heads/embedders run once over the shared backbone, so multi-task adds negligible compute over a specialist. **The full five-task unified model is the default** (base.yaml); a single-element `tasks` list gives a per-direction specialist (smoke.yaml uses a 2-task list; a 1-element `tasks` list via CLI override is a specialist).
- **`torch.compile`** wraps the training-forward path (CUDA only, `use_compile`). Sampling/validation use the **eager** `model_raw`, so variable val batch sizes never trigger recompiles; `drop_last=True` keeps the train batch fixed.
- **EMA** weights are kept on `model_raw`; validation swaps in the current EMA (`_ema_swapped`). `infer.py --use-ema` and the post-train `run_test` instead load EMA weights from disk — `run_test` loads the **best-by-val checkpoint's** EMA (`_load_eval_weights`), not the last.
- **Flow-matching loss combos**: only `(v, v)` handles `min_sigma > 0` exactly; the other combos raise rather than train on a wrong target when `min_sigma != 0`.
- **Data paths live under `rawdata/`** (gitignored), e.g. `rawdata/pulsedb/Train_Subset.npy`. Subject-split (`split_mode: subject` / `finetune_split_mode: stratified`) additionally needs the **sibling CSV** `<npy_basename>.csv` (e.g. `Train_Subset.csv`) for its `subject_id` column, row-aligned to the npy; the dataset raises if it's missing or row counts mismatch. The CSV's `sbp/dbp/map` columns are the **label** — never read them as input.
- **`split_mode`**: `segment` (base + pulsedb.yaml default) is a random per-segment train/val split — the same subject lands in both train and val, so val MAE is optimistic. `subject` (optional) splits train/val by the CSV's `subject_id` → subject-disjoint, matches the CalFree test setting (needs the sibling CSV for the train/val split). Either way, **CalFree (test) is the honest, subject-disjoint generalization gate.** Changing `split_mode` changes which segments are train vs val, so metrics aren't comparable across modes.
- **`run_test_after_train`** (pulsedb.yaml **false**, pulsedb_finetune.yaml **false**): the in-process post-train test is **OFF everywhere**. When enabled, `Trainer.run_test()` evaluates the **best-by-val EMA** model on the `test` split via the shared `_distributed_eval`, forcing both clinical truth sources (`eval_true_source: both` → `test/*` per-beat-on-true-wave + `test/cuff_*` CSV cuff) and writing `test_metrics.json`. ⚠️ It is off because it can **deadlock under DDP**: a per-rank failure (e.g. `build_dataset`) *before* `_distributed_eval`'s `all_gather` is swallowed by `_maybe_run_test`'s try/except → surviving ranks hang on the collective → the run CRASHES, losing it. **Don't re-enable it under multi-GPU.** Score the test split safely and standalone with `bash infer_pulsedb.sh <ckpt>` (same eval, no in-training collective).
- **Finetune flow (`data.finetune: true`)** repurposes the CalFree `test_npy`: a fixed-seed 8:1:1 split (`finetune_val_fraction`/`finetune_test_fraction`, default 0.1/0.1) makes train/val/**test** all read CalFree, non-overlapping; `finetune_split_mode` is `stratified` (default, per-subject-balanced) or `segment` (per-segment random), and `finetune_train_ratio` (default 1.0) sub-samples only the finetune-train split (val/test fixed → comparable across ratios). Top-level `split_mode` is ignored. The pretrained model is loaded via **`--init-ckpt`** → `training.init_from_ckpt` (`Trainer._maybe_init_from_ckpt`): it copies `model` (+ `model_ema`) weights only, then trains fresh (optimizer/epoch/step reset). Skipped when resuming an interrupted run. ⚠️ The split is per-segment within subjects, so the same subject can appear in both finetune-train and the held-out test (scored via `infer_pulsedb.sh`) → that test number is optimistic, not subject-disjoint.

## Git

Conventional Commits. `git commit`/`push` are only run on explicit user authorization. `gitpull.sh` targets a gh-proxy mirror of `github.com/jasonwei1002/BPFlow`. Never commit `rawdata/`, `wavflow/`, `output/`, checkpoints, or secrets (all gitignored).
