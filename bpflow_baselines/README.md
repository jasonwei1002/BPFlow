# bpflow_baselines

Faithful reproduction of the six MD-ViSCo baselines —
**MD-ViSCo, NABNet, PatchTST, PPG2ABP, P2E-WGAN, WaveNet** — on **bpflow's data,
splits and metrics**, for the four single-input directions
`ecg2abp`, `ppg2abp`, `ecg2ppg`, `ppg2ecg` **plus the dual-input
`ecg_ppg2abp`** (ECG+PPG→ABP, bpflow's flagship task). Single seed. Per-paper
hyperparameters.

The model architectures are vendored (Hydra-stripped) from
`MD-ViSCo/src/model/` under `models/`; everything else (data, training, eval) is
self-contained here and reuses `bpflow.data` (identical splits/normalization
constants) and `bpflow.eval` (identical metric definitions). No runtime
dependency on `MD-ViSCo/`.

## What "same as bpflow" means

- **Data + splits**: a thin adapter (`data.py`) wraps `bpflow.data.PulseDBDataset`,
  so the train/val/test partition, seeds, clip+recenter normalization and the
  finetune 8:1:1 stratified split are *exactly* bpflow's. Full-length waveforms
  are recovered losslessly via `unpatchify`.
- **Test metrics**: every prediction is reconstructed to mmHg (`->ABP`) or [0,1]
  (bridge) and scored with bpflow's own `evaluate` / `waveform_metrics`. The
  emitted `metrics.json` is structurally identical to bpflow's single-task infer
  output (`waveform` + `SBP`/`DBP`/`MAP` AAMI/BHS for `->ABP`; `waveform` only
  for bridge).

## Two-stage design (per MD-ViSCo)

Each baseline keeps its paper's normalization + reconstruction. Data/splits and
the `->ABP` mmHg test metric are bpflow's regardless of the internal mechanism.

| Model | `->ABP` target | reconstruction to mmHg | two-stage? |
|---|---|---|---|
| WaveNet, PPG2ABP, P2E-WGAN | global min-max over clip bounds [20,250] -> [0,1] | `w*(hi-lo)+lo` | single-stage / cascade |
| NABNet, PatchTST, MD-ViSCo | per-sample min-max -> [0,1] (shape only) | `w01*(SBP-DBP)+DBP`, SBP/DBP from a stage-2 BP head (predicts global-min-max-normalized SBP/DBP, de-normalized with [20,250]) | yes (BP predictor) |

- **PPG2ABP** is the two-net cascade (UNetDS64 deep-supervised → MultiResUNet1D).
- **P2E-WGAN** is the conditional WGAN-GP (Tanh generator + PatchGAN critic).
- **Bridge directions** (`ecg2ppg`/`ppg2ecg`) use bpflow's recenter scale for the
  target (so bridge metrics are comparable to bpflow) and run **stage-1 only**
  (no BP head), evaluated in [0,1].

## Dual-input direction `ecg_ppg2abp`

The flagship task gets every baseline as a same-input comparison, so bpflow's
ECG+PPG→ABP number is not confounded by "two inputs vs one". The source is
**channel-stacked** (ECG=ch0, PPG=ch1 → a 2-channel input `(2,L)`); the `→ABP`
target, normalization, BP head and metrics are otherwise unchanged. **It is not a
uniform "add a channel"** — each architecture needed model-specific handling
(verified against each original paper/official code, see
`plan/baselines-repro/dualinput-research.md`):

| Model | dual-input handling | faithfulness |
|---|---|---|
| NABNet | UNet already accepts `in_channels`; only the hardcoded `1` changed. BP head reads the ABP wave, unaffected. | in-design (original used multi-channel PPG derivatives) |
| PatchTST | passed as **2 channel-independent channels**; fusion only at the regression head (`forward` permute, not squeeze). | **native** PatchTST multivariate handling |
| WaveNet | decoupled input width from output width (`classes` doubled as both); `start_conv` takes 2, output stays 1. | channel-stack adaptation |
| PPG2ABP | **only stage-1 (UNetDS64)** takes 2 ch; stage-2 stays 1-ch (it refines stage-1's 1-ch output). | channel-stack adaptation |
| P2E-WGAN | generator + **conditional critic** both change; critic first block is `in+out` (=3), not `in*2`. | channel-stack adaptation |
| MD-ViSCo | UNet first conv **and** the BP head (fed the raw source) take 2 ch. | **native** — MD-ViSCo's own stage-1 channel-stacks multi-source: `extract_input` gathers sources into `[B,S_max,T]` (inactive slots zeroed via `src_mask`) → `conv_init_features(in_channels=S_max)`. Our 2-ch ECG+PPG matches it (both slots active, no zeroing). (`vital_encoders` in MD-ViSCo's stage-2 are for the BP/text refinement, not the ABP waveform; this port already omits that stage.) |

Channel count is derived from the direction (`norms.num_source_channels`); no
config change is needed. Baseline checkpoints load strictly via
`engine.load_state` (not bpflow's `load_model_state`, which drops `bp_head.*` — a
removed bpflow feature, but MD-ViSCo's live stage-2 head).

See `plan/baselines-repro/design.md` for the full contract and
`plan/baselines-repro/notes.md` for the source-of-truth on every constant.

## Commands

```bash
# CPU smoke test (this IS the correctness gate: overfits real segments, asserts
# loss collapse + a well-formed bpflow metrics report for all 6 models).
python -m bpflow_baselines.smoke_test                  # ~2 min on CPU
python -m bpflow_baselines.smoke_test --steps 120 --n 8

# Pretrain on PulseDB (Train_Subset; test = CalFree). <model> <direction>.
bash train_baseline_pulsedb.sh nabnet ecg2abp                 # all GPUs
bash train_baseline_pulsedb.sh patchtst ppg2abp --nproc 4
bash train_baseline_pulsedb.sh wavenet ecg2ppg training.lr=5e-4   # dotted overrides

# Finetune a pretrained checkpoint on CalFree (8:1:1 stratified), like bpflow.
bash finetune_baseline_pulsedb.sh nabnet ecg2abp output/baselines/<pretrain>/checkpoint_best.pth

# Score the held-out CalFree 10% test split (single process; same finetune split).
bash infer_baseline_pulsedb.sh nabnet ecg2abp output/baselines/<finetune>/checkpoint_best.pth
# -> writes <ckpt_dir>/infer_<direction>/metrics.json

# Raw entry points:
python -m bpflow_baselines.train --model wavenet --direction ppg2abp --config bpflow_baselines/config/wavenet.yaml
python -m bpflow_baselines.infer --model wavenet --direction ppg2abp --ckpt <path> --split test
```

`--nproc 1` runs a single process directly (no torchrun); `--nproc N` / `gpu` uses
DDP via torchrun.

### Full sweep + comparison table

```bash
# Run the whole 6x4 grid (pretrain -> finetune -> infer per cell). Resumable
# (--skip-existing), fault-tolerant (a failed cell doesn't abort the sweep),
# deterministic run dirs output/baselines/<model>_<direction>_{pre,ft}/.
bash run_baselines_grid.sh                                   # full grid, all GPUs
bash run_baselines_grid.sh --nproc 4 --skip-existing         # 4 GPUs, resume
bash run_baselines_grid.sh --models "nabnet patchtst" --directions "ecg2abp ppg2abp"
bash run_baselines_grid.sh --stages "infer"                  # only (re)score finetunes
bash run_baselines_grid.sh --dry-run                         # print the plan only
bash run_baselines_grid.sh -- training.batch_size=64         # extra overrides after --

# Aggregate every infer metrics.json into comparison tables (console + CSV + MD):
#   summary_abp.{csv,md}     -> wave MAE/RMSE/Pearson + SBP/DBP/MAP AAMI/BHS
#   summary_bridge.{csv,md}  -> wave MAE/RMSE/Pearson (normalized)
PYTHONPATH=. python -m bpflow_baselines.summarize            # root output/baselines
PYTHONPATH=. python -m bpflow_baselines.summarize --root output/baselines --out report/
```

`<model>` ∈ `{mdvisco, nabnet, patchtst, ppg2abp, p2e_wgan, wavenet}`,
`<direction>` ∈ `{ecg2abp, ppg2abp, ecg_ppg2abp, ecg2ppg, ppg2ecg}`.

The full grid is 6 models × 5 directions = 30 runs (single seed). Mirrors bpflow's
`train → finetune → infer` protocol; you may run any single stage.

## Per-paper hyperparameters (config/<model>.yaml)

Values below are each model's **original-paper** hyperparameters (verified against
the source papers / official code — see `plan/baselines-repro/hparam-verification.md`).

| Model | optimizer | batch | epochs/steps | scheduler | early stop | loss |
|---|---|---|---|---|---|---|
| MD-ViSCo | Adam 1e-3 | 2048 | 30K steps | plateau (pat 3) | 5 | MSE (+ BP L1†) |
| NABNet | Adam 3e-4 | 32 | 200 ep | none | 15 | MSE, D_S off (+ BP L1‡) |
| PatchTST | Adam 1e-4 | 128 | 100 ep | OneCycleLR | 100 | MSE (+ BP L1) |
| PPG2ABP | Adam 1e-3 | 256 | 100 ep | none | 20 | MAE (approx, deep-sup) + MSE (refine) |
| P2E-WGAN | Adam 2e-4, β=(0.5,0.999) | 192 | 25 ep | none | — | WGAN-GP (n_critic 3, λ_gp 10, λ_mse 50) |
| WaveNet§ | Adam 1e-3 | 32 | 150 ep | plateau (pat 6) | 12 | MSE |

† MD-ViSCo refinement原用 L_MAE + weighted-contrastive-loss(患者信息)；本复现无 demographics，简化为 L1。
‡ NABNet原 BP 估计用传统 ML(sklearn)；统一 DL 框架内用 L1 头替代。
§ WaveNet 原论文为音频 μ-law 生成，无血压训练 recipe → 训练超参为 adaptation（架构忠实）。

`batch_size` is the **GLOBAL** batch (the paper's value); under DDP it is split
across GPUs (`global // world_size` per GPU) so the effective global batch and the
paper's lr stay correct on any GPU count. Model/training selection picks the
checkpoint by **val waveform MAE** (mmHg for `->ABP`, normalized for bridge) —
aligned with bpflow's test metric. See `plan/baselines-repro/hparam-verification.md`
for the field-by-field check against each original paper.

## Deviations from MD-ViSCo (documented)

- **Single global normalization for source/bridge**: ECG/PPG inputs and bridge
  targets use bpflow's recenter scale (not MD-ViSCo's per-sample `minmax_zc`), to
  keep data identical to bpflow. Affects inputs only; `->ABP` follows the paper.
- **No double-scaling**: MD-ViSCo's eval assembly has a known double-de-normalize
  hazard for the scaling models; this reproduction reconstructs cleanly and scores
  with bpflow's `evaluate`, so the mmHg numbers are correct and comparable.
- **Minimal BP head for MD-ViSCo/NABNet**: the stage-2 SBP/DBP predictor is a
  compact CNN-encoder + MLP (no demographics / contrastive loss, which are unused
  here), faithful to "refinement = BP predictor" without the heavy text/PI stack.
- **Joint two-stage training**: stage-1 (waveform) and stage-2 (BP head) both read
  the source and are trained jointly in one run (equivalent to MD-ViSCo's
  sequential training since the BP head does not depend on stage-1 weights).
- **Dual-input is channel-stacked**: `ecg_ppg2abp` feeds ECG+PPG as a 2-channel
  input to every baseline (see the dual-input table above). This is **native** for
  PatchTST (channel-independence), NABNet (in-design), and MD-ViSCo (its own
  stage-1 channel-stacks multi-source via an `[B,S_max,T]` zero-masked input), and
  a pragmatic adaptation for WaveNet/PPG2ABP/P2E-WGAN (single-input by original
  design).
