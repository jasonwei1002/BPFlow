"""Config schema + small training helpers for BPFlow.

Config is OmegaConf-structured from dataclasses (frozen defaults) and supports
a ``_base_:`` include (same mechanism as WavFlow). Device selection is
explicit and CPU-capable so a tiny smoke test runs without a GPU.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .data import DEFAULT_MODALITY_DROPOUT_PROBS

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str = "bpflow_jointstream"
    patch_size: int = 10
    hidden_dim: int = 512
    num_heads: int = 8
    depth: int = 12
    mlp_ratio: float = 4.0
    joint_depth: int = 8  # number of 3-stream (ABP+ECG+PPG) joint-attention layers
    use_demo: bool = False  # condition on demographics (age/gender/height/weight/bmi)


@dataclass
class DataConfig:
    train_npy: str = "rawdata/pulsedb/Train_Subset.npy"
    test_npy: str = "rawdata/pulsedb/CalFree_Test_Subset.npy"
    seq_len: int = 1250
    ecg_channel: int = 0
    ppg_channel: int = 1
    abp_channel: int = 2
    split_seed: int = 42
    val_fraction: float = 0.2
    # ABP normalization constants — recomputed from the FULL train split
    # (seed=42, 80%, post-clip [20,250]); see plan/notes.md.
    abp_mean: float = 81.94
    abp_std: float = 24.43
    abp_clip_low: float = 20.0
    abp_clip_high: float = 250.0
    cond_recenter: bool = True
    # Conditioning modality (input "direction"): which signal(s) drive ABP.
    # "ecg_ppg" (default) = both; "ecg" = ECG only (PPG masked to 0); "ppg" =
    # PPG only. Masking keeps the architecture / params / checkpoint shape
    # identical across directions; train and infer MUST use the same value.
    cond_modality: str = "ecg_ppg"
    # Per-sample modality dropout (TRAIN split only) — trains ONE unified model
    # that handles ecg_ppg / ecg / ppg inputs. Each training sample randomly
    # picks a modality (others masked, same as cond_modality); val/test/infer
    # keep the fixed cond_modality above. Probs are [ecg_ppg, ecg, ppg], biased
    # to the joint case by default. Off → fixed cond_modality everywhere.
    modality_dropout: bool = False
    modality_dropout_probs: List[float] = field(
        default_factory=lambda: list(DEFAULT_MODALITY_DROPOUT_PROBS)
    )
    # Clinical-metric TRUE source for infer.py. "waveform" = derive SBP/DBP/MAP
    # per-beat from the true ABP wave (no CSV, no definitional offset); "csv" =
    # the CSV cuff [sbp, dbp, map] label; "both" = report both side by side. PRED
    # is always per-beat from the generated wave. (Trainer val/test stay waveform.)
    eval_true_source: str = "waveform"
    # Train/val split. "segment" = random per-segment (leaks subjects across
    # train/val → optimistic val); "subject" = subject-disjoint via CSV
    # subject_id (honest val, matches the CalFree test setting).
    split_mode: str = "segment"
    # Demographic z-score constants from the full train split (non-NaN); used
    # only when model.use_demo is true. height/weight/bmi are ~48% missing →
    # carried with a missing flag, NaN→0 (see standardize_demo).
    demo_age_mean: float = 61.11
    demo_age_std: float = 15.10
    demo_height_mean: float = 162.50
    demo_height_std: float = 9.64
    demo_weight_mean: float = 60.82
    demo_weight_std: float = 11.66
    demo_bmi_mean: float = 22.92
    demo_bmi_std: float = 3.44
    # Finetune mode: ignore train_npy/split_mode and instead split the CalFree
    # `test_npy` into train/val/test by a fixed-seed per-segment partition
    # (8:1:1 by default). Used by the finetune flow to adapt a pretrained model
    # to the CalFree domain and report on its own held-out test split.
    finetune: bool = False
    finetune_val_fraction: float = 0.1
    finetune_test_fraction: float = 0.1
    # Fraction of the finetune TRAIN split to actually use (data-efficiency
    # studies). 1.0 = all; e.g. 0.25 keeps a fixed-seed 25% subset of train.
    # val/test are NOT subsampled, so metrics stay comparable across ratios.
    finetune_train_ratio: float = 1.0
    # How the finetune 8:1:1 split is drawn:
    #   segment    = per-segment random over all CalFree (subjects leak across
    #                splits; original behavior).
    #   stratified = split each subject's own segments 8:1:1 → segment-balanced
    #                per subject; subjects still overlap all splits (NOT
    #                subject-disjoint). Needs the sibling CSV for subject_id.
    finetune_split_mode: str = "segment"


@dataclass
class SamplingConfig:
    prediction_type: str = "v"  # 'x' or 'v'
    method: str = "euler"  # 'euler' or 'adaptive'
    num_steps: int = 16
    noise_scale: float = 1.0
    noise_shift: float = 1.0
    min_sigma: float = 0.0
    logit_mean: float = 0.0  # training timestep t ~ sigmoid(N(mean, scale))
    logit_scale: float = 1.0


@dataclass
class TrainingConfig:
    loss_type: str = "v"  # 'x' or 'v'
    epochs: int = 100
    batch_size: int = 64
    # Batch size for validation / test (eager sampler, no backward → safe to make
    # large independent of the train batch). -1 = reuse batch_size. Set this larger
    # than a tiny train batch_size so small-batch runs don't pay a slow validation.
    val_batch_size: int = -1
    num_workers: int = 4
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    clip_grad_norm: float = 1.0
    seed: int = 14159265
    log_freq: int = 50
    ckpt_freq_epoch: int = 5
    val_freq_epoch: int = 5  # epochs between in-loop validation (0 = disabled)
    # Stride-subsample val to this fraction for in-loop evaluation (speed). 1.0 =
    # full val; e.g. 0.1 evaluates every 10th val segment — a representative ~10%
    # that still covers all subjects, fixed across epochs so the MAE trend stays
    # comparable. Applied before the DDP shard; the single val-subsampling knob.
    val_eval_fraction: float = 1.0
    # Monitor-only: per-modality flow-matching loss on val (ecg_ppg/ecg/ppg),
    # logged as val/loss_<modality> (fixed seed → epochs comparable). Useful to
    # watch a unified (modality_dropout) model. -1 = full val, 0 = off,
    # N > 0 = cap batches PER RANK.
    val_loss_max_batches: int = 50
    ema_decay: float = 0.9999
    use_ema: bool = True
    # ReduceLROnPlateau + early stop, counted in validation rounds w/o improvement
    # (= epochs when val_freq_epoch == 1). 0 disables that branch.
    lr_patience: int = 5       # val rounds w/o val improvement before lr *= lr_decay
    lr_decay: float = 0.1      # lr multiplier applied on each plateau
    early_stop_patience: int = 10  # val rounds w/o val improvement before stopping
    output_dir: str = "output"
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda'
    use_swanlab: bool = False  # log metrics to SwanLab (rank-0 only)
    swanlab_project: str = "bpflow"
    swanlab_mode: str = "online"  # online | local | offline | disabled ('cloud' = legacy alias for online)
    amp_dtype: str = "bfloat16"  # 'bfloat16' | 'float16' | 'float32' (cuda only)
    use_compile: bool = True  # torch.compile the training-forward path (CUDA only)
    max_steps: int = -1  # cap total steps (smoke); -1 = unlimited
    repeat_factor: int = 1  # repeat each batch along B (smoke overfit aid)
    # Initialize model (+ EMA) weights from this checkpoint at the start of a
    # FRESH run, then train normally (optimizer/epoch/step reset). Used by the
    # finetune flow; ignored when resuming an interrupted run. Empty = off.
    init_from_ckpt: str = ""
    # Resume an interrupted run IN PLACE: reuse this existing run dir as exp_dir
    # (no new timestamp), load its checkpoint_latest.pth, and continue the same
    # SwanLab run. Empty = off (fresh timestamped run). Set via --resume.
    resume_dir: str = ""
    # After training finishes, evaluate on the CalFree test set (best-by-val EMA
    # weights), log test/* to SwanLab, and write test_metrics.json. DDP-sharded.
    run_test_after_train: bool = False
    test_max_segments: int = -1  # cap test segments (-1 = all)


@dataclass
class BPFlowConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ----------------------------------------------------------------------------
# Config loading (with _base_ include)
# ----------------------------------------------------------------------------
def _merge_base(cfg: DictConfig, config_path: str) -> DictConfig:
    if "_base_" not in cfg:
        return cfg
    base_rel = str(cfg._base_)
    base_path = os.path.join(os.path.dirname(config_path), base_rel)
    base = OmegaConf.load(base_path)
    base = _merge_base(base, base_path)
    merged = OmegaConf.merge(base, cfg)
    del merged["_base_"]
    return merged  # type: ignore[return-value]


def load_config(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> DictConfig:
    schema = OmegaConf.structured(BPFlowConfig)
    if config_path is not None and os.path.exists(config_path):
        file_cfg = OmegaConf.load(config_path)
        file_cfg = _merge_base(file_cfg, config_path)
        cfg = OmegaConf.merge(schema, file_cfg)
    else:
        cfg = schema
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg  # type: ignore[return-value]


# Prefixes of state-dict keys for since-removed features. Pre-removal checkpoints
# carry these as now-unexpected keys; their body weights are still valid, so we
# strip them on load. Anything else stays, so a real mismatch still fails loudly.
#   empty_*                      — classifier-free-guidance null conditions
#   context_encoder.* / bp_head.* — ANIL/CAVIA per-subject context + cuff head
_LEGACY_KEY_PREFIXES = ("empty_", "context_encoder.", "bp_head.")


def drop_legacy_keys(state: dict) -> dict:
    """Drop removed-feature keys so pre-removal checkpoints still load."""
    return {k: v for k, v in state.items() if not k.startswith(_LEGACY_KEY_PREFIXES)}


# Prefixes of params ADDED after some checkpoints were saved. They are allowed to
# be MISSING on load (kept at fresh init) — e.g. loading a pre-null-token model
# for finetune — without masking a genuine architecture mismatch.
#   null_ecg / null_ppg — learned "modality absent" tokens (cond_mask)
_NEW_OPTIONAL_PREFIXES = ("null_ecg", "null_ppg")


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    """Load weights tolerantly: drop removed-feature keys, allow new-feature
    params to be missing (older checkpoints predate them), but still raise on any
    real missing/unexpected key (wrong/incompatible checkpoint)."""
    state = drop_legacy_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    optional_missing = [k for k in missing if k.startswith(_NEW_OPTIONAL_PREFIXES)]
    missing = [k for k in missing if not k.startswith(_NEW_OPTIONAL_PREFIXES)]
    if optional_missing:
        logger.warning(
            "Checkpoint missing new-feature params %s; kept at init "
            "(expected only for pre-null-token checkpoints).", optional_missing,
        )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model architecture "
            f"(missing={missing}, unexpected={list(unexpected)}). "
            "Use the config the checkpoint was trained with."
        )


# ----------------------------------------------------------------------------
# Reproducibility / device / optimizer helpers
# ----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(device_str)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def add_weight_decay(model: torch.nn.Module, weight_decay: float = 0.0, skip_list=()) -> List[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1-D params (norms) and biases are excluded from weight decay.
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def adjust_learning_rate(optimizer, global_step: int, cfg, lr_scale: float = 1.0) -> float:
    """Step-based linear warmup, then constant lr, times a plateau ``lr_scale``."""
    lr = float(cfg.training.lr)
    warmup = int(cfg.training.warmup_steps)
    if warmup > 0 and global_step < warmup:
        lr = lr * (global_step + 1) / warmup
    lr *= lr_scale
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


__all__ = [
    "ModelConfig",
    "DataConfig",
    "SamplingConfig",
    "TrainingConfig",
    "BPFlowConfig",
    "load_config",
    "drop_legacy_keys",
    "load_model_state",
    "set_seed",
    "pick_device",
    "is_main_process",
    "add_weight_decay",
    "adjust_learning_rate",
]
