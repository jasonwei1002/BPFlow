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
    # ANIL/CAVIA per-subject context: a low-dim vector adapted in the meta inner
    # loop (body frozen) and injected as a global_c add-on. Zero-init -> no-op start.
    use_context: bool = False
    context_dim: int = 64


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
    # Cuff SBP/DBP z-score constants (full train CSV) for the scalar-supervised
    # meta inner loop (model.use_context). These are the CALIBRATION cuff readings
    # carried by the support set; never the query's own SBP/DBP (the eval target).
    bp_sbp_mean: float = 118.60
    bp_sbp_std: float = 21.03
    bp_dbp_mean: float = 61.86
    bp_dbp_std: float = 12.65
    # Finetune mode: ignore train_npy/split_mode and instead split the CalFree
    # `test_npy` into train/val/test by a fixed-seed per-segment partition
    # (8:1:1 by default). Used by the finetune flow to adapt a pretrained model
    # to the CalFree domain and report on its own held-out test split.
    finetune: bool = False
    finetune_val_fraction: float = 0.1
    finetune_test_fraction: float = 0.1


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
    num_workers: int = 4
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    clip_grad_norm: float = 1.0
    seed: int = 14159265
    log_freq: int = 50
    ckpt_freq_epoch: int = 5
    val_freq_epoch: int = 5  # epochs between in-loop validation (0 = disabled)
    val_max_batches: int = 20  # cap val batches for speed
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
    swanlab_mode: str = "cloud"  # cloud | local | offline | disabled
    amp_dtype: str = "bfloat16"  # 'bfloat16' | 'float16' | 'float32' (cuda only)
    use_compile: bool = True  # torch.compile the training-forward path (CUDA only)
    max_steps: int = -1  # cap total steps (smoke); -1 = unlimited
    repeat_factor: int = 1  # repeat each batch along B (smoke overfit aid)
    # Initialize model (+ EMA) weights from this checkpoint at the start of a
    # FRESH run, then train normally (optimizer/epoch/step reset). Used by the
    # finetune flow; ignored when resuming an interrupted run. Empty = off.
    init_from_ckpt: str = ""
    # After training finishes, evaluate on the CalFree test set (best-by-val EMA
    # weights), log test/* to SwanLab, and write test_metrics.json. DDP-sharded.
    run_test_after_train: bool = False
    test_max_segments: int = -1  # cap test segments (-1 = all)


@dataclass
class MetaConfig:
    # ANIL/CAVIA meta-training (needs model.use_context: true). When enabled,
    # train() runs the episodic meta loop instead of the standard epoch loop;
    # validation/test become subject-disjoint K-shot adaptation (honest few-shot).
    enabled: bool = False
    # Inner-loop adaptation target:
    #   "scalar"   — cuff-only: adapt phi by matching the support's SBP/DBP via a
    #                BPHead (support carries ECG/PPG + cuff scalars, NO ABP).
    #   "waveform" — reference-ABP: adapt phi by the flow-matching loss on support
    #                ABP (needs the support's ABP waveform).
    inner_objective: str = "scalar"
    bp_loss_weight: float = 0.1  # outer-loss weight on the query SBP/DBP head term
    k_inner: int = 5            # inner adaptation steps on the support set
    inner_lr: float = 0.05      # inner-loop SGD step on the context phi
    support_size: int = 5       # Ks: support segments per training episode
    query_size: int = 5         # Kq: query segments per training episode
    meta_batch_subjects: int = 8  # subjects per outer meta-step (per rank)
    meta_steps: int = 20000     # total outer meta-steps
    log_every: int = 50
    ckpt_every: int = 1000
    val_every: int = 1000       # outer steps between K-shot validations
    val_subject_fraction: float = 0.1  # subject-disjoint val split of train subjects
    eval_ks: str = "0,1,3,5,10"        # K-shot points reported at eval
    eval_max_subjects: int = 200       # cap subjects per K-shot eval (per rank shard)
    eval_max_query: int = 40           # cap query segments per subject in eval
    num_workers: int = 4


@dataclass
class BPFlowConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)


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


def drop_legacy_keys(state: dict) -> dict:
    """Drop removed-feature keys so pre-removal checkpoints still load.

    The classifier-free-guidance null conditions (``empty_ecg``/``empty_ppg``) were
    removed; an old checkpoint carries them as now-unexpected keys. They were a
    no-op (zero contribution), so dropping them loses nothing. Everything else is
    left untouched, so a genuinely incompatible checkpoint still fails loudly.
    """
    return {k: v for k, v in state.items() if not k.startswith("empty_")}


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
    "set_seed",
    "pick_device",
    "is_main_process",
    "add_weight_decay",
    "adjust_learning_rate",
]
