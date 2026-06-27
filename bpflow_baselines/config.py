"""Config schema for the baseline reproduction.

Reuses bpflow's ``DataConfig`` (so ``bpflow.data.build_dataset`` works and the
splits/normalization constants are identical) and adds baseline-specific model /
training / experiment blocks. Supports the same ``_base_:`` include + dotted CLI
overrides as bpflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from bpflow.trainer_utils import DataConfig, _merge_base


@dataclass
class BaselineModelConfig:
    name: str = "wavenet"            # registry key (models/__init__.py)
    patch_size: int = 10             # only used by bpflow build_dataset (1250/10=125)
    params: Dict[str, Any] = field(default_factory=dict)  # per-model arch hyperparams


@dataclass
class BaselineExpConfig:
    direction: str = "ecg2abp"       # ecg2abp|ppg2abp|ecg_ppg2abp|ecg2ppg|ppg2ecg


@dataclass
class BaselineTrainingConfig:
    # optimizer (Adam, paper recipe)
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epochs: int = 100
    batch_size: int = 32             # GLOBAL batch (paper value); trainers split it as batch_size // world_size per GPU
    val_batch_size: int = 64
    num_workers: int = 4
    clip_grad_norm: float = 0.0      # 0 = off (papers mostly don't clip)
    seed: int = 42
    log_freq: int = 50
    ckpt_freq_epoch: int = 5
    val_freq_epoch: int = 1
    val_eval_fraction: float = 1.0
    # waveform loss base (per original paper): "mse" or "mae". aux_loss_base is
    # the deep-supervision aux loss ("" = same as loss_base; PPG2ABP uses
    # mse primary / mae aux).
    loss_base: str = "mse"
    aux_loss_base: str = ""
    # scheduler: "plateau" (ReduceLROnPlateau) | "onecycle" (per-step OneCycleLR,
    # PatchTST) | "none"
    scheduler: str = "none"
    lr_patience: int = 3
    lr_decay: float = 0.1
    onecycle_pct_start: float = 0.3
    # early stopping on val MAE (val rounds w/o improvement); 0 = off
    early_stop_patience: int = 0
    min_delta: float = 0.0
    # GAN-only (P2E-WGAN); ignored by the supervised trainer
    n_critic: int = 3
    lambda_gp: float = 10.0
    lambda_sample: float = 50.0
    gan_beta1: float = 0.5
    gan_beta2: float = 0.999
    # two-stage (NABNet/PatchTST/MD-ViSCo): weight of the BP-head L1 loss
    lambda_bp: float = 1.0
    output_dir: str = "output/baselines"
    device: str = "auto"
    use_swanlab: bool = False
    swanlab_project: str = "bpflow-baselines"
    swanlab_mode: str = "online"
    swanlab_group: str = ""          # group related runs (pretrain/finetune) on the dashboard
    amp_dtype: str = "float32"       # baselines default to fp32 (small nets, stable)
    max_steps: int = -1
    repeat_factor: int = 1
    init_from_ckpt: str = ""         # finetune: copy weights from this ckpt, reset optim
    resume_dir: str = ""
    run_name: str = ""


@dataclass
class BaselineConfig:
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    baseline: BaselineExpConfig = field(default_factory=BaselineExpConfig)


def load_config(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> DictConfig:
    schema = OmegaConf.structured(BaselineConfig)
    if config_path is not None and os.path.exists(config_path):
        file_cfg = OmegaConf.load(config_path)
        file_cfg = _merge_base(file_cfg, config_path)
        cfg = OmegaConf.merge(schema, file_cfg)
    else:
        cfg = schema
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg  # type: ignore[return-value]


def overrides_from_extra(extra: List[str]) -> dict:
    """Parse residual ``key=value`` CLI tokens into a nested override dict."""
    if not extra:
        return {}
    for tok in extra:
        if "=" not in tok:
            raise SystemExit(f"override {tok!r} must be key=value (dotted)")
    return OmegaConf.to_container(OmegaConf.from_dotlist(list(extra)))  # type: ignore[return-value]
