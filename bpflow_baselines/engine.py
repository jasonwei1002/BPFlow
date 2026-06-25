"""Shared training engine helpers: DDP, run dir, SwanLab, validation, ckpt.

Used by both the supervised trainer and the WGAN-GP trainer so they share the
exact same validation (reconstruct -> bpflow metric), checkpointing and logging.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from bpflow.trainer_utils import is_main_process

from .models.base import BaselineModule, pad_to_multiple
from .norms import ABP_TARGET_MODE
from .reconstruct import reconstruct_pred, reconstruct_true

logger = logging.getLogger(__name__)


def load_state(model: torch.nn.Module, state: dict) -> None:
    """Strict load for baseline checkpoints.

    Baseline architecture always matches its own checkpoint, so we load
    strictly. We deliberately do NOT reuse bpflow's ``load_model_state``: that
    drops ``bp_head.*`` keys (a since-removed bpflow cuff head), but for the
    MD-ViSCo baseline ``bp_head`` is a live stage-2 SBP/DBP predictor — dropping
    it would falsely report the head as missing.
    """
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# distributed
# ---------------------------------------------------------------------------
def setup_distributed() -> Dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank, "distributed": distributed}


def cleanup_distributed(dist_info: Dict[str, int]) -> None:
    if dist_info["distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def subsample_val(ds, fraction: float):
    """Stride-subsample a val dataset to ~fraction (covers the whole set evenly,
    fixed across epochs so the MAE trend stays comparable). 1.0 = no-op."""
    from torch.utils.data import Subset
    if fraction >= 1.0:
        return ds
    step = max(1, round(1.0 / max(fraction, 1e-6)))
    return Subset(ds, list(range(0, len(ds), step)))


def all_reduce_sum(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def all_reduce_grads_(module: torch.nn.Module, dist_info: Dict[str, int]) -> None:
    """Average ``module``'s gradients across DDP ranks in-place (no-op single proc).

    Needed for the WGAN-GP critic: its gradient penalty is a double-backward run
    on the *unwrapped* module (DDP does not support the GP create_graph backward),
    so those grads are never auto-synced. We all-reduce the whole critic grad once
    after backward so every rank applies an identical critic update.
    """
    if not dist_info["distributed"]:
        return
    ws = max(1, dist_info["world_size"])
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= ws


def broadcast_str(s: str, dist_info: Dict[str, int]) -> str:
    if not dist_info["distributed"]:
        return s
    obj = [s]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


# ---------------------------------------------------------------------------
# run dir + swanlab
# ---------------------------------------------------------------------------
def make_run_dir(cfg, dist_info: Dict[str, int]) -> str:
    resume = str(cfg.training.resume_dir)
    if resume:
        return resume
    name = str(cfg.training.run_name) or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(cfg.training.output_dir), name)
    run_dir = broadcast_str(run_dir, dist_info)
    if is_main_process():
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def init_swanlab(cfg):
    if not bool(cfg.training.use_swanlab) or not is_main_process():
        return None
    try:
        import swanlab
    except ImportError:
        logger.warning("swanlab not installed; metric logging disabled")
        return None
    from omegaconf import OmegaConf
    run = swanlab.init(
        project=str(cfg.training.swanlab_project),
        experiment_name=(str(cfg.training.run_name) or None),
        description=f"baseline {cfg.model.name} {cfg.baseline.direction}",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=str(cfg.training.swanlab_mode),
    )
    return run


def sw_log(run, data: Dict[str, float], step: int) -> None:
    if run is not None and is_main_process():
        run.log(data, step=step)


# ---------------------------------------------------------------------------
# validation: mean waveform MAE (mmHg for ->ABP, normalized for bridge)
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_mae(
    model: BaselineModule,
    loader: DataLoader,
    cfg,
    device: torch.device,
    *,
    gan_predict=None,
) -> float:
    """DDP-exact mean |pred - true| over the val shard. Lower is better.

    ``gan_predict``: optional callable(x)->wave for the GAN generator; else uses
    ``model(x, want_bp=...)``.
    """
    model.eval()
    direction = str(cfg.baseline.direction)
    tgt_is_abp = direction.endswith("2abp")
    abp_mode = ABP_TARGET_MODE.get(str(cfg.model.name), "global")
    want_bp = bool(getattr(model, "has_bp_head", False)) and tgt_is_abp
    gan_tanh = str(cfg.model.name) == "p2e_wgan"
    clip_lo = float(cfg.data.abp_clip_low)
    clip_hi = float(cfg.data.abp_clip_high)
    seq_len = int(cfg.data.seq_len)
    work_mult = int(getattr(model, "work_multiple", 1))

    sum_abs = 0.0
    n_elem = 0.0
    for batch in loader:
        x = pad_to_multiple(batch["x"].to(device), work_mult)
        if gan_predict is not None:
            wave = gan_predict(x)
            bp_pred = None
        else:
            out = model(x, want_bp=want_bp)
            wave = out["wave"]
            bp_pred = out.get("bp") if want_bp else None
        pred = reconstruct_pred(
            wave, seq_len=seq_len, tgt_is_abp=tgt_is_abp, abp_mode=abp_mode,
            clip_lo=clip_lo, clip_hi=clip_hi, bp_pred=bp_pred, gan_tanh=gan_tanh,
        ).cpu()
        true = reconstruct_true(batch, tgt_is_abp=tgt_is_abp).cpu()
        sum_abs += float((pred - true).abs().sum())
        n_elem += float(pred.numel())

    sum_abs = all_reduce_sum(sum_abs, device)
    n_elem = all_reduce_sum(n_elem, device)
    return sum_abs / max(n_elem, 1.0)


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model: BaselineModule, optimizer, epoch: int, best_val: float, cfg,
                    extra: Optional[Dict[str, object]] = None) -> None:
    from omegaconf import OmegaConf
    raw = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "abp_clip_low": float(cfg.data.abp_clip_low),
        "abp_clip_high": float(cfg.data.abp_clip_high),
    }
    if extra:  # e.g. the GAN trainer adds the discriminator optimizer state
        payload.update(extra)
    torch.save(payload, path)
