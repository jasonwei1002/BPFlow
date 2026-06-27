"""Supervised trainer for the non-GAN baselines (Adam + paper recipe).

Handles single-stage / cascade / two-stage(+BP head) models uniformly. DDP-aware,
validates with mean waveform MAE (mmHg for ->ABP, normalized for bridge), and
selects best by val MAE with optional ReduceLROnPlateau + early stopping.
P2E-WGAN uses gan_trainer.py instead.
"""

from __future__ import annotations

import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bpflow.trainer_utils import (
    add_weight_decay,
    is_main_process,
    pick_device,
    set_seed,
)

from . import engine
from .data import build_baseline_dataset
from .losses import bp_l1, waveform_loss
from .models.base import build_model, pad_to_multiple

logger = logging.getLogger(__name__)


def _make_loader(ds, batch_size, shuffle, num_workers, dist_info, seed):
    # drop_last follows the original `shuffle` intent (train=True drops the partial
    # last batch; val=False keeps all). Capture it BEFORE reassigning `shuffle`
    # below, else under DDP the train loader would keep a size-1 final batch and
    # crash BatchNorm (nabnet/ppg2abp) / break the fixed-batch assumption.
    drop_last = shuffle
    sampler = None
    if dist_info["distributed"]:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=drop_last)
        shuffle = False
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=drop_last, persistent_workers=num_workers > 0,
    ), sampler


def train(cfg) -> str:
    dist_info = engine.setup_distributed()
    device = pick_device(str(cfg.training.device))
    if dist_info["distributed"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_info['local_rank']}")
    set_seed(int(cfg.training.seed) + dist_info["rank"])

    direction = str(cfg.baseline.direction)
    tgt_is_abp = direction.endswith("2abp")

    if not is_main_process():
        logging.getLogger().setLevel(logging.WARNING)

    # model
    model = build_model(cfg).to(device)
    want_bp = bool(model.has_bp_head) and tgt_is_abp

    if str(cfg.training.init_from_ckpt):
        ckpt = torch.load(str(cfg.training.init_from_ckpt), map_location="cpu", weights_only=False)
        engine.load_state(model, ckpt["model"])
        if is_main_process():
            logger.info("init weights from %s", cfg.training.init_from_ckpt)

    ddp_model = model
    if dist_info["distributed"]:
        find_unused = bool(model.has_bp_head) and not tgt_is_abp  # bridge: bp head idle
        device_ids = [dist_info["local_rank"]] if torch.cuda.is_available() else None
        ddp_model = DDP(model, device_ids=device_ids, find_unused_parameters=find_unused)

    # data
    seed = int(cfg.training.seed)
    # batch_size is the GLOBAL batch (paper value); split it across DDP ranks so
    # the effective global batch matches the paper regardless of GPU count.
    per_gpu_bs = max(1, int(cfg.training.batch_size) // max(1, dist_info["world_size"]))
    if is_main_process() and dist_info["world_size"] > 1:
        logger.info("global batch %d over %d ranks -> %d per GPU",
                    int(cfg.training.batch_size), dist_info["world_size"], per_gpu_bs)
    train_ds = build_baseline_dataset(cfg, "train")
    val_ds = engine.subsample_val(build_baseline_dataset(cfg, "val"), float(cfg.training.val_eval_fraction))
    train_loader, train_sampler = _make_loader(
        train_ds, per_gpu_bs, True, int(cfg.training.num_workers), dist_info, seed)
    val_loader, _ = _make_loader(
        val_ds, int(cfg.training.val_batch_size), False, int(cfg.training.num_workers), dist_info, seed)

    # optimizer + scheduler
    groups = add_weight_decay(model, float(cfg.training.weight_decay))
    opt = torch.optim.Adam(
        groups, lr=float(cfg.training.lr),
        betas=(float(cfg.training.beta1), float(cfg.training.beta2)))
    best_val = float("inf")
    epochs_no_improve = 0
    lr_scale = 1.0
    work_mult = int(model.work_multiple)
    lambda_bp = float(cfg.training.lambda_bp)
    clip_grad = float(cfg.training.clip_grad_norm)
    max_steps = int(cfg.training.max_steps)
    loss_base = str(cfg.training.loss_base)
    aux_loss_base = str(cfg.training.aux_loss_base)
    # OneCycleLR (PatchTST): per-step schedule over the whole run.
    onecycle = None
    if str(cfg.training.scheduler) == "onecycle":
        spe = max(1, len(train_loader))
        total = max_steps if max_steps > 0 else int(cfg.training.epochs) * spe
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=float(cfg.training.lr), total_steps=total,
            pct_start=float(cfg.training.onecycle_pct_start))

    run_dir = engine.make_run_dir(cfg, dist_info)
    sw = engine.init_swanlab(cfg, run_dir)
    if is_main_process():
        logger.info("baseline=%s direction=%s run_dir=%s want_bp=%s N_train=%d N_val=%d",
                    cfg.model.name, direction, run_dir, want_bp, len(train_ds), len(val_ds))

    global_step = 0
    for epoch in range(int(cfg.training.epochs)):
        ddp_model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            x = pad_to_multiple(batch["x"].to(device), work_mult)
            y = pad_to_multiple(batch["y"].to(device), work_mult)
            out = ddp_model(x, want_bp=want_bp)
            loss = waveform_loss(out, y, base=loss_base, aux_base=aux_loss_base)
            if want_bp:
                loss = loss + lambda_bp * bp_l1(
                    out["bp"], batch["sbp"].to(device), batch["dbp"].to(device),
                    float(cfg.data.abp_clip_low), float(cfg.data.abp_clip_high))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            if onecycle is not None:
                onecycle.step()
            global_step += 1
            if is_main_process() and global_step % int(cfg.training.log_freq) == 0:
                logger.info("ep%d step%d loss=%.5f", epoch, global_step, float(loss))
                engine.sw_log(sw, {"train/loss": float(loss), "train/lr": opt.param_groups[0]["lr"]}, global_step)
            if max_steps > 0 and global_step >= max_steps:
                break

        # validation
        if int(cfg.training.val_freq_epoch) > 0 and (epoch + 1) % int(cfg.training.val_freq_epoch) == 0:
            val_mae = engine.validate_mae(model, val_loader, cfg, device)
            improved = val_mae < best_val
            significant = val_mae < best_val - float(cfg.training.min_delta)
            if is_main_process():
                logger.info("[val] ep%d MAE=%.4f (best=%.4f)", epoch, val_mae, best_val)
                engine.sw_log(sw, {"val/MAE": val_mae}, global_step)
            if improved:
                best_val = val_mae
                if is_main_process():
                    engine.save_checkpoint(os.path.join(run_dir, "checkpoint_best.pth"),
                                           model, opt, epoch, best_val, cfg)
            epochs_no_improve = 0 if significant else epochs_no_improve + 1
            # plateau decay
            if str(cfg.training.scheduler) == "plateau" and int(cfg.training.lr_patience) > 0 \
                    and epochs_no_improve > 0 and epochs_no_improve % int(cfg.training.lr_patience) == 0:
                lr_scale *= float(cfg.training.lr_decay)
                for pg in opt.param_groups:
                    pg["lr"] = float(cfg.training.lr) * lr_scale
                if is_main_process():
                    logger.info("plateau: lr_scale=%.2e", lr_scale)
            # early stop
            es = int(cfg.training.early_stop_patience)
            stop = es > 0 and epochs_no_improve >= es
            stop = engine.broadcast_str("1" if stop else "0", dist_info) == "1"
            if is_main_process():
                engine.save_checkpoint(os.path.join(run_dir, "checkpoint_latest.pth"),
                                       model, opt, epoch, best_val, cfg)
            if stop:
                if is_main_process():
                    logger.info("early stop at epoch %d (no improve %d)", epoch, epochs_no_improve)
                break
        if max_steps > 0 and global_step >= max_steps:
            break

    if is_main_process():
        engine.save_checkpoint(os.path.join(run_dir, "checkpoint_latest.pth"),
                               model, opt, int(cfg.training.epochs) - 1, best_val, cfg)
        if sw is not None:
            sw.finish()
        logger.info("done. best val MAE=%.4f run_dir=%s", best_val, run_dir)
    engine.cleanup_distributed(dist_info)
    return run_dir
