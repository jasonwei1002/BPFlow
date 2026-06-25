"""WGAN-GP trainer for P2E-WGAN (conditional Wasserstein GAN + gradient penalty).

Paper recipe: Adam(betas=(0.5,0.999)), n_critic=3, lambda_gp=10, generator gets an
extra lambda_sample * MSE(fake, real) reconstruction term (lambda_sample=50).
The conditional critic sees cat(source, waveform). Target is mapped to the
generator's Tanh range [-1,1] (ABP global [0,1] -> 2x-1; bridge recenter is
already within [-1,1]).
"""

from __future__ import annotations

import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bpflow.trainer_utils import is_main_process, pick_device, set_seed

from . import engine
from .data import build_baseline_dataset
from .models.base import pad_to_multiple
from .models import build_model

logger = logging.getLogger(__name__)


def _gradient_penalty(critic, cond, real, fake, device):
    b = real.shape[0]
    eps = torch.rand(b, 1, 1, device=device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(torch.cat([cond, inter], dim=1))
    grad = torch.autograd.grad(
        outputs=score, inputs=inter,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.reshape(b, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def _make_loader(ds, batch_size, shuffle, num_workers, dist_info, seed):
    sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed) if dist_info["distributed"] else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=shuffle, persistent_workers=num_workers > 0,
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

    model = build_model(cfg).to(device)
    gen_raw, critic_raw = model.generator, model.discriminator
    work_mult = int(model.work_multiple)

    if str(cfg.training.init_from_ckpt):
        ckpt = torch.load(str(cfg.training.init_from_ckpt), map_location="cpu", weights_only=False)
        engine.load_state(model, ckpt["model"])

    gen = gen_raw
    if dist_info["distributed"]:
        ids = [dist_info["local_rank"]] if torch.cuda.is_available() else None
        gen = DDP(gen_raw, device_ids=ids)
    # The critic is deliberately NOT DDP-wrapped: WGAN-GP's gradient penalty is a
    # double-backward (create_graph) that DDP cannot sync. We run the whole critic
    # step on critic_raw and all-reduce its grads manually (engine.all_reduce_grads_),
    # so the GP term is averaged across ranks too and every rank's critic stays
    # identical.

    seed = int(cfg.training.seed)
    # batch_size is the GLOBAL batch (paper value); split across DDP ranks.
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

    betas = (float(cfg.training.gan_beta1), float(cfg.training.gan_beta2))
    opt_g = torch.optim.Adam(gen_raw.parameters(), lr=float(cfg.training.lr), betas=betas)
    opt_d = torch.optim.Adam(critic_raw.parameters(), lr=float(cfg.training.lr), betas=betas)

    n_critic = int(cfg.training.n_critic)
    lam_gp = float(cfg.training.lambda_gp)
    lam_s = float(cfg.training.lambda_sample)
    max_steps = int(cfg.training.max_steps)

    run_dir = engine.make_run_dir(cfg, dist_info)
    sw = engine.init_swanlab(cfg)
    if is_main_process():
        logger.info("P2E-WGAN direction=%s run_dir=%s N_train=%d", direction, run_dir, len(train_ds))

    def gen_predict(x):
        return gen_raw(x)

    def to_tanh(y):
        # Map the target into the generator's Tanh [-1,1] range.
        # ABP: global-min-max [0,1] -> 2y-1. Bridge: bpflow recenter [-0.5,0.5] -> 2y.
        return 2.0 * y - 1.0 if tgt_is_abp else 2.0 * y

    global_step = 0
    best_val = float("inf")
    epochs_no_improve = 0
    crit_iter = 0
    for epoch in range(int(cfg.training.epochs)):
        gen.train(); critic_raw.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            x = pad_to_multiple(batch["x"].to(device), work_mult)
            real = pad_to_multiple(to_tanh(batch["y"].to(device)), work_mult)

            # ---- critic step (unwrapped critic_raw + manual grad all-reduce) ----
            with torch.no_grad():
                fake = gen_raw(x)
            d_real = critic_raw(torch.cat([x, real], dim=1)).mean()
            d_fake = critic_raw(torch.cat([x, fake], dim=1)).mean()
            gp = _gradient_penalty(critic_raw, x, real, fake, device)
            loss_d = d_fake - d_real + lam_gp * gp
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            engine.all_reduce_grads_(critic_raw, dist_info)  # sync incl. the GP term
            opt_d.step()
            crit_iter += 1

            # ---- generator step (every n_critic critic steps) ----
            if crit_iter % n_critic == 0:
                fake = gen(x)
                # freeze critic grads so G backward doesn't trip critic's DDP hooks
                for p in critic_raw.parameters():
                    p.requires_grad_(False)
                adv = -critic_raw(torch.cat([x, fake], dim=1)).mean()
                mse = torch.nn.functional.mse_loss(fake, real)
                loss_g = adv + lam_s * mse
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                for p in critic_raw.parameters():
                    p.requires_grad_(True)
                if is_main_process() and global_step % int(cfg.training.log_freq) == 0:
                    logger.info("ep%d step%d D=%.4f G=%.4f mse=%.5f", epoch, global_step,
                                float(loss_d), float(loss_g), float(mse))
                    engine.sw_log(sw, {"train/D": float(loss_d), "train/G": float(loss_g),
                                       "train/mse": float(mse)}, global_step)
            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                break

        if int(cfg.training.val_freq_epoch) > 0 and (epoch + 1) % int(cfg.training.val_freq_epoch) == 0:
            val_mae = engine.validate_mae(model, val_loader, cfg, device, gan_predict=gen_predict)
            # best_val / epochs_no_improve are tracked on ALL ranks (val_mae is the
            # same everywhere) so the early-stop decision is identical and ranks
            # break together — never deadlock on the next collective.
            improved = val_mae < best_val
            significant = val_mae < best_val - float(cfg.training.min_delta)
            if improved:
                best_val = val_mae
            epochs_no_improve = 0 if significant else epochs_no_improve + 1
            ckpt_extra = {"opt_d": opt_d.state_dict()}  # also save the critic optimizer
            if is_main_process():
                logger.info("[val] ep%d MAE=%.4f (best=%.4f)", epoch, val_mae, best_val)
                engine.sw_log(sw, {"val/MAE": val_mae}, global_step)
                if improved:
                    engine.save_checkpoint(os.path.join(run_dir, "checkpoint_best.pth"),
                                           model, opt_g, epoch, best_val, cfg, extra=ckpt_extra)
                engine.save_checkpoint(os.path.join(run_dir, "checkpoint_latest.pth"),
                                       model, opt_g, epoch, best_val, cfg, extra=ckpt_extra)
            es = int(cfg.training.early_stop_patience)
            stop = engine.broadcast_str("1" if (es > 0 and epochs_no_improve >= es) else "0", dist_info) == "1"
            if stop:
                if is_main_process():
                    logger.info("early stop at epoch %d (no improve %d)", epoch, epochs_no_improve)
                break
        if max_steps > 0 and global_step >= max_steps:
            break

    if is_main_process():
        engine.save_checkpoint(os.path.join(run_dir, "checkpoint_latest.pth"),
                               model, opt_g, int(cfg.training.epochs) - 1, best_val, cfg,
                               extra={"opt_d": opt_d.state_dict()})
        if sw is not None:
            sw.finish()
        logger.info("done. best val MAE=%.4f run_dir=%s", best_val, run_dir)
    engine.cleanup_distributed(dist_info)
    return run_dir
