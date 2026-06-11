"""Shared flow-matching helpers used by trainer / infer / smoke_test.

Factored out to keep a single definition of (a) how a FlowMatching is built
from config, (b) the ECG+PPG -> ABP sampling routine, and (c) the training
loss step -- previously duplicated across three modules.
"""

from contextlib import nullcontext
from typing import Optional

import torch

from .data import destandardize_abp, unpatchify
from .model import FlowMatching, log_normal_sample


def build_flow_matching(cfg) -> FlowMatching:
    """Construct FlowMatching from the shared ``cfg.sampling`` block."""
    s = cfg.sampling
    return FlowMatching(
        min_sigma=float(s.min_sigma),
        inference_mode=str(s.method),
        num_steps=int(s.num_steps),
        prediction_type=str(s.prediction_type),
        noise_scale=float(s.noise_scale),
        noise_shift=float(s.noise_shift),
    )


def _demo_to(demo, device: torch.device):
    """Move a (cont, gender) demographics sample to ``device``, or pass None."""
    if demo is None:
        return None
    cont, gender = demo
    return cont.to(device), gender.to(device)


@torch.no_grad()
def sample_abp(
    model: torch.nn.Module,
    fm: FlowMatching,
    cond_patches: torch.Tensor,
    *,
    generator: torch.Generator,
    device: torch.device,
    abp_mean: float,
    abp_std: float,
    autocast_ctx=None,
    demo=None,
    context=None,
) -> torch.Tensor:
    """Sample ABP waveforms (mmHg, shape (B, L)) from ECG+PPG condition patches.

    Assumes ``model`` is the raw (unwrapped) model already in the desired
    eval/param state. ``demo`` is an optional (cont, gender) demographics sample
    and ``context`` the optional per-subject ANIL context (already adapted),
    both used as global priors.
    """
    cond_patches = cond_patches.to(device)
    demo = _demo_to(demo, device)
    if context is not None:
        context = context.to(device)
    bs = cond_patches.shape[0]
    conditions = model.preprocess_conditions(cond_patches, demo, context)
    x0 = (
        torch.randn(
            bs, model.latent_seq_len, model.latent_dim, generator=generator, device=device
        )
        * fm.noise_scale
    )
    fn = lambda t, x: model.ode_wrapper(t, x, conditions)
    with (autocast_ctx if autocast_ctx is not None else nullcontext()):
        x1 = fm.to_data(fn, x0)
    wave = unpatchify(x1).float().cpu()
    return destandardize_abp(wave, abp_mean, abp_std)


def flow_matching_loss(
    model: torch.nn.Module,
    fm: FlowMatching,
    abp_patches: torch.Tensor,
    cond_patches: torch.Tensor,
    *,
    generator: torch.Generator,
    logit_mean: float,
    logit_scale: float,
    prediction_type: str,
    loss_type: str,
    demo=None,
    context=None,
) -> torch.Tensor:
    """One flow-matching training step: noise -> predict -> loss (mean scalar).

    ``model`` is called for the forward, so pass the DDP-wrapped module when
    training under DDP (gradient sync) and the raw module otherwise. ``demo`` is
    an optional (cont, gender) demographics sample and ``context`` the optional
    per-subject ANIL context vector (both already on the right device). Grads
    flow into ``context``, so the meta inner loop differentiates the loss wrt it.
    """
    t = log_normal_sample(abp_patches, generator=generator, m=logit_mean, s=logit_scale)
    x0, x1, xt, t_sh = fm.get_x0_xt_c(abp_patches, t, generator=generator)
    pred = model(xt, cond_patches, t_sh, demo, context)
    return fm.loss(prediction_type, loss_type, pred, x0, xt, x1, t_sh).mean()
