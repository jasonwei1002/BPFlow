"""Shared flow-matching helpers used by trainer / infer / smoke_test.

Factored out to keep a single definition of (a) how a FlowMatching is built
from config, (b) the ECG+PPG -> ABP sampling routine, and (c) the training
loss step -- previously duplicated across three modules.
"""

from contextlib import nullcontext

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


def _select_target(abp_patches, ecg_patches, ppg_patches, target_idx):
    """Gather each sample's TARGET-modality clean patches (B, N, P) by target_idx
    (0=ABP, 1=ECG, 2=PPG) from the three per-modality patch tensors."""
    b, n, p = abp_patches.shape
    allp = torch.stack([abp_patches, ecg_patches, ppg_patches], dim=1)  # (B,3,N,P)
    gi = target_idx.view(b, 1, 1, 1).expand(b, 1, n, p)
    return allp.gather(1, gi).squeeze(1)


@torch.no_grad()
def sample_target(
    model: torch.nn.Module,
    fm: FlowMatching,
    ecg_patches: torch.Tensor,
    ppg_patches: torch.Tensor,
    target_idx: torch.Tensor,
    cond_present: torch.Tensor,
    *,
    generator: torch.Generator,
    device: torch.device,
    abp_mean: float,
    abp_std: float,
    cond_recenter: bool = True,
    autocast_ctx=None,
    demo=None,
) -> torch.Tensor:
    """Sample the TARGET waveform (shape (B, L)) for each sample's task.

    ``model`` is the raw (unwrapped) model in the desired eval/param state.
    ``target_idx`` (B,) and ``cond_present`` (B, 3) define the task; ECG/PPG clean
    condition patches are always passed (gated inside the model). Output is
    de-normalized per the target modality: ABP -> mmHg (z-score inverse), ECG/PPG
    -> undo the -0.5 recenter. Handles a mixed-target batch, though eval/infer
    force a single direction.
    """
    ecg_patches = ecg_patches.to(device)
    ppg_patches = ppg_patches.to(device)
    target_idx = target_idx.to(device).long()
    cond_present = cond_present.to(device)
    demo = _demo_to(demo, device)
    bs = ecg_patches.shape[0]
    conditions = model.preprocess_conditions(ecg_patches, ppg_patches, target_idx, cond_present, demo)
    x0 = (
        torch.randn(bs, model.latent_seq_len, model.latent_dim, generator=generator, device=device)
        * fm.noise_scale
    )
    fn = lambda t, x: model.ode_wrapper(t, x, conditions)
    with (autocast_ctx if autocast_ctx is not None else nullcontext()):
        x1 = fm.to_data(fn, x0)
    wave = unpatchify(x1).float().cpu()  # (B, L) in normalized space
    ti = target_idx.cpu()
    abp_wave = destandardize_abp(wave, abp_mean, abp_std)
    other_wave = wave + 0.5 if cond_recenter else wave  # ECG/PPG recenter inverse
    is_abp = (ti == 0).view(-1, 1)
    return torch.where(is_abp, abp_wave, other_wave)


def flow_matching_loss(
    model: torch.nn.Module,
    fm: FlowMatching,
    abp_patches: torch.Tensor,
    ecg_patches: torch.Tensor,
    ppg_patches: torch.Tensor,
    target_idx: torch.Tensor,
    cond_present: torch.Tensor,
    *,
    generator: torch.Generator,
    logit_mean: float,
    logit_scale: float,
    prediction_type: str,
    loss_type: str,
    demo=None,
    per_sample: bool = False,
) -> torch.Tensor:
    """One flow-matching training step for a per-sample TARGET: select target ->
    noise -> predict -> loss.

    ``model`` is called for the forward, so pass the DDP-wrapped module under DDP.
    ``target_idx`` (B,) picks each sample's target modality; ``cond_present`` (B, 3)
    its conditions. Returns the mean scalar; ``per_sample=True`` returns the (B,)
    per-sample loss (to decompose train loss by task without an extra forward).
    """
    target_patches = _select_target(abp_patches, ecg_patches, ppg_patches, target_idx)
    t = log_normal_sample(target_patches, generator=generator, m=logit_mean, s=logit_scale)
    x0, x1, xt, t_sh = fm.get_x0_xt_c(target_patches, t, generator=generator)
    pred = model(xt, ecg_patches, ppg_patches, t_sh, target_idx, cond_present, demo)
    per = fm.loss(prediction_type, loss_type, pred, x0, xt, x1, t_sh)  # (B,)
    return per if per_sample else per.mean()
