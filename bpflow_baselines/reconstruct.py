"""Reconstruct mmHg ABP (or normalized bridge waveform) from model outputs.

Shared by the trainer's validation and infer.py so val-MAE and final metrics use
exactly the same path. Evaluation always lands in bpflow's spaces:
- ->ABP: mmHg, compared against ``abp_raw`` via ``bpflow.eval.evaluate``.
- bridge: [0,1] (recenter undone), via ``bpflow.eval.waveform_metrics``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .models.base import crop_center
from .norms import (
    abp_global_denorm,
    abp_perslot_denorm,
    bridge_denorm,
)


def reconstruct_pred(
    wave: torch.Tensor,
    *,
    seq_len: int,
    tgt_is_abp: bool,
    abp_mode: str,
    clip_lo: float,
    clip_hi: float,
    bp_pred: Optional[torch.Tensor] = None,
    gan_tanh: bool = False,
) -> torch.Tensor:
    """Model wave output (B,1,Lw) -> prediction (B, L).

    ->ABP: mmHg. bridge: [0,1]. ``bp_pred`` (B,2)=(SBP,DBP) needed for perslot.
    ``gan_tanh`` maps a Tanh [-1,1] output to [0,1] before global de-norm.
    """
    w = crop_center(wave, seq_len).squeeze(1)  # (B, L)
    if not tgt_is_abp:
        # bridge target lives in [0,1]. A Tanh generator (p2e_wgan) emits [-1,1]
        # -> map directly to [0,1]; a linear model emits recenter [-0.5,0.5] -> +0.5.
        if gan_tanh:
            return (w + 1.0) * 0.5
        return bridge_denorm(w)
    if abp_mode == "global":
        if gan_tanh:
            w = (w + 1.0) * 0.5
        return abp_global_denorm(w, clip_lo, clip_hi)
    # perslot two-stage: bp_pred is (SBP, DBP) in global-min-max [0,1] -> mmHg
    if bp_pred is None:
        raise ValueError("perslot reconstruction needs bp_pred (SBP, DBP)")
    sbp_mmhg = bp_pred[:, 0] * (clip_hi - clip_lo) + clip_lo
    dbp_mmhg = bp_pred[:, 1] * (clip_hi - clip_lo) + clip_lo
    return abp_perslot_denorm(w, sbp_mmhg, dbp_mmhg)


def reconstruct_true(batch, *, tgt_is_abp: bool) -> torch.Tensor:
    """Ground truth aligned with reconstruct_pred. ->ABP: abp_raw (mmHg).
    bridge: recenter target undone -> [0,1] (matches bpflow infer)."""
    if tgt_is_abp:
        return batch["abp_raw"]
    return bridge_denorm(batch["y"].squeeze(1))
