"""Training losses for the baselines (paper recipes).

- Plain / deep-supervision MSE on the stage-1 waveform (operates in the padded
  Lw space; aux outputs are interpolated to Lw and weighted with a linear decay).
- L1 on the stage-2 (SBP, DBP) head for the two-stage ->ABP models.
- WGAN-GP pieces live in gan_trainer.py (P2E-WGAN).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def _aux_weights(n: int) -> List[float]:
    """Linear-decay deep-supervision weights [1.0, 0.9, 0.8, ...] (min 0.1)."""
    return [max(1.0 - 0.1 * i, 0.1) for i in range(n)]


def _base_loss(name: str):
    """Pick the per-paper waveform base loss: 'mae' (L1) or 'mse' (L2)."""
    if name == "mae":
        return F.l1_loss
    if name == "mse":
        return F.mse_loss
    raise ValueError(f"loss base must be 'mae' or 'mse', got {name!r}")


def waveform_loss(out: Dict[str, object], y: torch.Tensor,
                  base: str = "mse", aux_base: str = "") -> torch.Tensor:
    """Primary-wave loss + deep-supervision loss on aux outputs.

    ``base`` is the loss for the primary output (e.g. PPG2ABP refinement = MSE);
    ``aux_base`` is the loss for the deep-supervision aux outputs (e.g. PPG2ABP
    approximation = MAE); empty ``aux_base`` reuses ``base``. ``out["wave"]`` and
    ``y`` are (B,1,Lw); aux entries are interpolated to Lw. Aux weights linear-decay.
    """
    wave = out["wave"]
    assert isinstance(wave, torch.Tensor)
    prim = _base_loss(base)
    loss = prim(wave, y)
    aux = out.get("wave_aux") or []
    assert isinstance(aux, list)
    if aux:
        afn = _base_loss(aux_base or base)
        weights = _aux_weights(len(aux))
        Lw = y.shape[-1]
        for w, a in zip(weights, aux):
            assert isinstance(a, torch.Tensor)
            if a.shape[-1] != Lw:
                a = F.interpolate(a, size=Lw, mode="linear", align_corners=False)
            loss = loss + w * afn(a, y)
    return loss


def bp_l1(bp_pred: torch.Tensor, sbp: torch.Tensor, dbp: torch.Tensor,
          lo: float, hi: float) -> torch.Tensor:
    """L1 loss on predicted (SBP, DBP), in global-min-max [0,1] space.

    Faithful to MD-ViSCo, whose refinement BP target is ``global_minmax`` over
    the dataset bounds (here the clip bounds [lo, hi]) -> the head regresses in
    [0,1], keeping this loss on the same scale as the waveform MSE (mmHg-scale
    targets would swamp it). ``bp_pred`` is (B,2)=(SBP,DBP) in [0,1]; ``sbp``/
    ``dbp`` are per-sample clip(ABP) max/min in mmHg.
    """
    target = (torch.stack([sbp, dbp], dim=-1) - lo) / (hi - lo)
    return F.l1_loss(bp_pred, target)
