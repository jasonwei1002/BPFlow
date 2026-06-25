"""Normalization + reconstruction helpers for the baseline reproduction.

All baselines consume bpflow's data (identical splits/segments) but follow each
paper's own *target* normalization (see plan/baselines-repro/design.md):

- ECG/PPG inputs and bridge targets: bpflow recenter scale (raw [0,1] - 0.5).
- ABP target, "global" scheme (PPG2ABP / P2E-WGAN / WaveNet):
    global min-max over the fixed clip bounds [clip_low, clip_high] -> [0,1],
    invertible by ``x * (hi - lo) + lo``.
- ABP target, "perslot" scheme (NABNet / PatchTST / MD-ViSCo two-stage):
    per-sample min-max -> [0,1] (shape only), reconstructed with the per-sample
    SBP (= clip(abp).max) / DBP (= clip(abp).min) predicted by the stage-2 head:
    ``w01 * (SBP - DBP) + DBP``.

Evaluation always reconstructs to mmHg and compares against ``abp_raw`` via
``bpflow.eval.evaluate`` so the test metrics match bpflow exactly.
"""

from __future__ import annotations

from typing import Tuple

import torch

# modality index convention shared with bpflow target_idx: abp=0, ecg=1, ppg=2
MOD_IDX = {"abp": 0, "ecg": 1, "ppg": 2}

# direction -> (source modality(ies), target modality). The source is a single
# modality string for 1->1 directions, or a tuple for multi-source directions
# (channel-stacked: ecg_ppg2abp feeds [ECG; PPG] as a 2-channel input, ECG=ch0,
# PPG=ch1 per bpflow's convention).
DIRECTIONS = {
    "ecg2abp": ("ecg", "abp"),
    "ppg2abp": ("ppg", "abp"),
    "ecg2ppg": ("ecg", "ppg"),
    "ppg2ecg": ("ppg", "ecg"),
    "ecg_ppg2abp": (("ecg", "ppg"), "abp"),
}


def direction_sources(direction: str) -> Tuple[str, ...]:
    """Source modalities of a direction, always returned as a tuple (>=1)."""
    src, _ = DIRECTIONS[direction]
    return src if isinstance(src, tuple) else (src,)


def num_source_channels(direction: str) -> int:
    """Number of channel-stacked input channels a direction feeds the model."""
    return len(direction_sources(direction))

# ABP target normalization scheme per model family
ABP_TARGET_MODE = {
    "ppg2abp": "global",
    "p2e_wgan": "global",
    "wavenet": "global",
    "nabnet": "perslot",
    "patchtst": "perslot",
    "mdvisco": "perslot",
}

RECENTER_SHIFT = 0.5  # bpflow ECG/PPG recenter offset (cond_recenter)


def clip_abp(abp: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clip raw mmHg ABP to bpflow's [lo, hi] data range."""
    return abp.clamp(lo, hi)


def abp_global_norm(abp_mmhg: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Global min-max ABP -> [0,1] using fixed clip bounds. Forward."""
    return (clip_abp(abp_mmhg, lo, hi) - lo) / (hi - lo)


def abp_global_denorm(x01: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Inverse of :func:`abp_global_norm` -> mmHg."""
    return x01 * (hi - lo) + lo


def abp_perslot_norm(abp_mmhg: torch.Tensor, lo: float, hi: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample min-max ABP -> [0,1] shape + (sbp, dbp) mmHg bounds.

    Operates over the last (time) dim. ``abp_mmhg`` is (..., L); returns
    (w01 (...,L), sbp (...), dbp (...)). Bounds are taken on the clipped wave so
    reconstruction lands in the same [lo, hi] range bpflow evaluates against.
    """
    abp_c = clip_abp(abp_mmhg, lo, hi)
    dbp = abp_c.amin(dim=-1)
    sbp = abp_c.amax(dim=-1)
    denom = (sbp - dbp).clamp_min(1e-6).unsqueeze(-1)
    w01 = (abp_c - dbp.unsqueeze(-1)) / denom
    return w01, sbp, dbp


def abp_perslot_denorm(w01: torch.Tensor, sbp: torch.Tensor, dbp: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`abp_perslot_norm` using predicted per-sample SBP/DBP.

    ``sbp``/``dbp`` are (...,) and broadcast over the time dim of ``w01`` (...,L).
    Guards against SBP < DBP ordering (matches MD-ViSCo ``_unscale_waveform``).
    """
    hi = torch.maximum(sbp, dbp).unsqueeze(-1)
    lo = torch.minimum(sbp, dbp).unsqueeze(-1)
    return w01 * (hi - lo) + lo


def bridge_denorm(x_recenter: torch.Tensor) -> torch.Tensor:
    """Undo recenter for bridge (ECG/PPG) -> [0,1], matching bpflow bridge eval."""
    return x_recenter + RECENTER_SHIFT
