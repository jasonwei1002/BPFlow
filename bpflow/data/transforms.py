"""Patching + normalization transforms for BPFlow signals.

All functions operate on the LAST dim being the time axis and are torch-only.
They accept an arbitrary batch prefix, e.g. (L,) or (B, L).

Normalization contract (see plan/notes.md):
- ABP target: physiological clip THEN fixed-constant z-score with stored
  global ABP_MEAN/ABP_STD (NOT per-sample, which would leak the answer).
- ECG/PPG condition: already in [0,1]; optionally recenter by -0.5. No clip.
"""

import torch


def standardize_abp(
    abp: torch.Tensor, mean: float, std: float, clip_low: float, clip_high: float
) -> torch.Tensor:
    """Clip ABP to a physiological range, then z-score with global constants."""
    abp = torch.clamp(abp, clip_low, clip_high)
    return (abp - mean) / std


def destandardize_abp(norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Inverse of :func:`standardize_abp` (no re-clip)."""
    return norm * std + mean


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(..., L) -> (..., N, P) with N = L // patch_size. Requires L % P == 0."""
    *prefix, length = x.shape
    if length % patch_size != 0:
        raise ValueError(f"length {length} not divisible by patch_size {patch_size}")
    n = length // patch_size
    return x.reshape(*prefix, n, patch_size)


def unpatchify(p: torch.Tensor) -> torch.Tensor:
    """(..., N, P) -> (..., L) with L = N * P."""
    *prefix, n, patch_size = p.shape
    return p.reshape(*prefix, n * patch_size)


def build_cond_patches(
    ecg: torch.Tensor, ppg: torch.Tensor, patch_size: int, recenter: bool = True
) -> torch.Tensor:
    """Stack ECG+PPG into channel-major condition patches.

    (..., L), (..., L) -> (..., N, 2P): per token, P ECG samples then P PPG.
    """
    if recenter:
        ecg = ecg - 0.5
        ppg = ppg - 0.5
    ecg_p = patchify(ecg, patch_size)  # (..., N, P)
    ppg_p = patchify(ppg, patch_size)  # (..., N, P)
    return torch.cat([ecg_p, ppg_p], dim=-1)  # (..., N, 2P)


def standardize_bp(
    sbp: torch.Tensor,
    dbp: torch.Tensor,
    *,
    sbp_mean: float,
    sbp_std: float,
    dbp_mean: float,
    dbp_std: float,
) -> torch.Tensor:
    """Z-score cuff calibration scalars into (..., 2) = [SBP_z, DBP_z].

    These are the CALIBRATION cuff readings carried by the support set, never
    the query segment's own SBP/DBP (that is the evaluation target and would
    leak). Fixed global constants, like ABP/demographics — never per-sample.
    """
    s = (sbp - sbp_mean) / sbp_std
    d = (dbp - dbp_mean) / dbp_std
    return torch.stack([s, d], dim=-1)


# Continuous demo channels, in the fixed order the model's DemoEncoder expects.
DEMO_CONT_DIM = 5  # [age, height, weight, bmi, body_missing_flag]


def standardize_demo(
    age: torch.Tensor,
    gender: torch.Tensor,
    height: torch.Tensor,
    weight: torch.Tensor,
    bmi: torch.Tensor,
    *,
    age_mean: float,
    age_std: float,
    height_mean: float,
    height_std: float,
    weight_mean: float,
    weight_std: float,
    bmi_mean: float,
    bmi_std: float,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Demographics -> (continuous (..., 5), gender_idx (...,) long).

    Continuous = [age, height, weight, bmi, body_missing_flag], each z-scored by
    the given fixed constants. height/weight/bmi are ~48% missing (jointly): a
    NaN there sets the body_missing flag to 1 and the value to 0 (= post-z-score
    mean), so the model sees "body metrics absent" rather than a fake number.
    gender is returned as a long index for an nn.Embedding.
    """
    body_missing = torch.isnan(height).to(age.dtype)
    age_z = torch.nan_to_num((age - age_mean) / age_std)
    h_z = torch.nan_to_num((height - height_mean) / height_std)
    w_z = torch.nan_to_num((weight - weight_mean) / weight_std)
    b_z = torch.nan_to_num((bmi - bmi_mean) / bmi_std)
    cont = torch.stack([age_z, h_z, w_z, b_z, body_missing], dim=-1)
    gender_idx = torch.nan_to_num(gender).long()
    return cont, gender_idx
