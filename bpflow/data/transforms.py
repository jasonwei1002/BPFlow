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
