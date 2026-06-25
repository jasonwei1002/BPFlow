"""Dataset adapter: reuse bpflow's PulseDB splits, emit per-direction waveforms.

Wraps a ``bpflow.data.PulseDBDataset`` (so the train/val/test split, seeds,
clip + recenter normalization and finetune 8:1:1 logic are *identical* to
bpflow) and re-derives full-length waveforms via the lossless ``unpatchify``.
Each item is a single source -> single target pair for one fixed ``direction``.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch.utils.data import Dataset

from bpflow.data import build_dataset as build_bpflow_dataset
from bpflow.data import unpatchify

from .norms import (
    ABP_TARGET_MODE,
    DIRECTIONS,
    MOD_IDX,
    abp_global_norm,
    abp_perslot_norm,
)

logger = logging.getLogger(__name__)

_PATCH_KEY = {"abp": "abp_patches", "ecg": "ecg_patches", "ppg": "ppg_patches"}


class BaselineDataset(Dataset):
    """One (source_waveform -> target_waveform) pair per item for ``direction``.

    Returns per item (collate adds the batch dim):
        x         (C, L)  source modality(ies), bpflow recenter scale (ECG/PPG-0.5);
                          C=1 for 1->1 directions, C=2 for ecg_ppg2abp ([ECG;PPG]).
        y         (1, L)  target normalized: ABP -> [0,1] (global or perslot),
                          bridge ECG/PPG -> recenter scale.
        abp_raw   (L,)    raw mmHg ABP (for eval; meaningful when target=ABP).
        sbp,dbp   ()      per-sample clip(ABP) max/min in mmHg (perslot 2-stage).
        target_idx ()     target modality index (0/1/2), bpflow convention.
        tgt_is_abp ()     bool flag.
    """

    def __init__(self, cfg, split: str, direction: str, abp_target_mode: str):
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}; expected {list(DIRECTIONS)}")
        if abp_target_mode not in ("global", "perslot"):
            raise ValueError(f"abp_target_mode must be global|perslot, got {abp_target_mode!r}")
        self.base = build_bpflow_dataset(cfg, split)
        self.direction = direction
        self.src, self.tgt = DIRECTIONS[direction]
        self.tgt_is_abp = self.tgt == "abp"
        self.abp_target_mode = abp_target_mode
        self.clip_lo = float(cfg.data.abp_clip_low)
        self.clip_hi = float(cfg.data.abp_clip_high)
        self.target_idx = MOD_IDX[self.tgt]

    def __len__(self) -> int:
        return len(self.base)

    def _wave(self, sample: Dict[str, torch.Tensor], mod: str) -> torch.Tensor:
        """Full-length waveform for ``mod`` in bpflow's normalized space (L,).

        ECG/PPG -> recenter scale (as stored); ABP -> z-score (unused here).
        """
        return unpatchify(sample[_PATCH_KEY[mod]])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = self.base[i]
        abp_raw = s["abp_raw"]  # (L,) raw mmHg
        if isinstance(self.src, tuple):  # multi-source: stack [ECG; PPG] -> (C, L)
            x = torch.stack([self._wave(s, m) for m in self.src], dim=0)
        else:
            x = self._wave(s, self.src).unsqueeze(0)  # (1, L) recenter scale

        sbp = torch.zeros(())
        dbp = torch.zeros(())
        if self.tgt_is_abp:
            if self.abp_target_mode == "global":
                y = abp_global_norm(abp_raw, self.clip_lo, self.clip_hi).unsqueeze(0)
            else:  # perslot
                w01, sbp, dbp = abp_perslot_norm(abp_raw, self.clip_lo, self.clip_hi)
                y = w01.unsqueeze(0)
        else:
            y = self._wave(s, self.tgt).unsqueeze(0)  # (1, L) recenter scale

        return {
            "x": x.float(),
            "y": y.float(),
            "abp_raw": abp_raw.float(),
            "sbp": sbp.float(),
            "dbp": dbp.float(),
            "target_idx": torch.tensor(self.target_idx, dtype=torch.long),
            "tgt_is_abp": torch.tensor(self.tgt_is_abp),
        }


def build_baseline_dataset(cfg, split: str) -> BaselineDataset:
    """Factory from a baseline config (needs cfg.baseline.{direction,model_name})."""
    direction = str(cfg.baseline.direction)
    model_name = str(cfg.model.name)
    abp_mode = ABP_TARGET_MODE.get(model_name, "global")
    return BaselineDataset(cfg, split, direction, abp_mode)
