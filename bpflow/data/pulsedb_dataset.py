"""PulseDB dataset for ECG+PPG -> ABP flow matching.

Reads the (N, 3, 1250) ``.npy`` arrays (channel order verified: 0=ECG,
1=PPG, 2=ABP). Train_Subset is split 80/20 with a fixed seed into train/val;
CalFree_Test_Subset is used whole as the (subject-disjoint) test set.

NOTE on splitting: a random segment-level 80/20 split can place segments from
the same subject into both train and val (PulseDB has many segments/subject),
so val is optimistic. CalFree is the honest generalization gate. The split is
random per the project spec; ``subject_wise`` is reserved for future use (the
.npy carries no subject ids; would require the .h5 ``subject_ids``).
"""

import logging
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_cond_patches, patchify, standardize_abp

log = logging.getLogger(__name__)


class PulseDBDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        split: str,
        *,
        split_seed: int = 42,
        val_fraction: float = 0.2,
        patch_size: int = 10,
        ecg_channel: int = 0,
        ppg_channel: int = 1,
        abp_channel: int = 2,
        abp_mean: float = 81.94,
        abp_std: float = 24.43,
        abp_clip_low: float = 20.0,
        abp_clip_high: float = 250.0,
        cond_recenter: bool = True,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        self.arr = np.load(npy_path, mmap_mode="r")
        if self.arr.ndim != 3 or self.arr.shape[1] < 3:
            raise ValueError(f"expected (N,3,L) array at {npy_path}, got {self.arr.shape}")
        total = self.arr.shape[0]

        if split in ("train", "val"):
            order = np.arange(total)
            np.random.default_rng(split_seed).shuffle(order)
            n_train = int((1.0 - val_fraction) * total)
            chosen = order[:n_train] if split == "train" else order[n_train:]
            self.indices = np.sort(chosen)
        else:  # test: use all of CalFree
            self.indices = np.arange(total)

        self.split = split
        self.patch_size = patch_size
        self.ecg_channel = ecg_channel
        self.ppg_channel = ppg_channel
        self.abp_channel = abp_channel
        self.abp_mean = abp_mean
        self.abp_std = abp_std
        self.abp_clip_low = abp_clip_low
        self.abp_clip_high = abp_clip_high
        self.cond_recenter = cond_recenter
        log.info(
            "PulseDBDataset[%s] %s: %d/%d segments (seed=%d, val_frac=%.2f)",
            split, npy_path, len(self.indices), total, split_seed, val_fraction,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = np.asarray(self.arr[self.indices[i]]).astype(np.float32)  # (3, L)
        ecg = torch.from_numpy(row[self.ecg_channel])
        ppg = torch.from_numpy(row[self.ppg_channel])
        abp = torch.from_numpy(row[self.abp_channel])  # raw mmHg

        abp_norm = standardize_abp(
            abp, self.abp_mean, self.abp_std, self.abp_clip_low, self.abp_clip_high
        )
        abp_patches = patchify(abp_norm, self.patch_size)  # (N, P)
        cond_patches = build_cond_patches(ecg, ppg, self.patch_size, self.cond_recenter)  # (N,2P)

        return {
            "abp_patches": abp_patches,
            "cond_patches": cond_patches,
            "abp_raw": abp,  # ground-truth mmHg for evaluation (unclipped)
        }


def build_dataset(cfg, split: str) -> PulseDBDataset:
    """Build a PulseDBDataset for ``split`` from a config object."""
    d = cfg.data
    npy_path = str(d.test_npy) if split == "test" else str(d.train_npy)
    return PulseDBDataset(
        npy_path,
        split,
        split_seed=int(d.split_seed),
        val_fraction=float(d.val_fraction),
        patch_size=int(cfg.model.patch_size),
        ecg_channel=int(d.ecg_channel),
        ppg_channel=int(d.ppg_channel),
        abp_channel=int(d.abp_channel),
        abp_mean=float(d.abp_mean),
        abp_std=float(d.abp_std),
        abp_clip_low=float(d.abp_clip_low),
        abp_clip_high=float(d.abp_clip_high),
        cond_recenter=bool(d.cond_recenter),
    )
