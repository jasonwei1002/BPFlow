"""PulseDB dataset for ECG+PPG -> ABP flow matching.

Reads the (N, 3, 1250) ``.npy`` arrays (channel order verified: 0=ECG,
1=PPG, 2=ABP). Train_Subset is split into train/val with a fixed seed;
CalFree_Test_Subset is used whole as the (subject-disjoint) test set.

Splitting (``split_mode``):
- ``"segment"`` — random per-segment 80/20. PulseDB has many segments/subject,
  so this places the same subject in both train and val → val is optimistic.
- ``"subject"`` — subject-disjoint, using the ``subject_id`` column of the
  sibling CSV (``<npy>.csv``, row-aligned to the npy). Honest val that matches
  the CalFree test setting. This is the preferred mode.

Demographics (``use_demo``): age/gender/height/weight/bmi are read from the same
CSV and returned per segment. height/weight/bmi are ~48% missing (jointly), so
they are carried with a missing flag (see :func:`standardize_demo`). The CSV
``sbp/dbp/map`` columns are the LABEL and are never read as input.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import (
    build_cond_patches,
    patchify,
    standardize_abp,
    standardize_demo,
)

log = logging.getLogger(__name__)

_DEMO_COLS = ["age", "gender", "height", "weight", "bmi"]


def _csv_path_for(npy_path: str) -> str:
    """Sibling CSV that is row-aligned to the npy (Train_Subset.npy -> .csv)."""
    return os.path.splitext(npy_path)[0] + ".csv"


class PulseDBDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        split: str,
        *,
        split_seed: int = 42,
        val_fraction: float = 0.2,
        split_mode: str = "segment",
        patch_size: int = 10,
        ecg_channel: int = 0,
        ppg_channel: int = 1,
        abp_channel: int = 2,
        abp_mean: float = 81.94,
        abp_std: float = 24.43,
        abp_clip_low: float = 20.0,
        abp_clip_high: float = 250.0,
        cond_recenter: bool = True,
        use_demo: bool = False,
        demo_consts: Optional[Dict[str, float]] = None,
        finetune: bool = False,
        finetune_val_fraction: float = 0.1,
        finetune_test_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        if split_mode not in ("segment", "subject"):
            raise ValueError(f"split_mode must be segment/subject, got {split_mode}")
        self.arr = np.load(npy_path, mmap_mode="r")
        if self.arr.ndim != 3 or self.arr.shape[1] < 3:
            raise ValueError(f"expected (N,3,L) array at {npy_path}, got {self.arr.shape}")
        total = self.arr.shape[0]

        # Read the sibling CSV once if subject-split or demographics need it.
        # Finetune never uses the subject split (it does a 3-way segment split).
        need_subject = (not finetune) and split in ("train", "val") and split_mode == "subject"
        frame = None
        if need_subject or use_demo:
            frame = self._read_csv(npy_path, total, need_subject, use_demo)

        if finetune:
            self.indices = self._compute_finetune_indices(
                split, total, split_seed, finetune_val_fraction, finetune_test_fraction
            )
        else:
            self.indices = self._compute_indices(
                split, total, split_mode, split_seed, val_fraction, frame
            )

        self.split = split
        self.finetune = finetune
        self.patch_size = patch_size
        self.ecg_channel = ecg_channel
        self.ppg_channel = ppg_channel
        self.abp_channel = abp_channel
        self.abp_mean = abp_mean
        self.abp_std = abp_std
        self.abp_clip_low = abp_clip_low
        self.abp_clip_high = abp_clip_high
        self.cond_recenter = cond_recenter
        self.use_demo = use_demo

        self.demo_cont = None
        self.demo_gender = None
        if use_demo:
            assert frame is not None
            self._build_demo(frame, demo_consts or {})

        mode = "finetune-3way" if finetune else split_mode
        log.info(
            "PulseDBDataset[%s] %s: %d/%d segments (mode=%s, seed=%d, val_frac=%.2f, demo=%s)",
            split, npy_path, len(self.indices), total, mode, split_seed,
            val_fraction, use_demo,
        )

    # -- setup helpers -----------------------------------------------------
    @staticmethod
    def _read_csv(npy_path: str, total: int, need_subject: bool, use_demo: bool):
        import pandas as pd

        csv_path = _csv_path_for(npy_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"need {csv_path} for subject-split/demographics, but it is "
                "missing. Use split_mode='segment', use_demo=false, or "
                "provide the CSV."
            )
        cols = ["subject_id"] if need_subject else []
        if use_demo:
            cols += _DEMO_COLS
        frame = pd.read_csv(csv_path, usecols=cols)
        if len(frame) != total:
            raise ValueError(
                f"CSV/npy row mismatch: {csv_path} has {len(frame)} rows, "
                f"{npy_path} has {total}. They must be row-aligned."
            )
        return frame

    @staticmethod
    def _compute_indices(split, total, split_mode, split_seed, val_fraction, frame):
        if split == "test":  # test uses all of CalFree
            return np.arange(total)
        if split_mode == "subject":
            assert frame is not None
            sids = frame["subject_id"].to_numpy()
            uniq = np.unique(sids)
            order = np.random.default_rng(split_seed).permutation(len(uniq))
            n_train_subj = int((1.0 - val_fraction) * len(uniq))
            train_subj = uniq[order[:n_train_subj]]
            in_train = np.isin(sids, train_subj)
            mask = in_train if split == "train" else ~in_train
            return np.nonzero(mask)[0]
        # segment mode: random per-segment split
        order = np.arange(total)
        np.random.default_rng(split_seed).shuffle(order)
        n_train = int((1.0 - val_fraction) * total)
        chosen = order[:n_train] if split == "train" else order[n_train:]
        return np.sort(chosen)

    @staticmethod
    def _compute_finetune_indices(split, total, split_seed, val_fraction, test_fraction):
        """Deterministic 3-way per-segment split of a single npy (train/val/test).

        One fixed-seed shuffle, then contiguous slices: test = last
        ``test_fraction``, val = the ``val_fraction`` before it, train = the
        rest. The same seed across the three build calls makes the partition
        consistent and non-overlapping (no segment is in two splits).
        """
        order = np.arange(total)
        np.random.default_rng(split_seed).shuffle(order)
        n_test = int(test_fraction * total)
        n_val = int(val_fraction * total)
        n_train = total - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"finetune split leaves no train: total={total}, val_frac={val_fraction}, "
                f"test_frac={test_fraction}"
            )
        if split == "train":
            chosen = order[:n_train]
        elif split == "val":
            chosen = order[n_train:n_train + n_val]
        else:  # test
            chosen = order[n_train + n_val:]
        return np.sort(chosen)

    def _build_demo(self, frame, demo_consts: Dict[str, float]) -> None:
        def col(name: str) -> torch.Tensor:
            return torch.tensor(frame[name].to_numpy(), dtype=torch.float32)

        cont, gender = standardize_demo(
            col("age"), col("gender"), col("height"), col("weight"), col("bmi"),
            **demo_consts,
        )
        self.demo_cont = cont[self.indices]  # (n, 5)
        self.demo_gender = gender[self.indices]  # (n,)

    # -- access ------------------------------------------------------------
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

        out = {
            "abp_patches": abp_patches,
            "cond_patches": cond_patches,
            "abp_raw": abp,  # ground-truth mmHg for evaluation (unclipped)
        }
        if self.use_demo:
            assert self.demo_cont is not None and self.demo_gender is not None
            out["demo_cont"] = self.demo_cont[i]      # (5,)
            out["demo_gender"] = self.demo_gender[i]  # () long
        return out


def build_dataset(cfg, split: str) -> PulseDBDataset:
    """Build a PulseDBDataset for ``split`` from a config object."""
    d = cfg.data
    finetune = bool(getattr(d, "finetune", False))
    # Finetune splits the CalFree `test_npy` 3 ways, so every split reads it.
    if finetune:
        npy_path = str(d.test_npy)
    else:
        npy_path = str(d.test_npy) if split == "test" else str(d.train_npy)
    demo_consts = {
        "age_mean": float(d.demo_age_mean), "age_std": float(d.demo_age_std),
        "height_mean": float(d.demo_height_mean), "height_std": float(d.demo_height_std),
        "weight_mean": float(d.demo_weight_mean), "weight_std": float(d.demo_weight_std),
        "bmi_mean": float(d.demo_bmi_mean), "bmi_std": float(d.demo_bmi_std),
    }
    return PulseDBDataset(
        npy_path,
        split,
        split_seed=int(d.split_seed),
        val_fraction=float(d.val_fraction),
        split_mode=str(d.split_mode),
        patch_size=int(cfg.model.patch_size),
        ecg_channel=int(d.ecg_channel),
        ppg_channel=int(d.ppg_channel),
        abp_channel=int(d.abp_channel),
        abp_mean=float(d.abp_mean),
        abp_std=float(d.abp_std),
        abp_clip_low=float(d.abp_clip_low),
        abp_clip_high=float(d.abp_clip_high),
        cond_recenter=bool(d.cond_recenter),
        use_demo=bool(cfg.model.use_demo),
        demo_consts=demo_consts,
        finetune=finetune,
        finetune_val_fraction=float(getattr(d, "finetune_val_fraction", 0.1)),
        finetune_test_fraction=float(getattr(d, "finetune_test_fraction", 0.1)),
    )
