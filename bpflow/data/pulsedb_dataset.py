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
_BP_COLS = ["sbp", "dbp", "map"]  # CSV cuff label -> optional clinical TRUE for eval
MODALITY_ORDER = ("ecg_ppg", "ecg", "ppg")  # order of modality_dropout_probs
# cond_mask = [ecg_present, ppg_present]; the model nulls the absent stream(s).
MODALITY_MASK = {"ecg_ppg": (1.0, 1.0), "ecg": (1.0, 0.0), "ppg": (0.0, 1.0)}
# single source of truth for the default per-sample dropout probs (~uniform,
# order = MODALITY_ORDER); referenced by the config schema + dataset/infer fallbacks.
DEFAULT_MODALITY_DROPOUT_PROBS = (0.34, 0.33, 0.33)


def trained_modalities(
    cond_modality: str,
    modality_dropout: bool,
    modality_dropout_probs: Optional[list] = None,
) -> set:
    """The set of conditioning directions a model was actually trained on.

    ``modality_dropout`` True  -> every modality with prob > 0 (the unified case;
    degenerate probs like [1,0,0] collapse to one direction). False -> just the
    fixed ``{cond_modality}`` (specialist). Shared by the infer guard (which side
    of the checkpoint is valid to evaluate) and the trainer's per-modality val
    loss (which directions are worth logging).
    """
    if modality_dropout:
        probs = (
            list(modality_dropout_probs)
            if modality_dropout_probs
            else list(DEFAULT_MODALITY_DROPOUT_PROBS)
        )
        if len(probs) != len(MODALITY_ORDER):
            raise ValueError(
                f"modality_dropout_probs must have {len(MODALITY_ORDER)} entries "
                f"(order {MODALITY_ORDER}), got {len(probs)}"
            )
        return {m for m, p in zip(MODALITY_ORDER, probs) if float(p) > 0}
    return {str(cond_modality)}


def _csv_path_for(npy_path: str) -> str:
    """Sibling CSV that is row-aligned to the npy (Train_Subset.npy -> .csv)."""
    return os.path.splitext(npy_path)[0] + ".csv"


def _front_ratio(idx: np.ndarray, ratio: float) -> np.ndarray:
    """Keep the front ``ratio`` fraction of an already-shuffled index array (at
    least 1 element). ``ratio >= 1`` or an empty array is a no-op. Front-slicing a
    shuffled pool makes smaller ratios nested subsets of larger ones — the single
    definition of the finetune ``train_ratio`` data-efficiency knob."""
    if ratio >= 1.0 or len(idx) == 0:
        return idx
    return idx[: max(1, int(round(ratio * len(idx))))]


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
        cond_modality: str = "ecg_ppg",
        modality_dropout: bool = False,
        modality_dropout_probs: Optional[list] = None,
        use_demo: bool = False,
        demo_consts: Optional[Dict[str, float]] = None,
        eval_true_source: str = "waveform",
        finetune: bool = False,
        finetune_val_fraction: float = 0.1,
        finetune_test_fraction: float = 0.1,
        finetune_train_ratio: float = 1.0,
        finetune_split_mode: str = "segment",
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        if split_mode not in ("segment", "subject"):
            raise ValueError(f"split_mode must be segment/subject, got {split_mode}")
        if cond_modality not in ("ecg", "ppg", "ecg_ppg"):
            raise ValueError(f"cond_modality must be ecg/ppg/ecg_ppg, got {cond_modality}")
        if not 0.0 < finetune_train_ratio <= 1.0:
            raise ValueError(f"finetune_train_ratio must be in (0, 1], got {finetune_train_ratio}")
        if finetune_split_mode not in ("segment", "stratified"):
            raise ValueError(f"finetune_split_mode must be segment/stratified, got {finetune_split_mode}")
        # Root-guard both finetune split paths: val+test must leave room for train.
        # (The stratified path slices per subject and would otherwise silently
        # mis-assign with a negative n_train; the segment path raises on its own.)
        if finetune and finetune_val_fraction + finetune_test_fraction >= 1.0:
            raise ValueError(
                "finetune val+test fractions must sum to < 1 (leave room for train), got "
                f"val={finetune_val_fraction} + test={finetune_test_fraction}"
            )
        self.arr = np.load(npy_path, mmap_mode="r")
        if self.arr.ndim != 3 or self.arr.shape[1] < 3:
            raise ValueError(f"expected (N,3,L) array at {npy_path}, got {self.arr.shape}")
        total = self.arr.shape[0]

        # Read the sibling CSV once if subject-split / demographics / CSV BP labels
        # need it. Non-finetune subject split needs it for train/val; finetune
        # needs it (all splits) only in the per-subject `stratified` mode. CSV BP
        # is an eval TRUE used by infer, so only val/test load it.
        need_subject = (not finetune) and split in ("train", "val") and split_mode == "subject"
        need_finetune_subj = finetune and finetune_split_mode == "stratified"
        needs_subject_col = need_subject or need_finetune_subj
        need_bp = eval_true_source in ("csv", "both") and split in ("val", "test")
        frame = None
        if needs_subject_col or use_demo or need_bp:
            frame = self._read_csv(npy_path, total, needs_subject_col, use_demo, need_bp)

        if finetune and finetune_split_mode == "stratified":
            assert frame is not None  # need_finetune_subj forced the CSV read above
            self.indices = self._compute_finetune_indices_stratified(
                split, frame["subject_id"].to_numpy(), split_seed,
                finetune_val_fraction, finetune_test_fraction, finetune_train_ratio,
            )
        elif finetune:
            self.indices = self._compute_finetune_indices(
                split, total, split_seed, finetune_val_fraction,
                finetune_test_fraction, finetune_train_ratio,
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
        self.cond_modality = cond_modality
        # Per-sample modality dropout is a TRAIN-only augmentation; val/test keep
        # the fixed cond_modality so their metrics stay comparable.
        self._dropout_active = bool(modality_dropout) and split == "train"
        self._dropout_probs = None
        if self._dropout_active:
            probs = (
                list(modality_dropout_probs)
                if modality_dropout_probs
                else list(DEFAULT_MODALITY_DROPOUT_PROBS)
            )
            if len(probs) != len(MODALITY_ORDER) or any(p < 0 for p in probs) or sum(probs) <= 0:
                raise ValueError(
                    f"modality_dropout_probs must be {len(MODALITY_ORDER)} non-negative "
                    f"values (order {MODALITY_ORDER}) summing > 0, got {probs}"
                )
            self._dropout_probs = torch.tensor(probs, dtype=torch.float32)
        self.use_demo = use_demo

        self.demo_cont = None
        self.demo_gender = None
        if use_demo:
            assert frame is not None
            self._build_demo(frame, demo_consts or {})

        self.bp_true = None
        if need_bp:
            assert frame is not None
            bp = np.stack([frame[c].to_numpy(dtype=np.float32) for c in _BP_COLS], axis=1)
            self.bp_true = torch.from_numpy(bp[self.indices])  # (n, 3) [SBP, DBP, MAP] mmHg

        if finetune:
            mode = f"finetune-3way-{finetune_split_mode}"
            if split == "train" and finetune_train_ratio < 1.0:
                mode += f"(train_ratio={finetune_train_ratio:g})"
        else:
            mode = split_mode
        log.info(
            "PulseDBDataset[%s] %s: %d/%d segments (mode=%s, seed=%d, val_frac=%.2f, demo=%s)",
            split, npy_path, len(self.indices), total, mode, split_seed,
            val_fraction, use_demo,
        )

    # -- setup helpers -----------------------------------------------------
    @staticmethod
    def _read_csv(npy_path: str, total: int, need_subject: bool, use_demo: bool,
                  need_bp: bool = False):
        import pandas as pd

        csv_path = _csv_path_for(npy_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"need {csv_path} for subject-split/demographics/CSV-BP truth, but it "
                "is missing. Use split_mode='segment', use_demo=false, "
                "eval_true_source='waveform', or provide the CSV."
            )
        cols = ["subject_id"] if need_subject else []
        if use_demo:
            cols += _DEMO_COLS
        if need_bp:
            cols += _BP_COLS
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
    def _compute_finetune_indices(
        split, total, split_seed, val_fraction, test_fraction, train_ratio=1.0
    ):
        """Deterministic 3-way per-segment split of a single npy (train/val/test).

        One fixed-seed shuffle, then contiguous slices: test = last
        ``test_fraction``, val = the ``val_fraction`` before it, train = the
        rest. The same seed across the three build calls makes the partition
        consistent and non-overlapping (no segment is in two splits).

        ``train_ratio`` < 1 keeps only that fraction of the train split (a
        data-efficiency knob). It slices the FRONT of the already-shuffled train
        pool, so smaller ratios are nested subsets of larger ones, and val/test
        are never subsampled — metrics stay comparable across ratios.
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
            chosen = _front_ratio(order[:n_train], train_ratio)
        elif split == "val":
            chosen = order[n_train:n_train + n_val]
        else:  # test
            chosen = order[n_train + n_val:]
        return np.sort(chosen)

    @staticmethod
    def _compute_finetune_indices_stratified(
        split, sids, split_seed, val_fraction, test_fraction, train_ratio=1.0
    ):
        """Per-subject stratified 3-way split (train/val/test).

        Each subject's own segments are split 8:1:1 individually, so every split
        is segment-balanced across subjects. Subjects therefore OVERLAP all three
        splits — this is NOT subject-disjoint (use it for balanced finetuning, not
        as an honest generalization gate).

        One global fixed-seed shuffle fixes each subject's internal order; then
        per subject: test = last ``test_fraction``, val = the ``val_fraction``
        before it, train = the rest. ``train_ratio`` < 1 keeps the FRONT fraction
        of each subject's train segments (per-subject nested subsampling; val/test
        untouched). A subject too small for a slice contributes 0 segments there.
        The same seed across the three build calls keeps the partition consistent
        and non-overlapping at the segment level.
        """
        sids = np.asarray(sids)
        total = len(sids)
        order = np.arange(total)
        np.random.default_rng(split_seed).shuffle(order)
        sids_shuf = sids[order]
        chosen = []
        for subj in np.unique(sids):
            seg = order[sids_shuf == subj]  # this subject's segment idx, shuffled order
            n_s = len(seg)
            n_test = int(test_fraction * n_s)
            n_val = int(val_fraction * n_s)
            n_train = n_s - n_val - n_test
            if split == "train":
                sel = _front_ratio(seg[:n_train], train_ratio)
            elif split == "val":
                sel = seg[n_train:n_train + n_val]
            else:  # test
                sel = seg[n_train + n_val:]
            if len(sel):
                chosen.append(sel)
        if not chosen:
            raise ValueError(
                f"stratified finetune split produced no '{split}' segments "
                f"(val_frac={val_fraction}, test_frac={test_fraction}); subjects "
                "likely have too few segments each for this slice."
            )
        return np.sort(np.concatenate(chosen))

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
        cond_patches = build_cond_patches(ecg, ppg, self.patch_size, self.cond_recenter)  # (N, 2P)
        # TRAIN-only: draw this sample's modality (torch RNG → DataLoader seeds it
        # per worker). Otherwise use the fixed cond_modality (val/test/infer). The
        # choice becomes a cond_mask; the model nulls the absent stream(s).
        if self._dropout_active:
            assert self._dropout_probs is not None
            modality = MODALITY_ORDER[int(torch.multinomial(self._dropout_probs, 1).item())]
        else:
            modality = self.cond_modality
        cond_mask = torch.tensor(MODALITY_MASK[modality], dtype=torch.float32)  # (2,) [ecg, ppg]

        out = {
            "abp_patches": abp_patches,
            "cond_patches": cond_patches,
            "cond_mask": cond_mask,  # (2,) [ecg_present, ppg_present] -> model nulls absent
            "abp_raw": abp,  # ground-truth mmHg waveform for eval (unclipped)
        }
        if self.bp_true is not None:
            out["bp_true"] = self.bp_true[i]  # (3,) CSV cuff [SBP, DBP, MAP] mmHg
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
        cond_modality=str(getattr(d, "cond_modality", "ecg_ppg")),
        modality_dropout=bool(getattr(d, "modality_dropout", False)),
        modality_dropout_probs=(
            list(d.modality_dropout_probs) if getattr(d, "modality_dropout_probs", None) else None
        ),
        use_demo=bool(cfg.model.use_demo),
        demo_consts=demo_consts,
        eval_true_source=str(getattr(d, "eval_true_source", "waveform")),
        finetune=finetune,
        finetune_val_fraction=float(getattr(d, "finetune_val_fraction", 0.1)),
        finetune_test_fraction=float(getattr(d, "finetune_test_fraction", 0.1)),
        finetune_train_ratio=float(getattr(d, "finetune_train_ratio", 1.0)),
        finetune_split_mode=str(getattr(d, "finetune_split_mode", "segment")),
    )
