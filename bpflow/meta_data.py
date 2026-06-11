"""Episodic (per-subject) data for ANIL meta-training and K-shot evaluation.

A task is one subject. Training draws a support set (Ks segments) and a disjoint
query set (Kq segments) from the same subject; the meta inner loop adapts phi on
support, the outer loss is on query. Evaluation is subject-disjoint K-shot: adapt
phi on K of a held-out subject's segments, predict the rest.

Subject grouping comes from the sibling CSV ``subject_id`` column (the same column
the subject-split / honest eval already rely on). Segments are turned into the
exact patches the model trains on (``standardize_abp`` + ``patchify`` for ABP,
``build_cond_patches`` for ECG/PPG), plus the raw mmHg ABP for evaluation metrics.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .data.pulsedb_dataset import _csv_path_for
from .data.transforms import build_cond_patches, patchify, standardize_abp

# (abp_support, cond_support, bp_support, abp_query, cond_query, bp_query)
# bp_* are z-scored [SBP, DBP] (n, 2). For the scalar (cuff) inner objective the
# support's abp is unused; it is still carried so one episode serves both modes.
Episode = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def load_bp_z(npy_path: str, cfg) -> np.ndarray:
    """Per-segment z-scored cuff [SBP, DBP] (total, 2), aligned to the npy rows."""
    import pandas as pd

    df = pd.read_csv(_csv_path_for(npy_path), usecols=["sbp", "dbp"])
    sbp = df["sbp"].to_numpy(dtype=np.float32)
    dbp = df["dbp"].to_numpy(dtype=np.float32)
    d = cfg.data
    z = np.stack(
        [(sbp - float(d.bp_sbp_mean)) / float(d.bp_sbp_std),
         (dbp - float(d.bp_dbp_mean)) / float(d.bp_dbp_std)],
        axis=1,
    )
    return z.astype(np.float32)


def subject_groups(npy_path: str, min_segs: int = 1) -> Dict[object, np.ndarray]:
    """subject_id -> segment row indices, keeping subjects with >= min_segs."""
    import pandas as pd

    csv_path = _csv_path_for(npy_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"meta-learning needs {csv_path} (subject_id column) for per-subject episodes"
        )
    sids = pd.read_csv(csv_path, usecols=["subject_id"])["subject_id"].to_numpy()
    groups: Dict[object, list] = {}
    for i, s in enumerate(sids):
        groups.setdefault(s, []).append(i)
    return {s: np.asarray(v, dtype=np.int64) for s, v in groups.items() if len(v) >= min_segs}


def split_subjects(
    keys: List[object], val_fraction: float, seed: int
) -> Tuple[List[object], List[object]]:
    """Subject-disjoint train/val split of subject keys (fixed seed)."""
    keys = list(keys)
    order = np.random.default_rng(seed).permutation(len(keys))
    n_val = max(1, int(val_fraction * len(keys)))
    val = [keys[i] for i in order[:n_val]]
    train = [keys[i] for i in order[n_val:]]
    return train, val


def shard_subjects(keys: List[object], rank: int, world_size: int) -> List[object]:
    """Strided subject shard for DDP (exact coverage, no duplicates)."""
    if world_size <= 1:
        return list(keys)
    return [keys[i] for i in range(rank, len(keys), world_size)]


def _seg_to_patches(arr, i: int, d, patch_size: int):
    """Raw (3,L) row -> (abp_patches (N,P), cond_patches (N,2P), abp_raw (L,))."""
    row = np.asarray(arr[int(i)]).astype(np.float32)
    ecg = torch.from_numpy(row[int(d.ecg_channel)])
    ppg = torch.from_numpy(row[int(d.ppg_channel)])
    abp = torch.from_numpy(row[int(d.abp_channel)])  # raw mmHg
    abp_norm = standardize_abp(
        abp, float(d.abp_mean), float(d.abp_std), float(d.abp_clip_low), float(d.abp_clip_high)
    )
    abp_p = patchify(abp_norm, patch_size)
    cond_p = build_cond_patches(ecg, ppg, patch_size, bool(d.cond_recenter))
    return abp_p, cond_p, abp


def stack_segments(arr, idxs, d, patch_size: int):
    """Stack segments -> (abp_patches (n,N,P), cond_patches (n,N,2P), abp_raw (n,L))."""
    abp_p, cond_p, raw = [], [], []
    for i in idxs:
        a, c, r = _seg_to_patches(arr, i, d, patch_size)
        abp_p.append(a); cond_p.append(c); raw.append(r)
    return torch.stack(abp_p), torch.stack(cond_p), torch.stack(raw)


class EpisodeDataset(IterableDataset):
    """Infinite stream of same-subject (support, query) episodes for meta-training.

    Each yielded episode carries z-scored cuff [SBP, DBP] for both support and
    query (from ``bp_z``), so the scalar (cuff) inner loop never needs the
    support's ABP waveform.
    """

    def __init__(self, npy_path: str, subj_to_idx: Dict[object, np.ndarray], bp_z: np.ndarray,
                 d, patch_size: int, ks: int, kq: int, seed: int) -> None:
        super().__init__()
        self.npy_path = npy_path
        self.subjects = list(subj_to_idx.keys())
        self.subj_to_idx = subj_to_idx
        self.bp_z = bp_z
        self.d = d
        self.patch_size = patch_size
        self.ks = ks
        self.kq = kq
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.seed + 1000 * wid)
        arr = np.load(self.npy_path, mmap_mode="r")  # per-worker mmap handle
        need = self.ks + self.kq
        while True:
            s = self.subjects[rng.integers(len(self.subjects))]
            idxs = self.subj_to_idx[s]
            pick = idxs[rng.permutation(len(idxs))[:need]]
            sup, qry = pick[: self.ks], pick[self.ks: need]
            abp_s, cond_s, _ = stack_segments(arr, sup, self.d, self.patch_size)
            abp_q, cond_q, _ = stack_segments(arr, qry, self.d, self.patch_size)
            bp_s = torch.from_numpy(self.bp_z[sup])
            bp_q = torch.from_numpy(self.bp_z[qry])
            yield abp_s, cond_s, bp_s, abp_q, cond_q, bp_q


def episode_collate(batch: List[Episode]) -> List[Episode]:
    """Keep the meta-batch as a list of episodes (NOT stacked across subjects)."""
    return list(batch)


def build_episode_loader(cfg, subj_to_idx: Dict[object, np.ndarray], bp_z: np.ndarray, *, seed: int) -> DataLoader:
    """DataLoader yielding lists of ``meta_batch_subjects`` episodes per step."""
    m = cfg.meta
    ds = EpisodeDataset(
        str(cfg.data.train_npy), subj_to_idx, bp_z, cfg.data,
        int(cfg.model.patch_size), int(m.support_size), int(m.query_size), seed,
    )
    return DataLoader(
        ds,
        batch_size=int(m.meta_batch_subjects),
        num_workers=int(m.num_workers),
        collate_fn=episode_collate,
        pin_memory=False,
        persistent_workers=int(m.num_workers) > 0,
    )


def eval_split(idxs: np.ndarray, k: int, max_query: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """For one subject: first ``k`` segments = support (calibration), rest = query.

    Returns (support_idx, query_idx). Query is capped at ``max_query``. With k=0
    the support is empty (calibration-free baseline, phi stays 0).
    """
    perm = idxs[rng.permutation(len(idxs))]
    sup = perm[:k]
    qry = perm[k: k + max_query] if max_query > 0 else perm[k:]
    return sup, qry
