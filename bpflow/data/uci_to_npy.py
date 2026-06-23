"""Convert the UCI cuff-less BP h5 files -> PulseDB-style (N, 3, L) float32 npy.

This is the provenance for ``rawdata/uci/*.npy`` (the inputs to uci.yaml /
uci_finetune.yaml). Run once from the repo root after placing the source h5:

    python -m bpflow.data.uci_to_npy

Channel order matches PulseDB so PulseDBDataset reads the output unchanged:
ch0=ECG, ch1=PPG, ch2=ABP (raw mmHg, taken from the ``ABP_GRND`` group). ECG/PPG
are already global-min-max [0,1] (the convention BPFlow expects; it recenters
conditions itself). The native 1024-sample window is kept (NOT resampled), so
uci.yaml uses seq_len 1024 / patch_size 8. Chunked via ``open_memmap`` so the
~8 GB train file never loads whole into RAM. Non-finite rows are counted (not
dropped) and reported, so a dirty source surfaces at conversion time.
"""
import sys
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from numpy.lib.format import open_memmap

PAIRS = [
    ("rawdata/uci/UCI_Train_Dataset_fold_1.h5", "rawdata/uci/UCI_Train_Dataset_fold_1.npy"),
    ("rawdata/uci/UCI_Test_Dataset_fold_1.h5", "rawdata/uci/UCI_Test_Dataset_fold_1.npy"),
]
CHUNK = 8192


def convert(src: str, dst: str) -> None:
    with h5py.File(src, "r") as h:
        ecg_ds = cast(h5py.Dataset, h["ECG"])
        ppg_ds = cast(h5py.Dataset, h["PPG"])
        abp_ds = cast(h5py.Dataset, h["ABP_GRND"])  # raw mmHg
        n, length = ecg_ds.shape
        out = open_memmap(dst, mode="w+", dtype=np.float32, shape=(n, 3, length))
        bad_rows = 0
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            block = np.stack([ecg_ds[s:e], ppg_ds[s:e], abp_ds[s:e]], axis=1).astype(np.float32)  # (b,3,L)
            bad_rows += int((~np.isfinite(block)).any(axis=(1, 2)).sum())  # rows w/ any NaN/Inf
            out[s:e] = block
            print(f"  {dst}: {e}/{n}", flush=True)
        out.flush()
        del out
    print(f"DONE {src} -> {dst}  (N={n}, L={length})  non-finite-rows={bad_rows}")


def main() -> None:
    # paths are repo-relative; run from the repo root
    for src, dst in PAIRS:
        if not Path(src).exists():
            sys.exit(f"missing {src}")
        convert(src, dst)


if __name__ == "__main__":
    main()
