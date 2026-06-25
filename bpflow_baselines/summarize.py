"""Aggregate baseline infer results into comparison tables.

Scans a root for ``**/infer_*/metrics.json`` (written by bpflow_baselines.infer),
labels each row from the sibling ``run_info.json`` (falling back to the path),
and emits two tables -- one for the ->ABP directions (full clinical block) and
one for the bridge directions (waveform only) -- to the console, CSV and Markdown.

    python -m bpflow_baselines.summarize
    python -m bpflow_baselines.summarize --root output/baselines --out output/baselines
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("summarize")

_DIRECTIONS = ("ecg2abp", "ppg2abp", "ecg_ppg2abp", "ecg2ppg", "ppg2ecg")
_MODEL_ORDER = ["mdvisco", "nabnet", "patchtst", "ppg2abp", "p2e_wgan", "wavenet"]


def _label(metrics_path: str) -> Dict[str, str]:
    """model + direction for a metrics.json, from run_info.json or the path."""
    info_path = os.path.join(os.path.dirname(metrics_path), "run_info.json")
    if os.path.exists(info_path):
        try:
            info = json.load(open(info_path))
            return {"model": str(info["model"]), "direction": str(info["direction"])}
        except Exception:  # noqa: BLE001
            pass
    # fallback: .../<model>_<direction>_<stage>/infer_<direction>/metrics.json
    infer_dir = os.path.basename(os.path.dirname(metrics_path))
    direction = infer_dir.replace("infer_", "")
    run_name = os.path.basename(os.path.dirname(os.path.dirname(metrics_path)))
    # Strip the LONGEST direction first so 'ecg_ppg2abp' is matched before its
    # substring 'ppg2abp' (otherwise 'mdvisco_ecg_ppg2abp_ft' -> 'mdvisco_ecg').
    for d in sorted(_DIRECTIONS, key=len, reverse=True):
        run_name = run_name.replace(f"_{d}_pre", "").replace(f"_{d}_ft", "").replace(f"_{d}", "")
    model = run_name
    return {"model": model, "direction": direction}


def _bp_block(row: dict, report: dict, key: str) -> None:
    blk = report.get(key)
    if not blk:
        return
    aami, bhs = blk.get("AAMI", {}), blk.get("BHS", {})
    row[f"{key}_ME"] = aami.get("ME")
    row[f"{key}_SDE"] = aami.get("SDE")
    row[f"{key}_AAMI"] = "pass" if aami.get("pass") else "fail"
    row[f"{key}_BHS"] = bhs.get("grade")


def collect(root: str) -> pd.DataFrame:
    rows: List[dict] = []
    for mp in sorted(glob.glob(os.path.join(root, "**", "infer_*", "metrics.json"), recursive=True)):
        try:
            report = json.load(open(mp))
        except Exception as e:  # noqa: BLE001
            logger.warning("skip unreadable %s: %s", mp, e)
            continue
        lab = _label(mp)
        wave = report.get("waveform", {})
        row = {
            "model": lab["model"], "direction": lab["direction"],
            "wave_MAE": wave.get("MAE"), "wave_RMSE": wave.get("RMSE"),
            "wave_Pearson": wave.get("Pearson"),
            "tgt_is_abp": lab["direction"].endswith("2abp"),
            "_path": mp,
        }
        for key in ("SBP", "DBP", "MAP"):
            _bp_block(row, report, key)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_m"] = df["model"].apply(lambda m: _MODEL_ORDER.index(m) if m in _MODEL_ORDER else 99)
    df = df.sort_values(["tgt_is_abp", "direction", "_m"], ascending=[False, True, True])
    return df.drop(columns=["_m"])


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _emit(df: pd.DataFrame, name: str, cols: List[str], out_dir: str, title: str) -> None:
    if df.empty:
        return
    sub = df[[c for c in cols if c in df.columns]].copy()
    logger.info("\n===== %s (%d runs) =====\n%s", title, len(sub), sub.to_string(index=False))
    csv_path = os.path.join(out_dir, f"{name}.csv")
    sub.to_csv(csv_path, index=False)
    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w") as f:
        f.write(f"### {title}\n\n{_to_markdown(sub)}\n")
    logger.info("  wrote %s , %s", csv_path, md_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/baselines")
    ap.add_argument("--out", default=None, help="output dir for CSV/MD (default = root)")
    args = ap.parse_args()
    out_dir = args.out or args.root
    os.makedirs(out_dir, exist_ok=True)

    df = collect(args.root)
    if df.empty:
        logger.info("no metrics.json found under %s/**/infer_*/  (run the grid first)", args.root)
        return

    abp = df[df["tgt_is_abp"]].drop(columns=["tgt_is_abp", "_path"])
    bridge = df[~df["tgt_is_abp"]].drop(columns=["tgt_is_abp", "_path"])

    abp_cols = ["model", "direction", "wave_MAE", "wave_RMSE", "wave_Pearson",
                "SBP_ME", "SBP_SDE", "SBP_AAMI", "SBP_BHS",
                "DBP_ME", "DBP_SDE", "DBP_AAMI", "DBP_BHS",
                "MAP_ME", "MAP_SDE", "MAP_AAMI", "MAP_BHS"]
    bridge_cols = ["model", "direction", "wave_MAE", "wave_RMSE", "wave_Pearson"]
    _emit(abp, "summary_abp", abp_cols, out_dir, "->ABP directions (mmHg; waveform + clinical)")
    _emit(bridge, "summary_bridge", bridge_cols, out_dir, "Bridge directions (normalized [0,1] waveform)")


if __name__ == "__main__":
    main()
