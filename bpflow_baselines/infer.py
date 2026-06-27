"""Score a trained baseline on bpflow's held-out test split (single process).

Reconstructs predictions to mmHg (->ABP) or [0,1] (bridge) and evaluates with
bpflow's own ``evaluate`` / ``waveform_metrics`` so the numbers are directly
comparable to bpflow. Writes metrics.json with the same structure bpflow's
single-task infer produces.

    python -m bpflow_baselines.infer --model nabnet --direction ecg2abp \
        --ckpt output/baselines/<run>/checkpoint_best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

from bpflow.eval import evaluate, format_report, waveform_metrics
from bpflow.trainer_utils import pick_device, set_seed

from .config import load_config, overrides_from_extra
from .data import build_baseline_dataset
from .engine import load_state, log_test_to_run
from .models.base import build_model, pad_to_multiple
from .norms import ABP_TARGET_MODE
from .reconstruct import reconstruct_pred, reconstruct_true

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run(cfg, ckpt_path: str, split: str, num: int, out_dir: str) -> dict:
    device = pick_device(str(cfg.training.device))
    direction = str(cfg.baseline.direction)
    tgt_is_abp = direction.endswith("2abp")
    model_name = str(cfg.model.name)
    abp_mode = ABP_TARGET_MODE.get(model_name, "global")
    gan_tanh = model_name == "p2e_wgan"

    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_state(model, ckpt["model"])
    want_bp = bool(model.has_bp_head) and tgt_is_abp
    work_mult = int(model.work_multiple)
    seq_len = int(cfg.data.seq_len)
    clip_lo = float(cfg.data.abp_clip_low)
    clip_hi = float(cfg.data.abp_clip_high)

    ds = build_baseline_dataset(cfg, split)
    if num > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(num, len(ds)))))
    loader = DataLoader(ds, batch_size=int(cfg.training.val_batch_size), shuffle=False,
                        num_workers=int(cfg.training.num_workers))

    preds, trues = [], []
    gen_predict = (lambda x: model.generator(x)) if gan_tanh else None
    for batch in loader:
        x = pad_to_multiple(batch["x"].to(device), work_mult)
        if gen_predict is not None:
            wave = gen_predict(x)
            bp_pred = None
        else:
            out = model(x, want_bp=want_bp)
            wave = out["wave"]
            bp_pred = out.get("bp") if want_bp else None
        pred = reconstruct_pred(wave, seq_len=seq_len, tgt_is_abp=tgt_is_abp, abp_mode=abp_mode,
                                clip_lo=clip_lo, clip_hi=clip_hi, bp_pred=bp_pred, gan_tanh=gan_tanh)
        preds.append(pred.cpu())
        trues.append(reconstruct_true(batch, tgt_is_abp=tgt_is_abp).cpu())

    pred = torch.cat(preds)
    true = torch.cat(trues)
    if tgt_is_abp:
        report = evaluate(pred, true)
    else:
        report = {"waveform": waveform_metrics(pred, true)}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    # sidecar metadata so summarize_baselines.py can label rows unambiguously
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({"model": model_name, "direction": direction, "split": split,
                   "n_eval": int(len(true)), "tgt_is_abp": bool(tgt_is_abp),
                   "ckpt": os.path.abspath(ckpt_path)}, f, indent=2)
    logger.info("[%s %s] %s\n%s", model_name, direction,
                f"N={len(true)}", format_report(report))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["mdvisco", "nabnet", "patchtst", "ppg2abp", "p2e_wgan", "wavenet"])
    p.add_argument("--direction", required=True,
                   choices=["ecg2abp", "ppg2abp", "ecg2ppg", "ppg2ecg", "ecg_ppg2abp"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num", type=int, default=-1)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    args, extra = p.parse_known_args()

    set_seed(args.seed)
    cfg_path = args.config or f"bpflow_baselines/config/{args.model}.yaml"
    cfg = load_config(cfg_path, overrides=overrides_from_extra(extra))
    cfg.model.name = args.model
    cfg.baseline.direction = args.direction
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt) or ".", f"infer_{args.direction}")
    report = run(cfg, args.ckpt, args.split, args.num, out_dir)
    # Append test/* to the finetune SwanLab run (its id was saved next to the
    # checkpoint), so test scores land in the SAME run as finetune — no separate
    # infer run. No-op unless use_swanlab=true and that id file exists.
    log_test_to_run(cfg, report, os.path.dirname(os.path.abspath(args.ckpt)))


if __name__ == "__main__":
    main()
