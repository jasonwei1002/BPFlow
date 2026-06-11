"""Standalone subject-disjoint K-shot evaluation of a trained meta checkpoint.

For each held-out subject, adapt the per-subject context phi on K calibration
segments and predict the rest; sweep K to get the calibration curve. K=0 is the
calibration-free baseline. Honest by construction: CalFree (test_npy) subjects are
a different file than the meta-training Train_Subset, so they are never seen.

    python -m bpflow.kshot_eval --config bpflow/config/meta.yaml --ckpt output/<ts>/checkpoint_best.pth --use-ema
    python -m bpflow.kshot_eval --config bpflow/config/meta.yaml --ckpt <path> --ks 0,1,3,5,10 --num 200
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from .eval import format_report
from .meta import kshot_evaluate
from .meta_data import load_bp_z, subject_groups
from .meta_train import _jsonable, _parse_ks
from .model import build_model
from .sampling import build_flow_matching
from .trainer_utils import load_config, pick_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="K-shot subject-disjoint evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ks", default=None, help="comma list, e.g. 0,1,3,5,10 (default: meta.eval_ks)")
    ap.add_argument("--num", type=int, default=None, help="cap subjects (default: meta.eval_max_subjects)")
    ap.add_argument("--max-query", type=int, default=None, help="cap query segs/subject")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="which npy's subjects to evaluate (test=CalFree, honest)")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not bool(cfg.model.use_context):
        raise ValueError("kshot_eval needs a context model (model.use_context: true)")
    device = pick_device(args.device or str(cfg.training.device))
    ks_list = _parse_ks(args.ks) if args.ks else _parse_ks(cfg.meta.eval_ks)
    max_subjects = args.num if args.num is not None else int(cfg.meta.eval_max_subjects)
    max_query = args.max_query if args.max_query is not None else int(cfg.meta.eval_max_query)

    npy_path = str(cfg.data.test_npy) if args.split == "test" else str(cfg.data.train_npy)
    groups = subject_groups(npy_path, min_segs=max(ks_list) + 1)
    arr = np.load(npy_path, mmap_mode="r")
    bp_z = load_bp_z(npy_path, cfg)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key = "model_ema" if (args.use_ema and "model_ema" in ckpt) else "model"
    if key == "model_ema":
        names = [n for n, _ in model.named_parameters()]
        sd = dict(model.state_dict())
        for n in names:
            sd[n] = ckpt["model_ema"][n]
        model.load_state_dict(sd)
        logger.info("loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        logger.info("loaded model weights")
    model.eval()
    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device).manual_seed(int(cfg.training.seed))

    logger.info("K-shot eval on %s: %d subjects (split=%s, Ks=%s, max_query=%d)",
                os.path.basename(npy_path), len(groups), args.split, ks_list, max_query)
    reports = kshot_evaluate(
        model, fm, cfg, arr, bp_z, groups, list(groups.keys()), ks_list,
        device=device, generator=gen, max_subjects=max_subjects, max_query=max_query,
    )
    for k in sorted(reports):
        logger.info("\n===== K = %d =====\n%s", k, format_report(reports[k]))

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    path = os.path.join(out_dir, f"kshot_{args.split}_metrics.json")
    with open(path, "w") as f:
        json.dump(_jsonable({str(k): r for k, r in reports.items()}), f, indent=2)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
