"""K-shot calibration sweep for BPFlow.

Measures how ABP reconstruction improves as the cuff-calibration support grows.
For each K in ``--ks`` it generates ABP on a split (default CalFree test) using
the FIRST K of each query's same-subject calibration segments. The K-shots are
NESTED (K=1 is a subset of K=3 ...) and every K reuses the same initial ODE
noise per batch, so the only variable across the sweep is the amount of
calibration. K=0 is the calibration-free baseline (``calib`` is dropped, so
``calib_emb`` is absent == the K=0 training regime).

Also reports, at the largest K, a BP-gap stratification: how SBP/DBP error grows
as the query's true BP drifts away from the calibration mean. PulseDB has no
timestamps, so this gap (not elapsed time) is the stand-in for "how far can one
calibration be trusted".

Requires ``model.use_calib`` and a checkpoint trained WITH calibration.

Run:
    python -m bpflow.kshot_sweep --config bpflow/config/gpu.yaml \
        --ckpt output/<ts>/checkpoint_best.pth --use-ema --ks 0,1,3,5,10 --num 2000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import build_dataset
from .eval import evaluate, format_report, segment_bp
from .infer import _load_weights
from .model import build_model
from .sampling import build_flow_matching, sample_abp
from .trainer_utils import load_config, pick_device, set_seed

logger = logging.getLogger(__name__)


def _truncate_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the first ``k`` calibration slots valid (nested K-shot)."""
    m = mask.clone()
    if k < m.shape[1]:
        m[:, k:] = 0.0
    return m


def _summary_row(k: int, report: Dict[str, dict]) -> Dict[str, float]:
    w, sbp, dbp, mp = report["waveform"], report["SBP"], report["DBP"], report["MAP"]
    return {
        "K": k,
        "MAE": w["MAE"], "RMSE": w["RMSE"], "Pearson": w["Pearson"],
        "SBP_ME": sbp["AAMI"]["ME"], "SBP_SDE": sbp["AAMI"]["SDE"], "SBP_AAMI": sbp["AAMI"]["pass"],
        "DBP_ME": dbp["AAMI"]["ME"], "DBP_SDE": dbp["AAMI"]["SDE"], "DBP_AAMI": dbp["AAMI"]["pass"],
        "MAP_ME": mp["AAMI"]["ME"],
    }


def _stratify_by_bp_gap(
    pred: torch.Tensor, gt: torch.Tensor, sup_sbp: torch.Tensor
) -> List[dict]:
    """SBP/DBP MAE binned by |query true SBP - calibration-mean SBP| (mmHg)."""
    pred_bp, true_bp = segment_bp(pred), segment_bp(gt)
    gap = (true_bp["SBP"] - sup_sbp).abs()
    edges = [0.0, 5.0, 10.0, 20.0, float("inf")]
    out: List[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (gap >= lo) & (gap < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        out.append({
            "sbp_gap_mmHg": f"[{lo:g},{hi:g})",
            "n": n,
            "SBP_MAE": (pred_bp["SBP"][sel] - true_bp["SBP"][sel]).abs().mean().item(),
            "DBP_MAE": (pred_bp["DBP"][sel] - true_bp["DBP"][sel]).abs().mean().item(),
        })
    return out


@torch.no_grad()
def run_sweep(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = load_config(args.config)
    if not bool(cfg.model.use_calib):
        raise ValueError("kshot_sweep needs model.use_calib=true and a calib-trained ckpt.")
    ks = sorted({int(x) for x in args.ks.split(",")})
    max_k = max(ks)
    if max_k > int(cfg.data.calib_k_max):
        raise ValueError(f"max K {max_k} exceeds calib_k_max {cfg.data.calib_k_max}")

    # Attach exactly max_k fixed calibration segments per query; truncate per K.
    OmegaConf.set_struct(cfg, False)
    cfg.data.calib_eval_k = max_k
    device = pick_device(str(cfg.training.device) if args.device == "auto" else args.device)

    ds = build_dataset(cfg, args.split)
    if args.num > 0 and args.num < len(ds):
        # NOTE: segments are subject-contiguous, so this takes the first subjects.
        ds = Subset(ds, list(range(args.num)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(cfg).to(device).eval()
    _load_weights(model, args.ckpt, args.use_ema, cfg)
    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device)
    cfg_s = float(args.cfg if args.cfg is not None else cfg.training.cfg_strength)
    use_demo = bool(cfg.model.use_demo)
    sm, ss = float(cfg.data.bp_sbp_mean), float(cfg.data.bp_sbp_std)

    preds: Dict[int, List[torch.Tensor]] = {k: [] for k in ks}
    gts: List[torch.Tensor] = []
    sup_sbp_l: List[torch.Tensor] = []  # calibration-mean SBP (mmHg) at max_k

    for bi, batch in enumerate(tqdm(loader, desc="kshot")):
        cond = batch["cond_patches"]
        cc, cbp, cm = batch["calib_cond"], batch["calib_bp"], batch["calib_mask"]
        demo = (batch["demo_cont"], batch["demo_gender"]) if use_demo and "demo_cont" in batch else None
        gts.append(batch["abp_raw"])
        # calibration-mean BP over the valid max_k support segments (denormalized)
        m = cm.unsqueeze(-1)  # (B,K,1)
        bp_mean_z = (cbp * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B,2) z-scored
        sup_sbp_l.append(bp_mean_z[:, 0] * ss + sm)
        for k in ks:
            gen.manual_seed(args.seed + bi)  # same init noise across K for this batch
            calib = None if k == 0 else (cc, cbp, _truncate_mask(cm, k))
            out = sample_abp(
                model, fm, cond, generator=gen, device=device,
                abp_mean=float(cfg.data.abp_mean), abp_std=float(cfg.data.abp_std),
                cfg_strength=cfg_s, demo=demo, calib=calib,
            )
            preds[k].append(out.cpu())

    gt = torch.cat(gts)
    reports = {k: evaluate(torch.cat(preds[k]), gt) for k in ks}
    table = [_summary_row(k, reports[k]) for k in ks]
    for k in ks:
        logger.info("K=%d (%d segments)\n%s", k, gt.shape[0], format_report(reports[k]))

    strat = []
    if max_k > 0:
        strat = _stratify_by_bp_gap(
            torch.cat(preds[max_k]), gt, torch.cat(sup_sbp_l)
        )

    logger.info("\n%s", _format_sweep(table, strat, max_k))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ks": ks, "split": args.split, "n_segments": int(gt.shape[0]),
        "use_ema": bool(args.use_ema), "cfg_strength": cfg_s,
        "table": table, "reports": reports,
        "bp_gap_stratification": {"at_K": max_k, "bins": strat},
    }
    (out_dir / "kshot_sweep.json").write_text(json.dumps(payload, indent=2))
    logger.info("results -> %s", out_dir / "kshot_sweep.json")
    if args.plot:
        _plot_curve(table, out_dir)


def _format_sweep(table: List[dict], strat: List[dict], max_k: int) -> str:
    lines = ["K-shot sweep (lower MAE / |ME| better; AAMI 1=pass):",
             "  K   MAE    RMSE   Pearson  SBP_ME  SBP_AAMI  DBP_ME  DBP_AAMI  MAP_ME"]
    for r in table:
        lines.append(
            f"  {r['K']:<3d} {r['MAE']:6.3f} {r['RMSE']:6.3f} {r['Pearson']:7.4f} "
            f"{r['SBP_ME']:+7.2f} {r['SBP_AAMI']:8.0f}  {r['DBP_ME']:+7.2f} "
            f"{r['DBP_AAMI']:8.0f}  {r['MAP_ME']:+7.2f}"
        )
    if strat:
        lines.append(f"\nBP-gap stratification at K={max_k} (|query SBP - calib-mean SBP|):")
        lines.append("  gap(mmHg)      n      SBP_MAE  DBP_MAE")
        for s in strat:
            lines.append(f"  {s['sbp_gap_mmHg']:<12s} {s['n']:6d}  {s['SBP_MAE']:7.2f}  {s['DBP_MAE']:7.2f}")
    return "\n".join(lines)


def _plot_curve(table: List[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [r["K"] for r in table]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(ks, [r["MAE"] for r in table], "o-", label="waveform MAE")
        ax1.set_xlabel("K (calibration shots)"); ax1.set_ylabel("MAE (mmHg)")
        ax1.set_title("Waveform MAE vs K"); ax1.grid(alpha=0.3)
        ax2.plot(ks, [abs(r["SBP_ME"]) for r in table], "o-", label="|SBP ME|")
        ax2.plot(ks, [abs(r["DBP_ME"]) for r in table], "s-", label="|DBP ME|")
        ax2.set_xlabel("K (calibration shots)"); ax2.set_ylabel("|mean error| (mmHg)")
        ax2.set_title("SBP/DBP bias vs K"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "kshot_curve.png", dpi=110)
        plt.close(fig)
        logger.info("curve -> %s", out_dir / "kshot_curve.png")
    except Exception as e:
        logger.warning("plot skipped: %s", e)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="BPFlow K-shot calibration sweep")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ks", default="0,1,3,5,10", help="comma-separated K values")
    ap.add_argument("--num", type=int, default=2000, help="max segments (-1 = all; subject-contiguous)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--cfg", type=float, default=None, help="override CFG strength")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=14159265)
    ap.add_argument("--out", default="output/kshot")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
