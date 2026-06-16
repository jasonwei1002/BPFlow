"""BPFlow inference + evaluation.

Loads a trained checkpoint, generates ABP waveforms from ECG+PPG for a chosen
split, denormalizes to mmHg, and reports waveform + clinical (AAMI/BHS) metrics.
The `test` split depends on the config: finetune.yaml (data.finetune true) -> the
held-out 10% of CalFree the finetune never trained on; otherwise -> the full
subject-disjoint CalFree test set.

Run (evaluate a finetuned checkpoint on the CalFree held-out test split):
    python -m bpflow.infer --config bpflow/config/finetune.yaml \
        --ckpt output/<finetune_ts>/checkpoint_best.pth --split test --num -1 --use-ema
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import MODALITY_MASK, MODALITY_ORDER, build_dataset, trained_modalities
from .eval import evaluate, format_report, segment_bp
from .model import build_model
from .sampling import build_flow_matching, sample_abp
from .trainer_utils import load_config, load_model_state, pick_device, set_seed

logger = logging.getLogger(__name__)


def _load_weights(model: torch.nn.Module, ckpt_path: str, use_ema: bool, cfg) -> set:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        # assert normalization constants match (else denormalized mmHg is biased)
        for key in ("abp_mean", "abp_std"):
            if key in ckpt and abs(float(ckpt[key]) - float(getattr(cfg.data, key))) > 1e-6:
                raise ValueError(
                    f"{key} mismatch: ckpt={ckpt[key]} cfg={getattr(cfg.data, key)}. "
                    "Use the config the checkpoint was trained with."
                )
        # The set of directions the checkpoint actually TRAINED on (caller picks
        # which to infer; inferring an untrained direction would feed the absent
        # stream's null token an unseen state → junk output). The trained set is:
        #   - modality_dropout: every modality with prob > 0 (the unified case;
        #     degenerate probs like [1,0,0] correctly collapse to one direction).
        #   - specialist (no dropout): just the fixed cond_modality.
        #   - old checkpoints predate both fields → {ecg_ppg}.
        ck_cfg = ckpt.get("config")
        ck_data = ck_cfg.get("data", {}) if isinstance(ck_cfg, dict) else {}
        ck_modality = str(ck_data.get("cond_modality", "ecg_ppg"))
        ck_dropout = bool(ck_data.get("modality_dropout", False))
        trained = trained_modalities(
            ck_modality, ck_dropout, ck_data.get("modality_dropout_probs")
        )
        if use_ema and "model_ema" in ckpt:
            state = ckpt["model_ema"]
            logger.info("Loading EMA weights from %s", ckpt_path)
        else:
            state = ckpt["model"]
            logger.info("Loading model weights from %s", ckpt_path)
    else:
        state = ckpt  # flat state_dict (no config) → assume the historical default
        trained = {"ecg_ppg"}
        logger.info("Loading flat state_dict from %s", ckpt_path)
    # Drops removed-feature keys (old CFG/meta), tolerates new params missing in
    # older checkpoints (null tokens), and flags any real architecture mismatch.
    load_model_state(model, state)
    return trained


def _build_reports(pred, gt, bp, source: str) -> dict:
    """Clinical/waveform reports for one direction's gathered predictions.

    ``source`` (waveform/csv/both) picks the BP TRUE: per-beat on the true wave
    (``true_waveform``) and/or the CSV cuff label (``true_csv``). PRED is always
    per-beat from the generated wave; waveform MAE/RMSE/Pearson are source-agnostic.
    """
    true_bp = {"SBP": bp[:, 0], "DBP": bp[:, 1], "MAP": bp[:, 2]} if bp is not None else None
    if source in ("csv", "both") and true_bp is None:
        raise RuntimeError(
            f"eval_true_source={source!r} needs the CSV cuff label but no bp_true was "
            "loaded (is the sibling CSV present and data.eval_true_source set?)."
        )
    reports: dict = {}
    if source in ("waveform", "both"):
        reports["true_waveform"] = evaluate(pred, gt)
    if source in ("csv", "both"):
        reports["true_csv"] = evaluate(pred, gt, true_bp)
    return reports


@torch.no_grad()
def _sample_and_gather(modality, loader, model, fm, gen, device, cfg, seed,
                       distributed, world_size, is_main, total):
    """Sample the whole (sharded) loader for ONE input direction, gather to rank 0.

    Overrides each batch's cond_mask with ``MODALITY_MASK[modality]`` so a single
    dataset build serves every direction (cond_patches always carries both
    streams; the model nulls the absent one). Re-seeds the generator so every
    direction sees identical initial noise → a paired comparison. ALL ranks must
    call this together — the all_gather is a collective. Returns ``(pred, gt, bp)``
    on rank 0 (cuff ``bp`` may be None); ``(None, None, None)`` on other ranks.
    """
    gen.manual_seed(seed)  # same noise across directions → paired comparison
    mask_row = torch.tensor(MODALITY_MASK[modality], dtype=torch.float32)
    use_demo = bool(cfg.model.use_demo)
    abp_mean, abp_std = float(cfg.data.abp_mean), float(cfg.data.abp_std)
    preds, gts, bps = [], [], []
    tag = f"infer:{modality} (rank0 shard of {total})" if distributed else f"infer:{modality} ({total})"
    for batch in tqdm(loader, desc=tag, disable=not is_main):
        demo = (batch["demo_cont"], batch["demo_gender"]) if use_demo and "demo_cont" in batch else None
        bs = batch["cond_patches"].shape[0]
        out = sample_abp(
            model, fm, batch["cond_patches"], generator=gen, device=device,
            abp_mean=abp_mean, abp_std=abp_std,
            demo=demo, cond_mask=mask_row.repeat(bs, 1),
        )
        preds.append(out.cpu())
        gts.append(batch["abp_raw"].cpu())
        if "bp_true" in batch:  # CSV cuff [SBP, DBP, MAP] -> clinical TRUE
            bps.append(batch["bp_true"].cpu())
    pred = torch.cat(preds, dim=0) if preds else None
    gt = torch.cat(gts, dim=0) if gts else None
    bp = torch.cat(bps, dim=0) if bps else None

    if distributed:
        import torch.distributed as dist

        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (pred, gt, bp))
        dist.barrier()
        if not is_main:
            return None, None, None
        pred_parts = [p for p, _, _ in gathered if p is not None]
        gt_parts = [g for _, g, _ in gathered if g is not None]
        bp_parts = [b for _, _, b in gathered if b is not None]
        pred = torch.cat(pred_parts, dim=0) if pred_parts else None  # None if every shard empty
        gt = torch.cat(gt_parts, dim=0) if gt_parts else None
        bp = torch.cat(bp_parts, dim=0) if bp_parts else None
    return pred, gt, bp


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = load_config(args.config)
    # --true-source overrides the config so the dataset loads the CSV cuff label
    # (bp_true) and the report uses the same source.
    if args.true_source is not None:
        cfg.data.eval_true_source = args.true_source

    # Distributed (single-node multi-GPU via torchrun). When WORLD_SIZE == 1 this
    # is identical to the old single-process path.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    is_main = rank == 0

    want = str(cfg.training.device) if args.device == "auto" else args.device
    if distributed:
        import torch.distributed as dist

        use_cuda = torch.cuda.is_available() and want != "cpu"
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = pick_device(want)

    ds = build_dataset(cfg, args.split)
    if args.num > 0 and args.num < len(ds):
        ds = Subset(ds, list(range(args.num)))
    total = len(ds)
    if distributed:
        # Strided shard: exact coverage, no padding/duplicates. Metrics are
        # set-level (MAE/RMSE/Pearson/AAMI/BHS) so the gather order is irrelevant.
        ds = Subset(ds, list(range(rank, total, world_size)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(cfg).to(device).eval()
    trained = _load_weights(model, args.ckpt, args.use_ema, cfg)

    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device)

    # Which input direction(s) to evaluate. 'all' (default) = every direction the
    # checkpoint trained on (unified -> 3; specialist -> its 1); else the pinned
    # one, which must be in the trained set (an untrained direction yields junk).
    if args.cond_modality == "all":
        modalities = [m for m in MODALITY_ORDER if m in trained]
    elif args.cond_modality in trained:
        modalities = [args.cond_modality]
    else:
        raise ValueError(
            f"--cond-modality {args.cond_modality!r} not in the checkpoint's trained "
            f"modalities {sorted(trained)}. Infer only a direction the checkpoint saw."
        )
    # Validate up front (BEFORE any collective) so every rank fails together — a
    # late/asymmetric raise inside the modality loop would deadlock DDP.
    if not modalities:  # e.g. a checkpoint whose modality_dropout_probs are all 0
        raise ValueError(f"no trained directions to evaluate (trained={sorted(trained)})")
    source = str(args.true_source or cfg.data.eval_true_source)
    if source not in ("waveform", "csv", "both"):
        raise ValueError(f"eval_true_source must be waveform/csv/both, got {source!r}")

    out_dir = Path(args.out)
    multi = len(modalities) > 1
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("evaluating modalities %s [true=%s]", modalities, source)

    # Every rank must enter _sample_and_gather for EVERY modality (its all_gather is
    # a collective). So rank-0-only reporting is wrapped in try/except and the first
    # error is DEFERRED: the loop keeps running (all collectives complete), the group
    # is torn down, and only then is the error re-raised — a reporting failure (e.g.
    # missing CSV under --true-source csv) can never strand the other ranks mid-loop.
    results: dict = {}
    deferred_error = None
    for m in modalities:
        pred, gt, bp = _sample_and_gather(
            m, loader, model, fm, gen, device, cfg, args.seed,
            distributed, world_size, is_main, total,
        )
        if not is_main:
            continue  # keep looping so every direction's all_gather stays aligned
        try:
            if pred is None:
                raise RuntimeError(f"no segments evaluated for modality {m!r}")
            reports = _build_reports(pred, gt, bp, source)
            results[m] = next(iter(reports.values())) if len(reports) == 1 else reports
            logger.info("[%s] %d segments across %d process(es)", m, pred.shape[0], world_size)
            for name, rep in reports.items():
                logger.info("[%s/%s]\n%s", m, name, format_report(rep))
            suffix = f"_{m}" if multi else ""
            arrow = f"{m.upper().replace('_', '+')} → ABP"  # e.g. "ECG+PPG -> ABP"
            if args.save_waveforms:
                import numpy as np

                np.save(out_dir / f"pred_mmhg{suffix}.npy", pred.numpy())
                np.save(out_dir / f"gt_mmhg{suffix}.npy", gt.numpy())
                logger.info("waveforms -> %s", out_dir)
            if args.plot > 0:
                _plot(pred, gt, out_dir, args.plot, suffix, title=arrow)
            if args.bland_altman:
                # Per-beat BP agreement, one figure per truth source (mirrors the
                # report keys). pred_bp is computed once and reused across sources.
                pred_bp = segment_bp(pred)
                if source in ("waveform", "both"):
                    _bland_altman(pred_bp, segment_bp(gt), out_dir, f"waveform{suffix}",
                                  title=f"{arrow}  (waveform truth)")
                if source in ("csv", "both"):  # bp guaranteed non-None by _build_reports
                    true_csv = {"SBP": bp[:, 0], "DBP": bp[:, 1], "MAP": bp[:, 2]}
                    _bland_altman(pred_bp, true_csv, out_dir, f"csv{suffix}",
                                  title=f"{arrow}  (cuff truth)")
        except Exception as e:  # noqa: BLE001 — defer until all collectives are done
            deferred_error = deferred_error or e

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    if not is_main:
        return
    if deferred_error is not None:
        raise deferred_error

    # single direction -> flat report (back-compat); multiple -> {modality: report}
    payload = results[modalities[0]] if not multi else results
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    logger.info("metrics (%s) -> %s", "+".join(modalities), out_dir / "metrics.json")


def _plot(pred: torch.Tensor, gt: torch.Tensor, out_dir: Path, k: int,
          suffix: str = "", title: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = min(k, pred.shape[0])
        fig, axes = plt.subplots(k, 1, figsize=(10, 2.2 * k))
        if k == 1:
            axes = [axes]
        for j in range(k):
            axes[j].plot(gt[j].numpy(), label="GT", lw=1.0)
            axes[j].plot(pred[j].numpy(), label="gen", lw=1.0, alpha=0.8)
            axes[j].set_ylabel("mmHg")
            axes[j].legend(loc="upper right", fontsize=8)
        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
        fig.savefig(out_dir / f"infer_recon{suffix}.png", dpi=300)
        plt.close(fig)
        logger.info("plot -> %s", out_dir / f"infer_recon{suffix}.png")
    except Exception as e:
        logger.warning("plot skipped: %s", e)


def _bland_altman(pred_bp: dict, true_bp: dict, out_dir: Path, name: str,
                  title: str = "") -> None:
    """3-panel (SBP/DBP/MAP) Bland-Altman agreement plot.

    Per BP value: x = (pred+true)/2, y = pred-true; horizontal lines mark the bias
    (mean diff) and the 95% limits of agreement (bias +/- 1.96 SD). ``name`` is the
    truth source + modality suffix, e.g. ``waveform`` / ``csv_ecg``; ``title`` is the
    figure suptitle (e.g. "ECG+PPG -> ABP  (cuff truth)").
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = ("SBP", "DBP", "MAP")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, key in zip(axes, keys):
            p = pred_bp[key].float().numpy()
            t = true_bp[key].float().numpy()
            mean = (p + t) / 2.0
            diff = p - t
            bias = float(diff.mean())
            sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
            hi, lo = bias + 1.96 * sd, bias - 1.96 * sd
            ax.scatter(mean, diff, s=6, alpha=0.3, edgecolors="none")
            ax.axhline(bias, color="C1", lw=1.2, label=f"bias {bias:+.2f}")
            ax.axhline(hi, color="C3", ls="--", lw=1.0, label=f"+1.96SD {hi:+.2f}")
            ax.axhline(lo, color="C3", ls="--", lw=1.0)
            ax.set_title(f"{key}  (n={diff.size})")
            ax.set_xlabel("mean of pred & true (mmHg)")
            ax.set_ylabel("pred - true (mmHg)")
            ax.legend(loc="upper right", fontsize=7)
        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95) if title else None)
        path = out_dir / f"bland_altman_{name}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logger.info("Bland-Altman -> %s", path)
    except Exception as e:
        logger.warning("Bland-Altman skipped: %s", e)


def main() -> None:
    rank = int(os.environ.get("RANK", 0))  # only rank 0 prints INFO under torchrun
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description="BPFlow inference + evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cond-modality", default="all",
                    choices=["all", "ecg_ppg", "ecg", "ppg"],
                    help="input direction(s) to evaluate. 'all' (default) = every "
                         "direction the checkpoint trained on (unified -> 3, "
                         "specialist -> 1); or pin one. Each appears under its name "
                         "in metrics.json when more than one is evaluated.")
    ap.add_argument("--num", type=int, default=-1, help="max segments (-1 = all)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--true-source", default=None, choices=["waveform", "csv", "both"],
                    help="clinical TRUE source (default: data.eval_true_source). "
                         "waveform=per-beat on true wave; csv=cuff label; both=report both")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=14159265)
    ap.add_argument("--out", default="output/infer")
    ap.add_argument("--save-waveforms", action="store_true")
    ap.add_argument("--plot", type=int, default=6, help="num example plots (0 = none)")
    ap.add_argument("--bland-altman", action=argparse.BooleanOptionalAction, default=True,
                    help="SBP/DBP/MAP Bland-Altman agreement plot per truth source "
                         "(--no-bland-altman to skip)")
    args = ap.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
