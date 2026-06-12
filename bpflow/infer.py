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

from .data import build_dataset, trained_modalities
from .eval import evaluate, format_report
from .model import build_model
from .sampling import build_flow_matching, sample_abp
from .trainer_utils import load_config, load_model_state, pick_device, set_seed

logger = logging.getLogger(__name__)


def _load_weights(model: torch.nn.Module, ckpt_path: str, use_ema: bool, cfg) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        # assert normalization constants match (else denormalized mmHg is biased)
        for key in ("abp_mean", "abp_std"):
            if key in ckpt and abs(float(ckpt[key]) - float(getattr(cfg.data, key))) > 1e-6:
                raise ValueError(
                    f"{key} mismatch: ckpt={ckpt[key]} cfg={getattr(cfg.data, key)}. "
                    "Use the config the checkpoint was trained with."
                )
        # Only infer with a modality the checkpoint actually TRAINED on — else the
        # absent stream's null token is in an unseen state and the output is junk.
        # The trained set is:
        #   - modality_dropout: every modality with prob > 0 (the unified case;
        #     degenerate probs like [1,0,0] correctly collapse to one direction).
        #   - specialist (no dropout): just the fixed cond_modality.
        #   - old checkpoints predate both fields → {ecg_ppg}.
        ck_cfg = ckpt.get("config")
        ck_data = ck_cfg.get("data", {}) if isinstance(ck_cfg, dict) else {}
        ck_modality = str(ck_data.get("cond_modality", "ecg_ppg"))
        ck_dropout = bool(ck_data.get("modality_dropout", False))
        cfg_modality = str(getattr(cfg.data, "cond_modality", "ecg_ppg"))
        trained = trained_modalities(
            ck_modality, ck_dropout, ck_data.get("modality_dropout_probs")
        )
        if cfg_modality not in trained:
            raise ValueError(
                f"cond_modality {cfg_modality!r} not in the checkpoint's trained "
                f"modalities {sorted(trained)} (modality_dropout={ck_dropout}). "
                "Infer only with a modality the checkpoint actually saw in training."
            )
        if use_ema and "model_ema" in ckpt:
            state = ckpt["model_ema"]
            logger.info("Loading EMA weights from %s", ckpt_path)
        else:
            state = ckpt["model"]
            logger.info("Loading model weights from %s", ckpt_path)
    else:
        state = ckpt  # flat state_dict
        logger.info("Loading flat state_dict from %s", ckpt_path)
    # Drops removed-feature keys (old CFG/meta), tolerates new params missing in
    # older checkpoints (null tokens), and flags any real architecture mismatch.
    load_model_state(model, state)


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
    _load_weights(model, args.ckpt, args.use_ema, cfg)

    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    preds, gts, bps = [], [], []
    use_demo = bool(cfg.model.use_demo)
    desc = f"infer (rank0 shard of {total})" if distributed else f"infer ({total})"
    for batch in tqdm(loader, desc=desc, disable=not is_main):
        demo = (batch["demo_cont"], batch["demo_gender"]) if use_demo and "demo_cont" in batch else None
        out = sample_abp(
            model, fm, batch["cond_patches"], generator=gen, device=device,
            abp_mean=float(cfg.data.abp_mean), abp_std=float(cfg.data.abp_std),
            demo=demo, cond_mask=batch.get("cond_mask"),
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
        dist.destroy_process_group()
        if not is_main:
            return
        pred = torch.cat([p for p, _, _ in gathered if p is not None], dim=0)
        gt = torch.cat([g for _, g, _ in gathered if g is not None], dim=0)
        bp_parts = [b for _, _, b in gathered if b is not None]
        bp = torch.cat(bp_parts, dim=0) if bp_parts else None

    # Clinical TRUE source(s): waveform (per-beat on the true wave), csv (cuff label),
    # or both. PRED is always per-beat from the generated wave.
    source = str(args.true_source or cfg.data.eval_true_source)
    true_bp = {"SBP": bp[:, 0], "DBP": bp[:, 1], "MAP": bp[:, 2]} if bp is not None else None
    if source in ("csv", "both") and true_bp is None:
        raise RuntimeError(
            f"eval_true_source='{source}' needs the CSV cuff label but no bp_true was "
            "loaded (is the sibling CSV present and data.eval_true_source set?)."
        )
    reports: dict = {}
    if source in ("waveform", "both"):
        reports["true_waveform"] = evaluate(pred, gt)
    if source in ("csv", "both"):
        reports["true_csv"] = evaluate(pred, gt, true_bp)
    logger.info("evaluated %d segments across %d process(es) [true=%s]",
                pred.shape[0], world_size, source)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # single source -> flat report (back-compat); both -> {true_waveform, true_csv}
    payload = next(iter(reports.values())) if len(reports) == 1 else reports
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    for name, rep in reports.items():
        logger.info("[%s]\n%s", name, format_report(rep))
    logger.info("metrics -> %s", out_dir / "metrics.json")

    if args.save_waveforms:
        import numpy as np

        np.save(out_dir / "pred_mmhg.npy", pred.numpy())
        np.save(out_dir / "gt_mmhg.npy", gt.numpy())
        logger.info("waveforms -> %s", out_dir)

    if args.plot > 0:
        _plot(pred, gt, out_dir, args.plot)


def _plot(pred: torch.Tensor, gt: torch.Tensor, out_dir: Path, k: int) -> None:
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
        fig.tight_layout()
        fig.savefig(out_dir / "infer_recon.png", dpi=110)
        plt.close(fig)
        logger.info("plot -> %s", out_dir / "infer_recon.png")
    except Exception as e:
        logger.warning("plot skipped: %s", e)


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
    args = ap.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
