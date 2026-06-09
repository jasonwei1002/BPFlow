"""BPFlow inference + evaluation.

Loads a trained checkpoint, generates ABP waveforms from ECG+PPG for a chosen
split (default: the subject-disjoint CalFree test set), denormalizes to mmHg,
and reports waveform + clinical (AAMI/BHS) metrics.

Run:
    python -m bpflow.infer --config bpflow/config/gpu.yaml \
        --ckpt output/bpflow_gpu_p10/checkpoint_latest.pth --split test --num 2000
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import build_dataset
from .eval import evaluate, format_report
from .model import build_model
from .sampling import build_flow_matching, sample_abp
from .trainer_utils import load_config, pick_device, set_seed

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
        if use_ema and "model_ema" in ckpt:
            state = ckpt["model_ema"]
            logger.info("Loading EMA weights from %s", ckpt_path)
        else:
            state = ckpt["model"]
            logger.info("Loading model weights from %s", ckpt_path)
    else:
        state = ckpt  # flat state_dict
        logger.info("Loading flat state_dict from %s", ckpt_path)
    # strict=False only to tolerate non-persistent buffers (none are saved);
    # any real missing/unexpected key means a wrong/incompatible checkpoint.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model architecture "
            f"(missing={missing}, unexpected={unexpected}). "
            "Use the config the checkpoint was trained with."
        )


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = load_config(args.config)

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
    cfg_s = float(args.cfg if args.cfg is not None else cfg.training.cfg_strength)

    preds, gts = [], []
    desc = f"infer (rank0 shard of {total})" if distributed else f"infer ({total})"
    for batch in tqdm(loader, desc=desc, disable=not is_main):
        out = sample_abp(
            model, fm, batch["cond_patches"], generator=gen, device=device,
            abp_mean=float(cfg.data.abp_mean), abp_std=float(cfg.data.abp_std), cfg_strength=cfg_s,
        )
        preds.append(out.cpu())
        gts.append(batch["abp_raw"].cpu())

    pred = torch.cat(preds, dim=0) if preds else None
    gt = torch.cat(gts, dim=0) if gts else None

    if distributed:
        import torch.distributed as dist

        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (pred, gt))
        dist.barrier()
        dist.destroy_process_group()
        if not is_main:
            return
        pred = torch.cat([p for p, _ in gathered if p is not None], dim=0)
        gt = torch.cat([g for _, g in gathered if g is not None], dim=0)

    report = evaluate(pred, gt)
    logger.info("evaluated %d segments across %d process(es)", pred.shape[0], world_size)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    logger.info("\n%s", format_report(report))
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
    ap.add_argument("--cfg", type=float, default=None, help="override CFG strength")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=14159265)
    ap.add_argument("--out", default="output/infer")
    ap.add_argument("--save-waveforms", action="store_true")
    ap.add_argument("--plot", type=int, default=6, help="num example plots (0 = none)")
    args = ap.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
