"""BPFlow inference + evaluation.

Loads a trained checkpoint, generates ABP waveforms from ECG+PPG for a chosen
split, denormalizes to mmHg, and reports waveform + clinical (AAMI/BHS) metrics.
The `test` split depends on the config: pulsedb_finetune.yaml (data.finetune true) -> the
held-out 10% of CalFree the finetune never trained on; otherwise -> the full
subject-disjoint CalFree test set.

Run (evaluate a finetuned checkpoint on the CalFree held-out test split):
    python -m bpflow.infer --config bpflow/config/pulsedb_finetune.yaml \
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

from .data import TASK_ORDER, TASK_SPEC, build_dataset, trained_tasks, unpatchify
from .eval import evaluate, format_report, segment_bp, waveform_metrics
from .model import build_model
from .sampling import build_flow_matching, sample_target
from .trainer_utils import load_config, load_model_state, pick_device, set_seed

logger = logging.getLogger(__name__)


def _arrow(task: str) -> str:
    """Readable 'COND -> TARGET' label from a task name (ppg2abp -> 'PPG -> ABP')."""
    cond, tgt = task.split("2", 1)
    return f"{cond.upper().replace('_', '+')} -> {tgt.upper()}"


def _read_checkpoint(ckpt_path: str):
    """Load a checkpoint once (weights + stored config) so architecture restore and
    weight loading share a single read — checkpoints are large."""
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _restore_arch_from_ckpt(cfg, ckpt) -> None:
    """Rebuild the model with the architecture the checkpoint was TRAINED with.

    Training selects architecture via CLI overrides (e.g. ``model.stream_fusion=late``
    for the late-fusion ablation), but ``bpflow.infer`` takes no dotted overrides — so
    without this a checkpoint trained with a non-default architecture can't be rebuilt
    and ``load_state_dict`` fails on the mismatched keys. The checkpoint stores its full
    resolved config and every ``build_model`` input lives under ``cfg.model``, so copy
    those fields over in place. Data-side constants (``abp_mean``/``abp_std``) are left
    alone — ``_load_weights`` still asserts them. Old/flat checkpoints without a stored
    config are left as-is (the passed-in config wins).
    """
    ck_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    ck_model = ck_cfg.get("model") if isinstance(ck_cfg, dict) else None
    if not isinstance(ck_model, dict):
        return
    changed = {}
    for key, val in ck_model.items():
        if key in cfg.model and cfg.model[key] != val:
            changed[key] = (cfg.model[key], val)
            cfg.model[key] = val
    if changed:
        logger.info(
            "Restored architecture from checkpoint config: %s",
            ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in changed.items()),
        )


def _load_weights(model: torch.nn.Module, ckpt, use_ema: bool, cfg, ckpt_path: str = "") -> set:
    if isinstance(ckpt, dict) and "model" in ckpt:
        # assert normalization constants match (else denormalized mmHg is biased)
        for key in ("abp_mean", "abp_std"):
            if key in ckpt and abs(float(ckpt[key]) - float(getattr(cfg.data, key))) > 1e-6:
                raise ValueError(
                    f"{key} mismatch: ckpt={ckpt[key]} cfg={getattr(cfg.data, key)}. "
                    "Use the config the checkpoint was trained with."
                )
        # The set of TASKS the checkpoint actually TRAINED on (caller picks which to
        # evaluate — an untrained task yields junk). New checkpoints store
        # data.tasks/task_probs; trained_tasks() resolves the set. A checkpoint with
        # no tasks field (e.g. base default) → all five tasks.
        ck_cfg = ckpt.get("config")
        ck_data = ck_cfg.get("data", {}) if isinstance(ck_cfg, dict) else {}
        trained = trained_tasks(ck_data.get("tasks") or None, ck_data.get("task_probs") or None)
        if use_ema and "model_ema" in ckpt:
            state = ckpt["model_ema"]
            logger.info("Loading EMA weights from %s", ckpt_path)
        else:
            state = ckpt["model"]
            logger.info("Loading model weights from %s", ckpt_path)
    else:
        state = ckpt  # flat state_dict (no config) → assume the historical default
        trained = {"ecg_ppg2abp"}
        logger.info("Loading flat state_dict from %s", ckpt_path)
    # Drops removed-feature keys, flags any real architecture mismatch.
    load_model_state(model, state)
    return trained


def _build_report(pred, gt, target_idx: int) -> dict:
    """Report for one task's gathered predictions.

    ABP target (``target_idx == 0``): clinical + waveform metrics, with SBP/DBP/MAP
    derived per-beat from the true ABP wave (no definitional offset). ECG/PPG
    translation target: waveform-only (MAE/RMSE/Pearson in normalized [0,1] units;
    clinical BP metrics don't apply).
    """
    if target_idx != 0:  # ECG / PPG translation target -> waveform fidelity only
        return {"waveform": waveform_metrics(pred, gt)}
    return evaluate(pred, gt)


@torch.no_grad()
def _sample_and_gather(task, loader, model, fm, gen, device, cfg, seed,
                       distributed, world_size, is_main, total):
    """Sample the whole (sharded) loader for ONE task, gather to rank 0.

    ``task`` is a TASK_SPEC name (e.g. ppg2abp, ppg2ecg): it fixes the target
    modality + present conditions. A single dataset build serves every task
    (ecg/ppg patches always carried; absent streams masked out of joint attention).
    The ground truth is the TARGET modality's clean wave (abp_raw for ABP; the
    recentered ECG/PPG patches un-patchified + un-recentered otherwise). Re-seeds the
    generator so every task sees identical initial noise → a paired comparison. ALL
    ranks must call this together — the all_gather is a collective. Returns
    ``(pred, gt)`` on rank 0; ``(None, None)`` on other ranks.
    """
    gen.manual_seed(seed)  # same noise across tasks → paired comparison
    tidx, cp = TASK_SPEC[task]
    cp_row = torch.tensor(cp, dtype=torch.float32)
    abp_mean, abp_std = float(cfg.data.abp_mean), float(cfg.data.abp_std)
    cond_recenter = bool(cfg.data.cond_recenter)
    shift = 0.5 if cond_recenter else 0.0  # un-recenter ECG/PPG ground truth
    preds, gts = [], []
    tag = f"infer:{task} (rank0 shard of {total})" if distributed else f"infer:{task} ({total})"
    for batch in tqdm(loader, desc=tag, disable=not is_main):
        bs = batch["ecg_patches"].shape[0]
        out = sample_target(
            model, fm, batch["ecg_patches"], batch["ppg_patches"],
            torch.full((bs,), tidx, dtype=torch.long), cp_row.view(1, 3).repeat(bs, 1),
            generator=gen, device=device, abp_mean=abp_mean, abp_std=abp_std,
            cond_recenter=cond_recenter,
        )
        preds.append(out.cpu())
        if tidx == 0:
            gts.append(batch["abp_raw"].cpu())
        else:  # ECG/PPG target: clean wave = un-patchify the recentered patches + shift
            key = "ecg_patches" if tidx == 1 else "ppg_patches"
            gts.append((unpatchify(batch[key]) + shift).cpu())
    pred = torch.cat(preds, dim=0) if preds else None
    gt = torch.cat(gts, dim=0) if gts else None

    if distributed:
        import torch.distributed as dist

        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (pred, gt))
        dist.barrier()
        if not is_main:
            return None, None
        pred_parts = [p for p, _ in gathered if p is not None]
        gt_parts = [g for _, g in gathered if g is not None]
        pred = torch.cat(pred_parts, dim=0) if pred_parts else None  # None if every shard empty
        gt = torch.cat(gt_parts, dim=0) if gt_parts else None
    return pred, gt


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = load_config(args.config)
    # Optional sampling-steps override (for the quality-vs-steps ablation). Baked
    # into FlowMatching by build_flow_matching(cfg) below, so set it before that.
    if args.num_steps is not None:
        if args.num_steps < 1:
            raise ValueError(f"--num-steps must be >= 1, got {args.num_steps}")
        cfg.sampling.num_steps = int(args.num_steps)

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

    # Restore the checkpoint's own architecture (e.g. stream_fusion=late) BEFORE
    # building — infer takes no dotted overrides, so the config file alone can't
    # spell out an ablated architecture. Reads the ckpt once and reuses it below.
    ckpt = _read_checkpoint(args.ckpt)
    _restore_arch_from_ckpt(cfg, ckpt)
    model = build_model(cfg).to(device).eval()
    trained = _load_weights(model, ckpt, args.use_ema, cfg, args.ckpt)

    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device)

    # Which task(s) to evaluate. 'all' (default) = EVERY task the checkpoint trained
    # on (in TASK_ORDER); else the pinned one, which must be in the trained set (an
    # untrained task yields junk). ABP-target tasks get clinical+waveform; ECG/PPG
    # translation tasks get waveform-only.
    if args.task == "all":
        tasks_to_eval = [t for t in TASK_ORDER if t in trained]
    elif args.task in trained:
        tasks_to_eval = [args.task]
    else:
        raise ValueError(
            f"--task {args.task!r} not in the checkpoint's trained tasks "
            f"{sorted(trained)}. Evaluate only a task the checkpoint saw."
        )
    # Validate up front (BEFORE any collective) so every rank fails together — a
    # late/asymmetric raise inside the task loop would deadlock DDP.
    if not tasks_to_eval:
        raise ValueError(f"no trained tasks to evaluate (trained={sorted(trained)})")

    out_dir = Path(args.out)
    multi = len(tasks_to_eval) > 1
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("evaluating tasks %s", tasks_to_eval)

    # Every rank must enter _sample_and_gather for EVERY task (its all_gather is a
    # collective). So rank-0-only reporting is wrapped in try/except and the first
    # error is DEFERRED: the loop keeps running (all collectives complete), the group
    # is torn down, and only then is the error re-raised — a reporting failure (e.g.
    # an empty shard) can never strand the other ranks mid-loop.
    results: dict = {}
    deferred_error = None
    for t in tasks_to_eval:
        target_idx = TASK_SPEC[t][0]
        pred, gt = _sample_and_gather(
            t, loader, model, fm, gen, device, cfg, args.seed,
            distributed, world_size, is_main, total,
        )
        if not is_main:
            continue  # keep looping so every task's all_gather stays aligned
        try:
            if pred is None:
                raise RuntimeError(f"no segments evaluated for task {t!r}")
            report = _build_report(pred, gt, target_idx)
            results[t] = report
            logger.info("[%s] %d segments across %d process(es)", t, pred.shape[0], world_size)
            logger.info("[%s]\n%s", t, format_report(report))
            suffix = f"_{t}" if multi else ""
            arrow = _arrow(t)
            if args.save_waveforms:
                import numpy as np

                np.save(out_dir / f"pred{suffix}.npy", pred.numpy())
                np.save(out_dir / f"gt{suffix}.npy", gt.numpy())
                logger.info("waveforms -> %s", out_dir)
            if args.plot > 0:
                _plot(pred, gt, out_dir, args.plot, suffix, title=arrow)
            # Bland-Altman is BP-beat agreement -> ABP-target tasks only (segment_bp
            # is meaningless on an ECG/PPG waveform).
            if args.bland_altman and target_idx == 0:
                _bland_altman(segment_bp(pred), segment_bp(gt), out_dir, f"waveform{suffix}",
                              title=f"{arrow}  (waveform truth)")
        except Exception as e:  # noqa: BLE001 — defer until all collectives are done
            deferred_error = deferred_error or e

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    if not is_main:
        return
    if deferred_error is not None:
        raise deferred_error

    # single task -> flat report (back-compat); multiple -> {task: report}
    payload = results[tasks_to_eval[0]] if not multi else results
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    logger.info("metrics (%s) -> %s", "+".join(tasks_to_eval), out_dir / "metrics.json")


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
    (mean diff) and the 95% limits of agreement (bias +/- 1.96 SD). ``name`` is a
    filename suffix (e.g. ``waveform`` or ``waveform_ppg2abp``); ``title`` is the
    figure suptitle (e.g. "ECG+PPG -> ABP  (waveform truth)").
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
    ap.add_argument("--task", default="all",
                    choices=["all", *TASK_ORDER],
                    help="task(s) to evaluate. 'all' (default) = EVERY task the "
                         "checkpoint trained on (->ABP: clinical+waveform; ECG/PPG "
                         "translation: waveform-only); or pin one. Each appears under "
                         "its name in metrics.json when more than one is evaluated.")
    ap.add_argument("--num", type=int, default=-1, help="max segments (-1 = all)")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override sampling.num_steps (ODE steps); default = config value. "
                         "Use for the quality-vs-steps ablation, e.g. --num-steps 4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--use-ema", action="store_true")
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
