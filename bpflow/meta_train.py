"""ANIL/CAVIA meta-training loop.

Reuses a constructed ``Trainer`` for all the heavy machinery (model build, EMA,
optimizer, checkpointing, SwanLab, exp_dir, DDP setup) and only replaces the data
loop + validation with the episodic meta loop and subject-disjoint K-shot eval.

Subjects come from Train_Subset (subject-disjoint train/val split); the honest
final generalization test is on CalFree subjects, which are a different file and
so never seen in meta-training. Validation/test report a K-shot curve
(K = 0,1,3,5,… cuff segments adapted per held-out subject).

DDP: every rank runs the same number of outer steps and per-episode backwards, so
the gradient all-reduce stays synchronized; K-shot eval runs on rank 0 (params +
EMA are identical across ranks) while the others wait at a barrier.
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch

from .meta import kshot_evaluate, meta_train_step
from .meta_data import (
    build_episode_loader,
    load_bp_z,
    shard_subjects,
    split_subjects,
    subject_groups,
)
from .trainer_utils import is_main_process

logger = logging.getLogger(__name__)


def _parse_ks(spec: str) -> List[int]:
    return sorted({int(x) for x in str(spec).split(",") if x.strip() != ""})


def _make_grad_reducer(trainer):
    """Manual all-reduce of raw-model grads once per meta-step (None if single-process).

    The meta loop runs on the un-wrapped model_raw (bp_head is used outside the
    generative forward, so DDP's autograd hooks wouldn't track it); averaging grads
    by hand after the meta-batch backward is simpler and exactly correct.
    """
    if not trainer.distributed:
        return None
    import torch.distributed as dist

    world = trainer.world_size

    def reduce_grads(model_raw) -> None:
        for p in model_raw.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world

    return reduce_grads


def _log_kshot(trainer, reports: Dict[int, dict], prefix: str, step: int) -> None:
    """Log a K-shot report set to SwanLab (per-K flattened) and the terminal."""
    flat: dict = {}
    for k, report in reports.items():
        flat.update(trainer._flat_report(report, f"{prefix}/K{k}"))
    trainer._sw_log(flat, step)
    lines = [f"[{prefix}] K-shot curve @ step {step}"]
    for k in sorted(reports):
        w = reports[k]["waveform"]
        sbp, dbp = reports[k]["SBP"]["AAMI"], reports[k]["DBP"]["AAMI"]
        lines.append(
            f"  K={k:<2d}  MAE {w['MAE']:.3f}  RMSE {w['RMSE']:.3f}  r {w['Pearson']:.3f}"
            f"  | SBP SDE {sbp['SDE']:.2f} AAMI {'P' if sbp['pass'] else 'F'}"
            f"  DBP SDE {dbp['SDE']:.2f} AAMI {'P' if dbp['pass'] else 'F'}"
        )
    logger.info("\n".join(lines))


def _kshot_eval(trainer, arr, bp_z, groups, subjects, ks_list, prefix, step) -> Dict[int, dict]:
    """Rank-0 K-shot eval with the EMA weights; barrier-guarded under DDP."""
    cfg = trainer.cfg
    m = cfg.meta
    trainer._barrier()
    reports: Dict[int, dict] = {}
    if is_main_process():
        was_training = trainer.model_raw.training
        trainer.model_raw.eval()
        with trainer._ema_swapped(trainer.use_ema):
            reports = kshot_evaluate(
                trainer.model_raw, trainer.fm, cfg, arr, bp_z, groups, subjects, ks_list,
                device=trainer.device, generator=trainer.gen,
                max_subjects=int(m.eval_max_subjects), max_query=int(m.eval_max_query),
            )
        if was_training:
            trainer.model_raw.train()
        if reports:
            _log_kshot(trainer, reports, prefix, step)
    trainer._barrier()
    return reports


def meta_train(trainer) -> None:
    cfg = trainer.cfg
    m = cfg.meta
    if not bool(cfg.model.use_context):
        raise ValueError("meta.enabled requires model.use_context: true (the ANIL context)")

    groups = subject_groups(str(cfg.data.train_npy), min_segs=int(m.support_size) + int(m.query_size))
    train_subj, val_subj = split_subjects(
        list(groups.keys()), float(m.val_subject_fraction), int(cfg.data.split_seed)
    )
    rank_subj = shard_subjects(train_subj, trainer.rank, trainer.world_size)
    rank_groups = {s: groups[s] for s in rank_subj}
    bp_z = load_bp_z(str(cfg.data.train_npy), cfg)
    loader = build_episode_loader(cfg, rank_groups, bp_z, seed=int(cfg.training.seed) + trainer.rank)
    episodes_iter = iter(loader)
    arr = np.load(str(cfg.data.train_npy), mmap_mode="r")
    ks_list = _parse_ks(m.eval_ks)
    steps = int(m.meta_steps)
    reduce_grads = _make_grad_reducer(trainer)

    if is_main_process():
        logger.info(
            "Meta-train (%s inner): %d train / %d val subjects | %d steps, Ks=%d Kq=%d, "
            "meta_bs=%d, inner=%d@lr%.3g, context_dim=%d",
            str(m.inner_objective), len(train_subj), len(val_subj), steps,
            int(m.support_size), int(m.query_size), int(m.meta_batch_subjects),
            int(m.k_inner), float(m.inner_lr), int(cfg.model.context_dim),
        )

    for step in range(trainer.global_step, steps):
        episodes_cpu = next(episodes_iter)
        episodes = [tuple(t.to(trainer.device, non_blocking=True) for t in ep) for ep in episodes_cpu]
        ql = meta_train_step(
            trainer.model_raw, trainer.fm, episodes,
            cfg=cfg, generator=trainer.gen, optimizer=trainer.optimizer, reduce_grads=reduce_grads,
        )
        trainer._update_ema()
        trainer.global_step = step + 1

        if is_main_process() and (step % int(m.log_every) == 0 or step == steps - 1):
            logger.info("meta step %d/%d  query_loss %.4f", step, steps, ql)
            trainer._sw_log({"meta/query_loss": ql, "meta/step": step}, step)

        if (step + 1) % int(m.val_every) == 0 or step == steps - 1:
            reports = _kshot_eval(trainer, arr, bp_z, groups, val_subj, ks_list, "val", step)
            if is_main_process():
                # best-by-val = waveform MAE at the largest K (best-adapted)
                if reports:
                    mae = reports[max(reports)]["waveform"]["MAE"]
                    if mae < trainer.best_val:
                        trainer.best_val = mae
                        trainer.save_checkpoint(epoch=0, filename="checkpoint_best.pth")
                trainer.save_checkpoint(epoch=0, filename="checkpoint_latest.pth")

    _meta_test(trainer, ks_list)


def _meta_test(trainer, ks_list: List[int]) -> None:
    """Honest final K-shot test on CalFree subjects (a different file -> unseen)."""
    cfg = trainer.cfg
    groups = subject_groups(str(cfg.data.test_npy), min_segs=max(ks_list) + 1)
    arr = np.load(str(cfg.data.test_npy), mmap_mode="r")
    bp_z = load_bp_z(str(cfg.data.test_npy), cfg)
    reports = _kshot_eval(trainer, arr, bp_z, groups, list(groups.keys()), ks_list, "test", trainer.global_step)
    if is_main_process() and reports:
        out = {str(k): r for k, r in reports.items()}
        path = os.path.join(trainer.exp_dir, "kshot_test_metrics.json")
        with open(path, "w") as f:
            json.dump(_jsonable(out), f, indent=2)
        logger.info("Wrote K-shot test metrics -> %s", path)


def _jsonable(obj):
    """Recursively make a report dict JSON-serializable (numpy/bool/tensor scalars)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj
