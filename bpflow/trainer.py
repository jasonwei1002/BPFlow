"""Device-agnostic BPFlow trainer.

Single-stream conditional flow matching: ECG+PPG -> ABP. Runs on CPU
(smoke), single GPU, or DDP (torchrun, WORLD_SIZE>1). All CUDA use is gated
on availability so a tiny CPU smoke test never hits a hard cuda call.
"""

import contextlib
import functools
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import MODALITY_MASK, MODALITY_ORDER, build_dataset, trained_modalities
from .eval import aami, bhs, format_report, segment_bp
from .eval.metrics import _pearson
from .model import build_model
from .sampling import build_flow_matching, flow_matching_loss, sample_abp
from .trainer_utils import (
    add_weight_decay,
    adjust_learning_rate,
    is_main_process,
    load_model_state,
    pick_device,
    set_seed,
)

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int, base_seed: int = 0) -> None:
    """Seed a DataLoader worker's torch RNG from the run seed so modality_dropout
    draws are tied to cfg.training.seed (reproducible at a fixed num_workers),
    not to incidental prior main-process RNG consumption. No-op for non-dropout
    runs (__getitem__ draws no randomness then)."""
    torch.manual_seed(base_seed + worker_id)


class Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        set_seed(int(cfg.training.seed))

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.distributed = self.world_size > 1
        self.device = pick_device(str(cfg.training.device))
        self.is_cuda = self.device.type == "cuda"

        if self.distributed:
            import torch.distributed as dist

            backend = "nccl" if self.is_cuda else "gloo"
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.is_cuda:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")

        # Run dir: either resume IN PLACE (reuse an existing dir as-is, all ranks
        # derive the same path from the shared config — no broadcast) or start a
        # fresh timestamped run (rank-0 picks it, broadcast under DDP so all ranks
        # agree on exp_dir — _maybe_resume reads it on every rank). SwanLab gets no
        # experiment_name and auto-generates its own run id (resumed below).
        resume_dir = str(cfg.training.resume_dir)
        if resume_dir:
            self.exp_dir = os.path.normpath(resume_dir)
            self.run_name = os.path.basename(self.exp_dir)
        else:
            self.run_name = self._make_run_name()
            self.exp_dir = os.path.join(str(cfg.training.output_dir), self.run_name)
        if is_main_process():
            os.makedirs(self.exp_dir, exist_ok=True)

        self._build_data()
        self._build_model()
        self._build_optimizer()

        self.global_step = 0
        self.start_epoch = 0
        self.best_val = float("inf")
        self.epochs_no_improve = 0  # consecutive validations w/o val improvement
        self.lr_scale = 1.0         # plateau lr multiplier (applied in _train_step)
        self.should_stop = False    # set by early stopping
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(cfg.training.seed) + self.rank)
        self._reset_epoch_loss()  # per-modality epoch accumulators (reset each epoch)
        self._resumed = False
        self.sw_run_id = None       # this run's SwanLab id (saved into checkpoints)
        self._resume_sw_id = None   # prior run's SwanLab id read on resume (to continue it)
        self._maybe_resume()
        self._maybe_init_from_ckpt()

        self.sw = None
        self._init_swanlab()

    # -- setup -------------------------------------------------------------
    def _build_data(self) -> None:
        self.train_ds = build_dataset(self.cfg, "train")
        sampler = None
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                self.train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
        self.sampler = sampler
        num_workers = int(self.cfg.training.num_workers)
        self.loader = DataLoader(
            self.train_ds,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=self.is_cuda,
            persistent_workers=num_workers > 0,
            # tie modality_dropout draws to cfg.seed (per worker, per rank), not to
            # incidental prior RNG state; no-op for non-dropout runs.
            worker_init_fn=(
                functools.partial(
                    _seed_worker, base_seed=int(self.cfg.training.seed) + self.rank * 1000
                )
                if num_workers > 0
                else None
            ),
        )
        # Validation loader (built on EVERY rank; validation is DDP-sharded and
        # gathered). val split is a held-out 20% of Train_Subset (same seed).
        self.val_loader = None
        if int(self.cfg.training.val_freq_epoch) > 0:
            val_ds = build_dataset(self.cfg, "val")
            frac = float(getattr(self.cfg.training, "val_eval_fraction", 1.0))
            if not 0.0 < frac <= 1.0:
                raise ValueError(f"val_eval_fraction must be in (0, 1], got {frac}")
            if frac < 1.0:
                # Stride subsample a representative ~frac of val (covers all
                # subjects uniformly; fixed across epochs → MAE trend comparable).
                # Applied BEFORE the DDP shard so every rank sees the same set.
                step = max(1, int(round(1.0 / frac)))
                val_ds = Subset(val_ds, list(range(0, len(val_ds), step)))
            if self.distributed:
                # strided shard: exact coverage, no padding/duplicates
                val_ds = Subset(val_ds, list(range(self.rank, len(val_ds), self.world_size)))
            self.val_loader = DataLoader(
                val_ds,
                batch_size=int(self.cfg.training.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.training.num_workers),
                drop_last=False,
                pin_memory=self.is_cuda,
            )

    def _build_model(self) -> None:
        if self.is_cuda:
            # TF32 for fp32 matmul/conv paths (zero-risk alongside bf16 training).
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = build_model(self.cfg).to(self.device)
        self.model_raw = model
        if self.distributed and self.is_cuda:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], broadcast_buffers=False
            )
        else:
            self.model = model
        # Compile the training-forward path (CUDA only). Sampling/validation use
        # the eager model_raw, so variable val batch sizes never trigger a
        # recompile; the fixed train batch (drop_last) keeps one compiled graph.
        if self.is_cuda and bool(self.cfg.training.use_compile):
            self.model = torch.compile(self.model)
            if is_main_process():
                logger.info("torch.compile enabled (first step compiles, then speeds up)")
        self.fm = build_flow_matching(self.cfg)
        self.use_ema = bool(self.cfg.training.use_ema)
        self.ema_decay = float(self.cfg.training.ema_decay)
        self.ema_params: Optional[List[torch.Tensor]] = (
            [p.detach().clone() for p in self.model_raw.parameters()] if self.use_ema else None
        )
        if is_main_process():
            logger.info("BPFlowModel '%s': %.3fM params on %s",
                        self.cfg.model.name, self.model_raw.num_parameters() / 1e6, self.device)

    def _build_optimizer(self) -> None:
        groups = add_weight_decay(self.model_raw, float(self.cfg.training.weight_decay))
        self.optimizer = torch.optim.AdamW(
            groups, lr=float(self.cfg.training.lr), betas=(0.9, 0.95)
        )

    def _make_run_name(self) -> str:
        """Timestamp run name; rank 0 picks it and broadcasts so DDP ranks agree."""
        name = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main_process() else ""
        if self.distributed:
            import torch.distributed as dist

            obj = [name]
            dist.broadcast_object_list(obj, src=0)
            name = obj[0]
        return name

    def _init_swanlab(self) -> None:
        """Start a SwanLab run on rank 0 if enabled (lazy import, never fatal)."""
        if not bool(self.cfg.training.use_swanlab) or not is_main_process():
            return
        try:
            import swanlab
        except ImportError:
            logger.warning("use_swanlab=true but swanlab is not installed; skipping (pip install swanlab).")
            return
        # No experiment_name -> SwanLab auto-generates its run id. Record run_name
        # (the checkpoint dir) in config so the run links back to its checkpoints.
        config = OmegaConf.to_container(self.cfg, resolve=True)
        assert isinstance(config, dict)
        config["run_name"] = self.run_name
        mode = str(self.cfg.training.swanlab_mode)
        # Resume the same run only when we have a prior id AND the run is cloud
        # (id/resume are online-only in SwanLab; 'cloud' is the legacy alias).
        # resume="allow" => continue if it still exists, else start a fresh run.
        resume_kwargs: dict = {}
        if self._resume_sw_id and mode in ("online", "cloud"):
            resume_kwargs = {"id": str(self._resume_sw_id), "resume": "allow"}
        run = swanlab.init(
            project=str(self.cfg.training.swanlab_project),
            description="BPFlow ECG+PPG -> ABP flow matching",
            config=config,
            mode=mode,
            **resume_kwargs,
        )
        self.sw = swanlab
        # Remember this run's id so checkpoints can point --resume back to it.
        self.sw_run_id = getattr(run, "id", None)
        logger.info(
            "SwanLab enabled (project=%s, ckpt_dir=%s, mode=%s, run_id=%s%s)",
            self.cfg.training.swanlab_project, self.exp_dir, mode,
            self.sw_run_id, " [resumed]" if resume_kwargs else "",
        )

    def _sw_log(self, data: dict, step: int) -> None:
        if self.sw is not None:
            self.sw.log(data, step=step)

    @staticmethod
    def _flat_report(report: dict, prefix: str = "val") -> dict:
        """Flatten the eval report into scalar metrics for logging (val/ or test/).

        Top-level SBP/DBP/MAP → ``{prefix}/{key}_*`` (waveform-truth). A ``cuff``
        block, when present (post-train test with the CSV label), → a parallel
        ``{prefix}/cuff_{key}_*`` so both truth sources are logged separately.
        """
        out = {f"{prefix}/{k}": float(v) for k, v in report["waveform"].items()}
        sources = [("", report)]
        if "cuff" in report:
            sources.append(("cuff_", report["cuff"]))
        for infix, rep in sources:
            for key in ("SBP", "DBP", "MAP"):
                a, b = rep[key]["AAMI"], rep[key]["BHS"]
                out[f"{prefix}/{infix}{key}_ME"] = float(a["ME"])
                out[f"{prefix}/{infix}{key}_SDE"] = float(a["SDE"])
                out[f"{prefix}/{infix}{key}_AAMI_pass"] = float(a["pass"])
                out[f"{prefix}/{infix}{key}_within5mmHg"] = float(b["<=5mmHg"])
        return out

    def _autocast(self):
        if not self.is_cuda or str(self.cfg.training.amp_dtype) == "float32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if str(self.cfg.training.amp_dtype) == "bfloat16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)

    # -- training ----------------------------------------------------------
    def _prepare_batch(self, batch):
        abp = batch["abp_patches"].to(self.device, non_blocking=True)
        cond = batch["cond_patches"].to(self.device, non_blocking=True)
        cond_mask = batch["cond_mask"].to(self.device, non_blocking=True)
        rf = int(self.cfg.training.repeat_factor)
        if rf > 1:
            abp = abp.repeat(rf, 1, 1)
            cond = cond.repeat(rf, 1, 1)
            cond_mask = cond_mask.repeat(rf, 1)
        # Demographics ride along as a global prior (not a CFG-dropped condition).
        demo = None
        if bool(self.cfg.model.use_demo) and "demo_cont" in batch:
            cont = batch["demo_cont"].to(self.device, non_blocking=True)
            gender = batch["demo_gender"].to(self.device, non_blocking=True)
            if rf > 1:
                cont = cont.repeat(rf, 1)
                gender = gender.repeat(rf)
            demo = (cont, gender)
        return abp, cond, cond_mask, demo

    def _train_step(self, batch) -> dict:
        lr = adjust_learning_rate(self.optimizer, self.global_step, self.cfg, self.lr_scale)
        with self._autocast():
            abp, cond, cond_mask, demo = self._prepare_batch(batch)
            loss_vec = flow_matching_loss(
                self.model, self.fm, abp, cond,
                generator=self.gen,
                logit_mean=float(self.cfg.sampling.logit_mean),
                logit_scale=float(self.cfg.sampling.logit_scale),
                prediction_type=str(self.cfg.sampling.prediction_type),
                loss_type=str(self.cfg.training.loss_type),
                demo=demo,
                cond_mask=cond_mask,
                per_sample=True,  # (B,) so we can decompose train loss by modality
            )
            loss = loss_vec.mean()

        # Full-epoch per-modality train loss, decomposed from this same forward.
        self._accumulate_modality_loss(loss_vec, cond_mask)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.model_raw.parameters(), float(self.cfg.training.clip_grad_norm)
        )
        self.optimizer.step()
        self._update_ema()
        return {"loss": float(loss.detach()), "lr": lr, "grad_norm": float(gnorm), "bs": abp.shape[0]}

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_params is None:
            return
        for ema, p in zip(self.ema_params, self.model_raw.parameters()):
            ema.lerp_(p.detach(), 1.0 - self.ema_decay)

    # Full-epoch train loss broken down by modality, decomposed FROM the actual
    # training loss (no extra forward): each sample's per-sample loss is bucketed
    # by its live modality_dropout mask (matched against the shared MODALITY_MASK
    # table, in MODALITY_ORDER), accumulated over the epoch, all-reduced across
    # ranks, and logged as train/loss_<modality> + train/loss_epoch.

    def _all_reduce_sum(self, *tensors: torch.Tensor) -> None:
        """In-place SUM all_reduce across ranks (no-op when not distributed). The
        caller must invoke this on ALL ranks together — it is a collective."""
        if not self.distributed:
            return
        import torch.distributed as dist

        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def _reset_epoch_loss(self) -> None:
        n = len(MODALITY_ORDER)
        self._ep_loss_sum = torch.zeros(n, device=self.device)  # order = MODALITY_ORDER
        self._ep_loss_cnt = torch.zeros(n, device=self.device)

    @torch.no_grad()
    def _accumulate_modality_loss(self, loss_vec: torch.Tensor, cond_mask: torch.Tensor) -> None:
        """Bucket per-sample train losses by the sample's presence mask, keyed by
        the shared MODALITY_MASK table so bucket i always means MODALITY_ORDER[i]."""
        lv = loss_vec.detach().float()
        for i, name in enumerate(MODALITY_ORDER):
            me, mp = MODALITY_MASK[name]
            m = (cond_mask[:, 0] == me) & (cond_mask[:, 1] == mp)
            if m.any():
                self._ep_loss_sum[i] += lv[m].sum()
                self._ep_loss_cnt[i] += m.sum()

    def _log_epoch_loss(self, epoch: int) -> None:
        """Log the full-epoch (all-rank) per-modality train loss. ALL ranks must
        call this — the all_reduce is a collective — then only rank 0 logs."""
        s, c = self._ep_loss_sum.clone(), self._ep_loss_cnt.clone()
        self._all_reduce_sum(s, c)
        if not is_main_process():
            return
        data: dict = {}
        for i, name in enumerate(MODALITY_ORDER):
            if c[i] > 0:
                data[f"train/loss_{name}"] = float(s[i] / c[i])
        tot_c = float(c.sum())
        if tot_c > 0:
            data["train/loss_epoch"] = float(s.sum() / tot_c)
        if data:
            self._sw_log(data, self.global_step)
            logger.info("[train] epoch %d full-set loss  %s", epoch,
                        "  ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in data.items()))

    def train(self) -> None:
        max_steps = int(self.cfg.training.max_steps)
        log_freq = int(self.cfg.training.log_freq)
        ckpt_freq = int(self.cfg.training.ckpt_freq_epoch)
        val_freq = int(self.cfg.training.val_freq_epoch)
        n_epochs = int(self.cfg.training.epochs)
        if is_main_process():
            logger.info("Start training: %d epochs, max_steps=%d", n_epochs, max_steps)
        start = time.time()
        # Checkpoint "epoch" field = the next epoch index to run on resume, so
        # range(start_epoch, n_epochs) is exact. A completed epoch E saves E+1;
        # a mid-epoch max_steps stop saves E (resume re-runs the partial epoch).
        for epoch in range(self.start_epoch, n_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self.model.train()
            self._reset_epoch_loss()  # accumulate per-modality train loss over the epoch
            # Per-epoch progress bar (rank 0 only; a no-op on other ranks).
            pbar = tqdm(
                self.loader,
                desc=f"epoch {epoch}/{n_epochs}",
                disable=not is_main_process(),
                dynamic_ncols=True,
                leave=False,
            )
            for batch in pbar:
                metrics = self._train_step(batch)
                self.global_step += 1
                if is_main_process():
                    pbar.set_postfix_str(
                        f"loss={metrics['loss']:.4f} lr={metrics['lr']:.2e} gnorm={metrics['grad_norm']:.2f}"
                    )
                if is_main_process() and self.global_step % log_freq == 0:
                    self._sw_log(
                        {
                            "train/loss": metrics["loss"],
                            "train/lr": metrics["lr"],
                            "train/grad_norm": metrics["grad_norm"],
                            "train/epoch": epoch,
                        },
                        self.global_step,
                    )
                if 0 < max_steps <= self.global_step:
                    pbar.close()
                    self._log_epoch_loss(epoch)  # flush partial-epoch loss (all ranks: collective)
                    if is_main_process():
                        logger.info("Reached max_steps=%d, stopping.", max_steps)
                        self.save_checkpoint(epoch, "checkpoint_latest.pth")
                    self._barrier()
                    return
            pbar.close()
            self._log_epoch_loss(epoch)  # all-rank full-epoch per-modality train loss
            done = epoch + 1
            if val_freq > 0 and done % val_freq == 0:
                self._run_validation(done)
                if self.should_stop:
                    if is_main_process():
                        self.save_checkpoint(done, "checkpoint_latest.pth")
                        logger.info("Early stopped at epoch %d (best val MAE %.4f).", done, self.best_val)
                    self._barrier()
                    self._maybe_run_test(done)
                    return
            if is_main_process() and ckpt_freq > 0 and done % ckpt_freq == 0:
                self.save_checkpoint(done, "checkpoint_latest.pth")
            self._barrier()
        # Final validation — only if the last epoch wasn't already validated in-loop
        # (n_epochs % val_freq == 0), else it validates twice at the same epoch →
        # spurious epochs_no_improve increment + a duplicate val/* log at step=n_epochs.
        if val_freq > 0 and n_epochs % val_freq != 0:
            self._run_validation(n_epochs)
        if is_main_process():
            logger.info("Training done in %.1fs", time.time() - start)
            self.save_checkpoint(n_epochs, "checkpoint_latest.pth")
        self._maybe_run_test(n_epochs)

    def _run_validation(self, done_epochs: int) -> None:
        # validate() returns the real MAE on rank 0 and inf elsewhere, so all
        # plateau/early-stop decisions are made on rank 0 then broadcast. val/*
        # is logged to SwanLab with step=done_epochs (epoch x-axis).
        val_mae = self.validate(done_epochs)
        if is_main_process():
            if val_mae < self.best_val:
                self.best_val = val_mae
                self.epochs_no_improve = 0
                self.save_checkpoint(done_epochs, "checkpoint_best.pth")
                best_path = os.path.abspath(os.path.join(self.exp_dir, "checkpoint_best.pth"))
                logger.info("New best val MAE %.4f mmHg @ epoch %d -> %s", val_mae, done_epochs, best_path)
            else:
                self.epochs_no_improve += 1
                logger.info("val MAE %.4f: no improvement for %d val round(s) (best %.4f)",
                            val_mae, self.epochs_no_improve, self.best_val)
                lr_pat = int(self.cfg.training.lr_patience)
                if lr_pat > 0 and self.epochs_no_improve % lr_pat == 0:
                    self.lr_scale *= float(self.cfg.training.lr_decay)
                    logger.info("LR plateau (%d rounds): lr_scale -> %.4g (base_lr x this)",
                                self.epochs_no_improve, self.lr_scale)
                es_pat = int(self.cfg.training.early_stop_patience)
                if es_pat > 0 and self.epochs_no_improve >= es_pat:
                    self.should_stop = True
                    logger.info("Early stop: no val improvement for %d val round(s).", self.epochs_no_improve)
        self._sync_val_state()
        self._barrier()

    def _sync_val_state(self) -> None:
        """Broadcast rank-0 plateau/early-stop state to all ranks (DDP)."""
        if not self.distributed:
            return
        import torch.distributed as dist

        obj = [self.best_val, self.epochs_no_improve, self.lr_scale, self.should_stop]
        dist.broadcast_object_list(obj, src=0)
        self.best_val, self.epochs_no_improve, self.lr_scale, self.should_stop = obj

    # -- sampling ----------------------------------------------------------
    def _batch_demo(self, batch):
        """Pull a (cont, gender) demographics sample from a batch, or None.

        Left on CPU; ``sample_abp`` moves it to device. No repeat/drop here —
        sampling uses the natural batch.
        """
        if not bool(self.cfg.model.use_demo) or "demo_cont" not in batch:
            return None
        return batch["demo_cont"], batch["demo_gender"]

    @torch.no_grad()
    def _sample_cond(self, cond_patches: torch.Tensor, demo=None, cond_mask=None) -> torch.Tensor:
        """Core sampler; assumes model is already in the desired eval/param state."""
        return sample_abp(
            self.model_raw, self.fm, cond_patches,
            generator=self.gen, device=self.device,
            abp_mean=float(self.cfg.data.abp_mean), abp_std=float(self.cfg.data.abp_std),
            autocast_ctx=self._autocast(),
            demo=demo,
            cond_mask=cond_mask,
        )

    @contextlib.contextmanager
    def _ema_swapped(self, use_ema: bool):
        """Temporarily load EMA params into the live model, then restore.

        Param swaps run under ``no_grad`` so the in-place copy_ on leaves that
        require grad is always safe, regardless of the caller's autograd state.
        """
        if use_ema and self.ema_params is not None:
            with torch.no_grad():
                backup = [p.detach().clone() for p in self.model_raw.parameters()]
                for p, e in zip(self.model_raw.parameters(), self.ema_params):
                    p.copy_(e)
            try:
                yield
            finally:
                with torch.no_grad():
                    for p, b in zip(self.model_raw.parameters(), backup):
                        p.copy_(b)
        else:
            yield

    @torch.no_grad()
    def _distributed_eval(self, loader, desc: str, max_batches: int = -1):
        """Sample this rank's loader shard, gather PER-SEGMENT quantities to rank 0,
        and assemble a full report there (None on other ranks).

        Memory-light: gathers each segment's BP values + waveform error summaries
        (a handful of scalars), never the waveforms, so full-set validation scales
        to many GPUs. ``max_batches>0`` caps batches PER RANK (<=0 = full set).
        Assumes model_raw is already in the desired param state (the caller does
        the EMA swap / best-weights load). ALL ranks must call this together — the
        all_gather is a collective; an early per-rank return would deadlock.
        """
        was_training = self.model_raw.training
        self.model_raw.eval()
        # `*_t` = waveform-truth BP (per-beat on the true wave); `*_c` = CSV cuff
        # label, gathered only when the dataset loaded bp_true (eval_true_source
        # csv/both) → enables a second, parallel clinical truth source.
        keys = ("sbp_p", "dbp_p", "map_p", "sbp_t", "dbp_t", "map_t", "ae", "se", "r",
                "sbp_c", "dbp_c", "map_c")
        cols: dict = {k: [] for k in keys}
        for bi, batch in enumerate(tqdm(loader, desc=desc, leave=False, disable=not is_main_process())):
            if 0 < max_batches <= bi:
                break
            pred = self._sample_cond(
                batch["cond_patches"], self._batch_demo(batch), batch.get("cond_mask")
            )  # (b, L) mmHg on CPU
            gt = batch["abp_raw"]
            pbp, tbp = segment_bp(pred), segment_bp(gt)
            err = pred - gt
            cols["sbp_p"].append(pbp["SBP"]); cols["dbp_p"].append(pbp["DBP"]); cols["map_p"].append(pbp["MAP"])
            cols["sbp_t"].append(tbp["SBP"]); cols["dbp_t"].append(tbp["DBP"]); cols["map_t"].append(tbp["MAP"])
            cols["ae"].append(err.abs().mean(dim=1))   # (b,) per-segment MAE
            cols["se"].append((err ** 2).mean(dim=1))  # (b,) per-segment MSE
            cols["r"].append(_pearson(pred, gt))        # (b,) per-segment Pearson
            cuff = batch.get("bp_true")  # (b, 3) [SBP, DBP, MAP] cuff label, or None
            if cuff is not None:
                cols["sbp_c"].append(cuff[:, 0]); cols["dbp_c"].append(cuff[:, 1]); cols["map_c"].append(cuff[:, 2])
        if was_training:
            self.model_raw.train()
        # Drop empty cuff cols when no bp_true was present → gathered dict keys stay
        # consistent across ranks (all ranks share the same config).
        local = {k: torch.cat(v) for k, v in cols.items() if v} if cols["ae"] else None

        if self.distributed:
            import torch.distributed as dist

            gathered: list = [None] * self.world_size
            dist.all_gather_object(gathered, local)
            self._barrier()
            if not is_main_process():
                return None
            parts = [g for g in gathered if g is not None]
            if not parts:
                return None
            merged = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        else:
            if local is None:
                return None
            merged = local
        return self._assemble_report(merged)

    @staticmethod
    def _assemble_report(m: dict) -> dict:
        """Standard eval report from gathered per-segment quantities.

        MAE/RMSE aggregate exactly because every segment has equal length L (so a
        mean-of-per-segment-means equals the global mean; same for MSE). Pearson is
        the per-segment-r mean, matching ``waveform_metrics``. Top-level SBP/DBP/MAP
        are the waveform-truth clinical metrics (PRED per-beat vs the true wave's
        per-beat). When the CSV cuff label was gathered (``*_c`` present), a parallel
        ``cuff`` block reports the same clinical metrics against the cuff label.
        """
        report = {"waveform": {
            "MAE": m["ae"].mean().item(),
            "RMSE": m["se"].mean().sqrt().item(),
            "Pearson": m["r"].mean().item(),
        }}
        pred_bp = {"SBP": m["sbp_p"], "DBP": m["dbp_p"], "MAP": m["map_p"]}
        true_bp = {"SBP": m["sbp_t"], "DBP": m["dbp_t"], "MAP": m["map_t"]}
        for key in ("SBP", "DBP", "MAP"):
            report[key] = {"AAMI": aami(pred_bp[key], true_bp[key]), "BHS": bhs(pred_bp[key], true_bp[key])}
        if "sbp_c" in m:
            cuff_bp = {"SBP": m["sbp_c"], "DBP": m["dbp_c"], "MAP": m["map_c"]}
            report["cuff"] = {
                key: {"AAMI": aami(pred_bp[key], cuff_bp[key]), "BHS": bhs(pred_bp[key], cuff_bp[key])}
                for key in ("SBP", "DBP", "MAP")
            }
        return report

    @torch.no_grad()
    def _val_modality_losses(self, max_batches: int) -> dict:
        """Per-modality flow-matching loss on the val split (ecg_ppg / ecg / ppg).

        Monitor-only — NOT used for best/plateau/early-stop (val MAE drives those);
        handy to watch how a unified (modality_dropout) model fits each direction.
        Per batch the SAME noise & timesteps (fixed per-batch seed) are reused
        across the three modalities, so only cond_mask differs → a paired,
        epoch-comparable signal (the val loader is unshuffled). Comparable WITHIN a
        run; the absolute value depends on world_size (max_batches is per-rank).
        DDP: each rank sums its shard into a device
        tensor, then ONE all_reduce; every rank must call this together (the
        all_reduce is a collective), so it runs before validate()'s rank-0 early
        return. Uses eager model_raw under the caller's EMA swap.
        """
        if self.val_loader is None or max_batches == 0:  # 0 = off; <0 = full; >0 = cap
            return {}
        # only the directions the model trains on (specialist -> its 1 direction;
        # dropout -> every modality with prob>0), so a specialist doesn't pay 3x
        # for two uninformative directions. Same on every rank → DDP-safe size.
        trained = trained_modalities(
            str(self.cfg.data.cond_modality),
            bool(self.cfg.data.modality_dropout),
            self.cfg.data.modality_dropout_probs,
        )
        active = [m for m in MODALITY_ORDER if m in trained]
        base_seed = int(self.cfg.training.seed)
        use_demo = bool(self.cfg.model.use_demo)
        was_training = self.model_raw.training
        self.model_raw.eval()
        gen = torch.Generator(device=self.device)
        # one (1,2) mask per active modality, built once; _apply_null needs it
        # broadcast to the batch, so .repeat(bs,1) below (a contiguous view).
        masks = [torch.tensor([MODALITY_MASK[m]], device=self.device) for m in active]
        sums = torch.zeros(len(active), device=self.device)  # loss*bs, no per-batch sync
        n = 0
        for bi, batch in enumerate(self.val_loader):
            if 0 < max_batches <= bi:
                break
            abp = batch["abp_patches"].to(self.device, non_blocking=True)
            cond = batch["cond_patches"].to(self.device, non_blocking=True)
            bs = abp.shape[0]
            demo = None
            if use_demo and "demo_cont" in batch:
                demo = (batch["demo_cont"].to(self.device), batch["demo_gender"].to(self.device))
            with self._autocast():
                for i, mask in enumerate(masks):
                    gen.manual_seed(base_seed + bi)  # same noise/t across modalities & epochs
                    loss = flow_matching_loss(
                        self.model_raw, self.fm, abp, cond,
                        generator=gen,
                        logit_mean=float(self.cfg.sampling.logit_mean),
                        logit_scale=float(self.cfg.sampling.logit_scale),
                        prediction_type=str(self.cfg.sampling.prediction_type),
                        loss_type=str(self.cfg.training.loss_type),
                        demo=demo, cond_mask=mask.repeat(bs, 1),
                    )
                    sums[i] += loss.detach().float() * bs  # fp32 like the train path
            n += bs
        if was_training:
            self.model_raw.train()
        t = torch.cat([sums.new_tensor([float(n)]), sums])  # [n, sum_ecg_ppg, sum_ecg, sum_ppg]
        self._all_reduce_sum(t)
        vals = t.tolist()  # single device→host transfer
        n = max(int(vals[0]), 1)
        return {name: vals[i + 1] / n for i, name in enumerate(active)}

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """DDP-sharded validation; mean waveform MAE (mmHg) on rank 0 (inf
        elsewhere). Uses the current in-memory EMA weights. Evaluates the whole
        val_loader, which is already stride-subsampled to ``val_eval_fraction``.

        ``val/*`` metrics are logged to SwanLab with ``step=epoch`` (an epoch
        x-axis, not the global train step) since validation is once-per-epoch."""
        if self.val_loader is None:
            return float("inf")
        with self._ema_swapped(self.use_ema):
            report = self._distributed_eval(self.val_loader, "val", -1)
            # all ranks must enter (all_reduce inside) — before the rank-0 return
            mod_losses = self._val_modality_losses(int(self.cfg.training.val_loss_max_batches))
        if report is None or not is_main_process():
            return float("inf")
        logger.info("[val] %s", format_report(report).replace("\n", " | "))
        metrics = self._flat_report(report, "val")
        for k, v in mod_losses.items():
            metrics[f"val/loss_{k}"] = v
        self._sw_log(metrics, epoch)
        if mod_losses:
            logger.info("[val] flow-matching loss  %s",
                        "  ".join(f"{k}={v:.4f}" for k, v in mod_losses.items()))
        return float(report["waveform"]["MAE"])

    # -- final test --------------------------------------------------------
    def _maybe_run_test(self, epoch: int) -> None:
        """After training, optionally evaluate on the CalFree test set.

        Gated by ``training.run_test_after_train``. Runs on ALL ranks (DDP-sharded
        + gathered), logs ``test/*`` to SwanLab (at ``step=epoch``, the same epoch
        axis as ``val/*``) and writes ``test_metrics.json``. Wrapped so a post-train
        eval failure never crashes an otherwise-finished run.
        """
        if not bool(self.cfg.training.run_test_after_train):
            return
        self._barrier()  # ensure rank 0 has flushed checkpoint_best.pth to disk
        try:
            self.run_test(epoch)
        except Exception as e:
            if is_main_process():
                logger.error("Post-train test evaluation failed: %s", e)

    @torch.no_grad()
    def run_test(self, epoch: int) -> None:
        """Evaluate the best-by-val (EMA) model on CalFree test; log + save metrics.

        Forces BOTH clinical truth sources: PRED per-beat is scored against the
        true-wave per-beat (test/*) AND, when the sibling CSV is present, the CSV
        cuff label (test/cuff_*) — logged separately. Falls back to waveform-truth
        only if the CSV is genuinely absent.
        """
        # Override eval_true_source to "both" so the dataset loads the cuff label;
        # _distributed_eval then auto-reports the parallel cuff clinical block.
        cfg_test = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        cfg_test.data.eval_true_source = "both"
        try:
            test_ds = build_dataset(cfg_test, "test")
        except FileNotFoundError:
            logger.warning("CSV cuff label not found; post-train test runs waveform-truth only")
            test_ds = build_dataset(self.cfg, "test")
        total = len(test_ds)
        max_seg = int(self.cfg.training.test_max_segments)
        if 0 < max_seg < total:
            test_ds = Subset(test_ds, list(range(max_seg)))
            total = max_seg
        shard = test_ds
        if self.distributed:
            # Strided shard: exact coverage, no padding. Metrics are set-level.
            shard = Subset(test_ds, list(range(self.rank, total, self.world_size)))
        loader = DataLoader(
            shard, batch_size=int(self.cfg.training.batch_size), shuffle=False,
            num_workers=int(self.cfg.training.num_workers), pin_memory=self.is_cuda,
        )
        src = self._load_eval_weights(os.path.join(self.exp_dir, "checkpoint_best.pth"))
        report = self._distributed_eval(loader, f"test ({total})", -1)
        if report is None or not is_main_process():
            return
        logger.info("[test] %d segments (weights=%s) — clinical vs true wave\n%s",
                    total, src, format_report(report))
        if "cuff" in report:
            logger.info("[test] clinical vs CSV cuff\n%s",
                        format_report({"waveform": report["waveform"], **report["cuff"]}))
        self._sw_log(self._flat_report(report, "test"), epoch)  # epoch axis, like val/*
        path = os.path.join(self.exp_dir, "test_metrics.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("test metrics -> %s", path)

    def _load_eval_weights(self, best_path: str) -> str:
        """Load best-checkpoint EMA (or model) weights into model_raw for testing.

        Returns a short tag of what was loaded. Falls back to the in-memory EMA /
        last weights when no best checkpoint exists (e.g. validation disabled).
        Mutates model_raw in place — fine, since training is already over.
        """
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            if self.use_ema and "model_ema" in ckpt:
                ema = ckpt["model_ema"]
                for n, p in self.model_raw.named_parameters():
                    if n in ema:  # new params absent in older ckpts keep their init
                        p.data.copy_(ema[n].to(self.device))
                return "best/ema"
            load_model_state(self.model_raw, ckpt["model"])
            return "best/model"
        if self.use_ema and self.ema_params is not None:
            for p, e in zip(self.model_raw.parameters(), self.ema_params):
                p.data.copy_(e)
            return "current/ema"
        return "current/last"

    # -- checkpoint --------------------------------------------------------
    def save_checkpoint(self, epoch: int, filename: str) -> None:
        if not is_main_process():
            return
        ckpt = {
            "model": self.model_raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "epochs_no_improve": self.epochs_no_improve,
            "lr_scale": self.lr_scale,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "abp_mean": float(self.cfg.data.abp_mean),
            "abp_std": float(self.cfg.data.abp_std),
            "swanlab_run_id": self.sw_run_id,  # so --resume continues this run
        }
        if self.ema_params is not None:
            names = [n for n, _ in self.model_raw.named_parameters()]
            ckpt["model_ema"] = {n: e.detach().cpu() for n, e in zip(names, self.ema_params)}
        path = os.path.join(self.exp_dir, filename)
        torch.save(ckpt, path)
        logger.info("Saved checkpoint -> %s (epoch %d, step %d)", path, epoch, self.global_step)

    def _ema_from_ckpt(self, ema: dict) -> list:
        """EMA tensors aligned to current params; a param absent in an older
        checkpoint (e.g. the null tokens) falls back to its current init value."""
        return [
            (ema[n] if n in ema else p.detach().clone()).to(self.device)
            for n, p in self.model_raw.named_parameters()
        ]

    def _maybe_resume(self) -> None:
        path = os.path.join(self.exp_dir, "checkpoint_latest.pth")
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        load_model_state(self.model_raw, ckpt["model"])
        if self.ema_params is not None and "model_ema" in ckpt:
            self.ema_params = self._ema_from_ckpt(ckpt["model_ema"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val = float(ckpt.get("best_val", float("inf")))
        self.epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        self.lr_scale = float(ckpt.get("lr_scale", 1.0))
        # Continue the same SwanLab run if the checkpoint recorded one (older
        # checkpoints lack the key -> None -> a fresh run is started instead).
        self._resume_sw_id = ckpt.get("swanlab_run_id")
        self._resumed = True
        if is_main_process():
            logger.info("Resumed from %s (epoch %d, step %d, lr_scale %.4g, swanlab=%s)",
                        path, self.start_epoch, self.global_step, self.lr_scale,
                        self._resume_sw_id or "fresh")

    def _maybe_init_from_ckpt(self) -> None:
        """Initialize model (+ EMA) weights from a pretrained checkpoint.

        For the finetune flow: load only the weights (architecture must match),
        leaving optimizer/epoch/step/best_val fresh so training restarts from 0.
        Skipped when resuming an interrupted run (resume already loaded weights)
        or when ``init_from_ckpt`` is empty.
        """
        path = str(self.cfg.training.init_from_ckpt)
        if not path or self._resumed:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"init_from_ckpt not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        load_model_state(self.model_raw, ckpt["model"])
        if self.ema_params is not None and "model_ema" in ckpt:
            self.ema_params = self._ema_from_ckpt(ckpt["model_ema"])
        if is_main_process():
            has_ema = "model_ema" in ckpt
            logger.info("Initialized weights from %s (ema=%s); optimizer/epoch/step reset",
                        path, has_ema)

    def _barrier(self) -> None:
        if self.distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()

    def close(self) -> None:
        if self.sw is not None:
            self.sw.finish()
        if self.distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
