"""Device-agnostic BPFlow trainer.

Single-stream conditional flow matching: ECG+PPG -> ABP. Runs on CPU
(smoke), single GPU, or DDP (torchrun, WORLD_SIZE>1). All CUDA use is gated
on availability so a tiny CPU smoke test never hits a hard cuda call.
"""

import contextlib
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

from .data import build_dataset
from .eval import aami, bhs, format_report, segment_bp
from .eval.metrics import _pearson
from .model import build_model
from .sampling import build_flow_matching, flow_matching_loss, sample_abp
from .trainer_utils import (
    add_weight_decay,
    adjust_learning_rate,
    is_main_process,
    pick_device,
    set_seed,
)

logger = logging.getLogger(__name__)


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

        # Run is named by a timestamp (rank-0 picks it, broadcast under DDP so all
        # ranks agree on exp_dir — _maybe_resume reads it on every rank). SwanLab
        # gets no experiment_name and auto-generates its own run id.
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
        self._resumed = False
        self._maybe_resume()
        self._maybe_init_from_ckpt()

        self.sw = None
        self._init_swanlab()

    # -- setup -------------------------------------------------------------
    def _build_data(self) -> None:
        # Meta-training builds its own per-subject episode loader (meta_data); the
        # standard segment loader + val loader are unused, so skip them entirely.
        if bool(self.cfg.meta.enabled):
            self.train_ds = None
            self.loader = None
            self.sampler = None
            self.val_loader = None
            return
        self.train_ds = build_dataset(self.cfg, "train")
        sampler = None
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                self.train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
        self.sampler = sampler
        self.loader = DataLoader(
            self.train_ds,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(self.cfg.training.num_workers),
            drop_last=True,
            pin_memory=self.is_cuda,
            persistent_workers=int(self.cfg.training.num_workers) > 0,
        )
        # Validation loader (built on EVERY rank; validation is DDP-sharded and
        # gathered). val split is a held-out 20% of Train_Subset (same seed).
        self.val_loader = None
        if int(self.cfg.training.val_freq_epoch) > 0:
            val_ds = build_dataset(self.cfg, "val")
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
        swanlab.init(
            project=str(self.cfg.training.swanlab_project),
            description="BPFlow ECG+PPG -> ABP flow matching",
            config=config,
            mode=str(self.cfg.training.swanlab_mode),
        )
        self.sw = swanlab
        logger.info(
            "SwanLab enabled (project=%s, ckpt_dir=%s, mode=%s)",
            self.cfg.training.swanlab_project, self.exp_dir, self.cfg.training.swanlab_mode,
        )

    def _sw_log(self, data: dict, step: int) -> None:
        if self.sw is not None:
            self.sw.log(data, step=step)

    @staticmethod
    def _flat_report(report: dict, prefix: str = "val") -> dict:
        """Flatten the eval report into scalar metrics for logging (val/ or test/)."""
        out = {f"{prefix}/{k}": float(v) for k, v in report["waveform"].items()}
        for key in ("SBP", "DBP", "MAP"):
            a, b = report[key]["AAMI"], report[key]["BHS"]
            out[f"{prefix}/{key}_ME"] = float(a["ME"])
            out[f"{prefix}/{key}_SDE"] = float(a["SDE"])
            out[f"{prefix}/{key}_AAMI_pass"] = float(a["pass"])
            out[f"{prefix}/{key}_within5mmHg"] = float(b["<=5mmHg"])
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
        rf = int(self.cfg.training.repeat_factor)
        if rf > 1:
            abp = abp.repeat(rf, 1, 1)
            cond = cond.repeat(rf, 1, 1)
        # Demographics ride along as a global prior (not a CFG-dropped condition).
        demo = None
        if bool(self.cfg.model.use_demo) and "demo_cont" in batch:
            cont = batch["demo_cont"].to(self.device, non_blocking=True)
            gender = batch["demo_gender"].to(self.device, non_blocking=True)
            if rf > 1:
                cont = cont.repeat(rf, 1)
                gender = gender.repeat(rf)
            demo = (cont, gender)
        # optional classifier-free training: drop the ECG/PPG condition to learned
        # null (the demo prior is intentionally left untouched).
        p = float(self.cfg.training.label_drop_prob)
        if p > 0.0:
            mask = (torch.rand(cond.shape[0], device=self.device, generator=self.gen) < p)
            null = torch.cat(
                [
                    self.model_raw.empty_ecg.expand(cond.shape[0], cond.shape[1], -1),
                    self.model_raw.empty_ppg.expand(cond.shape[0], cond.shape[1], -1),
                ],
                dim=-1,
            )
            cond = torch.where(mask[:, None, None], null, cond)
        return abp, cond, demo

    def _train_step(self, batch) -> dict:
        lr = adjust_learning_rate(self.optimizer, self.global_step, self.cfg, self.lr_scale)
        with self._autocast():
            abp, cond, demo = self._prepare_batch(batch)
            loss = flow_matching_loss(
                self.model, self.fm, abp, cond,
                generator=self.gen,
                logit_mean=float(self.cfg.sampling.logit_mean),
                logit_scale=float(self.cfg.sampling.logit_scale),
                prediction_type=str(self.cfg.sampling.prediction_type),
                loss_type=str(self.cfg.training.loss_type),
                demo=demo,
            )

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
                    if is_main_process():
                        logger.info("Reached max_steps=%d, stopping.", max_steps)
                        self.save_checkpoint(epoch, "checkpoint_latest.pth")
                    self._barrier()
                    return
            pbar.close()
            done = epoch + 1
            if val_freq > 0 and done % val_freq == 0:
                self._run_validation(done)
                if self.should_stop:
                    if is_main_process():
                        self.save_checkpoint(done, "checkpoint_latest.pth")
                        logger.info("Early stopped at epoch %d (best val MAE %.4f).", done, self.best_val)
                    self._barrier()
                    self._maybe_run_test()
                    return
            if is_main_process() and ckpt_freq > 0 and done % ckpt_freq == 0:
                self.save_checkpoint(done, "checkpoint_latest.pth")
            self._barrier()
        if val_freq > 0:
            self._run_validation(n_epochs)
        if is_main_process():
            logger.info("Training done in %.1fs", time.time() - start)
            self.save_checkpoint(n_epochs, "checkpoint_latest.pth")
        self._maybe_run_test()

    def _run_validation(self, done_epochs: int) -> None:
        # validate() returns the real MAE on rank 0 and inf elsewhere, so all
        # plateau/early-stop decisions are made on rank 0 then broadcast.
        val_mae = self.validate()
        if is_main_process():
            if val_mae < self.best_val:
                self.best_val = val_mae
                self.epochs_no_improve = 0
                self.save_checkpoint(done_epochs, "checkpoint_best.pth")
                logger.info("New best val MAE %.4f mmHg @ epoch %d -> checkpoint_best.pth", val_mae, done_epochs)
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
    def _sample_cond(self, cond_patches: torch.Tensor, demo=None) -> torch.Tensor:
        """Core sampler; assumes model is already in the desired eval/param state."""
        return sample_abp(
            self.model_raw, self.fm, cond_patches,
            generator=self.gen, device=self.device,
            abp_mean=float(self.cfg.data.abp_mean), abp_std=float(self.cfg.data.abp_std),
            cfg_strength=float(self.cfg.training.cfg_strength),
            autocast_ctx=self._autocast(),
            demo=demo,
        )

    @contextlib.contextmanager
    def _ema_swapped(self, use_ema: bool):
        """Temporarily load EMA params into the live model, then restore.

        Param swaps run under ``no_grad`` so this is safe even when the caller
        keeps autograd enabled (e.g. meta K-shot eval, whose inner loop needs
        grad): in-place copy_ on a leaf that requires grad would otherwise raise.
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
    def sample(self, cond_patches: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """cond_patches (B,N,2P) -> ABP waveform in mmHg (B,L)."""
        was_training = self.model_raw.training
        self.model_raw.eval()
        with self._ema_swapped(use_ema):
            out = self._sample_cond(cond_patches)
        if was_training:
            self.model_raw.train()
        return out

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
        keys = ("sbp_p", "dbp_p", "map_p", "sbp_t", "dbp_t", "map_t", "ae", "se", "r")
        cols: dict = {k: [] for k in keys}
        for bi, batch in enumerate(tqdm(loader, desc=desc, leave=False, disable=not is_main_process())):
            if 0 < max_batches <= bi:
                break
            pred = self._sample_cond(
                batch["cond_patches"], self._batch_demo(batch)
            )  # (b, L) mmHg on CPU
            gt = batch["abp_raw"]
            pbp, tbp = segment_bp(pred), segment_bp(gt)
            err = pred - gt
            cols["sbp_p"].append(pbp["SBP"]); cols["dbp_p"].append(pbp["DBP"]); cols["map_p"].append(pbp["MAP"])
            cols["sbp_t"].append(tbp["SBP"]); cols["dbp_t"].append(tbp["DBP"]); cols["map_t"].append(tbp["MAP"])
            cols["ae"].append(err.abs().mean(dim=1))   # (b,) per-segment MAE
            cols["se"].append((err ** 2).mean(dim=1))  # (b,) per-segment MSE
            cols["r"].append(_pearson(pred, gt))        # (b,) per-segment Pearson
        if was_training:
            self.model_raw.train()
        local = {k: torch.cat(v) for k, v in cols.items()} if cols["ae"] else None

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
        the per-segment-r mean, matching ``waveform_metrics``.
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
        return report

    @torch.no_grad()
    def validate(self) -> float:
        """Full-set, DDP-sharded validation; mean waveform MAE (mmHg) on rank 0
        (inf elsewhere). Uses the current in-memory EMA weights. ``val_max_batches``
        (if > 0) caps batches per rank; <= 0 means the full val split."""
        if self.val_loader is None:
            return float("inf")
        with self._ema_swapped(self.use_ema):
            report = self._distributed_eval(
                self.val_loader, "val", int(self.cfg.training.val_max_batches)
            )
        if report is None or not is_main_process():
            return float("inf")
        logger.info("[val] %s", format_report(report).replace("\n", " | "))
        self._sw_log(self._flat_report(report, "val"), self.global_step)
        return float(report["waveform"]["MAE"])

    # -- final test --------------------------------------------------------
    def _maybe_run_test(self) -> None:
        """After training, optionally evaluate on the CalFree test set.

        Gated by ``training.run_test_after_train``. Runs on ALL ranks (DDP-sharded
        + gathered), logs ``test/*`` to SwanLab and writes ``test_metrics.json``.
        Wrapped so a post-train eval failure never crashes an otherwise-finished run.
        """
        if not bool(self.cfg.training.run_test_after_train):
            return
        self._barrier()  # ensure rank 0 has flushed checkpoint_best.pth to disk
        try:
            self.run_test()
        except Exception as e:
            if is_main_process():
                logger.error("Post-train test evaluation failed: %s", e)

    @torch.no_grad()
    def run_test(self) -> None:
        """Evaluate the best-by-val (EMA) model on CalFree test; log + save metrics."""
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
        logger.info("[test] %d segments (weights=%s)\n%s", total, src, format_report(report))
        self._sw_log(self._flat_report(report, "test"), self.global_step)
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
                names = [n for n, _ in self.model_raw.named_parameters()]
                for p, n in zip(self.model_raw.parameters(), names):
                    p.data.copy_(ckpt["model_ema"][n].to(self.device))
                return "best/ema"
            self.model_raw.load_state_dict(ckpt["model"])
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
        }
        if self.ema_params is not None:
            names = [n for n, _ in self.model_raw.named_parameters()]
            ckpt["model_ema"] = {n: e.detach().cpu() for n, e in zip(names, self.ema_params)}
        path = os.path.join(self.exp_dir, filename)
        torch.save(ckpt, path)
        logger.info("Saved checkpoint -> %s (epoch %d, step %d)", path, epoch, self.global_step)

    def _maybe_resume(self) -> None:
        path = os.path.join(self.exp_dir, "checkpoint_latest.pth")
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model_raw.load_state_dict(ckpt["model"])
        if self.ema_params is not None and "model_ema" in ckpt:
            names = [n for n, _ in self.model_raw.named_parameters()]
            self.ema_params = [ckpt["model_ema"][n].to(self.device) for n in names]
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val = float(ckpt.get("best_val", float("inf")))
        self.epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        self.lr_scale = float(ckpt.get("lr_scale", 1.0))
        self._resumed = True
        if is_main_process():
            logger.info("Resumed from %s (epoch %d, step %d, lr_scale %.4g)",
                        path, self.start_epoch, self.global_step, self.lr_scale)

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
        self.model_raw.load_state_dict(ckpt["model"])
        if self.ema_params is not None and "model_ema" in ckpt:
            names = [n for n, _ in self.model_raw.named_parameters()]
            self.ema_params = [ckpt["model_ema"][n].to(self.device) for n in names]
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
