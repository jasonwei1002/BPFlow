"""Device-agnostic BPFlow trainer.

Single-stream conditional flow matching: ECG+PPG -> ABP. Runs on CPU
(smoke), single GPU, or DDP (torchrun, WORLD_SIZE>1). All CUDA use is gated
on availability so a tiny CPU smoke test never hits a hard cuda call.
"""

import contextlib
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import build_dataset
from .eval import evaluate, format_report
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
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(cfg.training.seed) + self.rank)
        self._maybe_resume()

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
        # Validation loader (built only on the main process; in-loop val runs
        # there). val split is a held-out 20% of Train_Subset (same seed).
        self.val_loader = None
        if int(self.cfg.training.val_freq_epoch) > 0 and is_main_process():
            val_ds = build_dataset(self.cfg, "val")
            self.val_loader = DataLoader(
                val_ds,
                batch_size=int(self.cfg.training.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.training.num_workers),
                drop_last=False,
                pin_memory=self.is_cuda,
            )

    def _build_model(self) -> None:
        model = build_model(self.cfg).to(self.device)
        self.model_raw = model
        if self.distributed and self.is_cuda:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], broadcast_buffers=False
            )
        else:
            self.model = model
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
    def _flat_val(report: dict) -> dict:
        """Flatten the eval report into scalar metrics for logging."""
        out = {f"val/{k}": float(v) for k, v in report["waveform"].items()}
        for key in ("SBP", "DBP", "MAP"):
            a, b = report[key]["AAMI"], report[key]["BHS"]
            out[f"val/{key}_ME"] = float(a["ME"])
            out[f"val/{key}_SDE"] = float(a["SDE"])
            out[f"val/{key}_AAMI_pass"] = float(a["pass"])
            out[f"val/{key}_within5mmHg"] = float(b["<=5mmHg"])
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
        # optional classifier-free training: drop condition to learned null
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
        return abp, cond

    def _train_step(self, batch) -> dict:
        lr = adjust_learning_rate(self.optimizer, self.global_step, self.cfg)
        with self._autocast():
            abp, cond = self._prepare_batch(batch)
            loss = flow_matching_loss(
                self.model, self.fm, abp, cond,
                generator=self.gen,
                logit_mean=float(self.cfg.sampling.logit_mean),
                logit_scale=float(self.cfg.sampling.logit_scale),
                prediction_type=str(self.cfg.sampling.prediction_type),
                loss_type=str(self.cfg.training.loss_type),
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
            if is_main_process() and ckpt_freq > 0 and done % ckpt_freq == 0:
                self.save_checkpoint(done, "checkpoint_latest.pth")
            self._barrier()
        if val_freq > 0:
            self._run_validation(n_epochs)
        if is_main_process():
            logger.info("Training done in %.1fs", time.time() - start)
            self.save_checkpoint(n_epochs, "checkpoint_latest.pth")

    def _run_validation(self, done_epochs: int) -> None:
        val_mae = self.validate()
        if is_main_process() and val_mae < self.best_val:
            self.best_val = val_mae
            self.save_checkpoint(done_epochs, "checkpoint_best.pth")
            logger.info("New best val MAE %.3f mmHg @ epoch %d -> checkpoint_best.pth", val_mae, done_epochs)
        self._barrier()

    # -- sampling ----------------------------------------------------------
    @torch.no_grad()
    def _sample_cond(self, cond_patches: torch.Tensor) -> torch.Tensor:
        """Core sampler; assumes model is already in the desired eval/param state."""
        return sample_abp(
            self.model_raw, self.fm, cond_patches,
            generator=self.gen, device=self.device,
            abp_mean=float(self.cfg.data.abp_mean), abp_std=float(self.cfg.data.abp_std),
            cfg_strength=float(self.cfg.training.cfg_strength),
            autocast_ctx=self._autocast(),
        )

    @contextlib.contextmanager
    def _ema_swapped(self, use_ema: bool):
        """Temporarily load EMA params into the live model, then restore."""
        if use_ema and self.ema_params is not None:
            backup = [p.detach().clone() for p in self.model_raw.parameters()]
            for p, e in zip(self.model_raw.parameters(), self.ema_params):
                p.copy_(e)
            try:
                yield
            finally:
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
    def validate(self) -> float:
        """Generate on the held-out val split; return mean waveform MAE (mmHg)."""
        if self.val_loader is None or not is_main_process():
            return float("inf")
        was_training = self.model_raw.training
        self.model_raw.eval()
        preds, gts = [], []
        max_b = int(self.cfg.training.val_max_batches)
        with self._ema_swapped(self.use_ema):
            for bi, batch in enumerate(self.val_loader):
                if 0 < max_b <= bi:
                    break
                preds.append(self._sample_cond(batch["cond_patches"]))
                gts.append(batch["abp_raw"])
        if was_training:
            self.model_raw.train()
        if not preds:
            return float("inf")
        report = evaluate(torch.cat(preds), torch.cat(gts))
        logger.info("[val] %s", format_report(report).replace("\n", " | "))
        self._sw_log(self._flat_val(report), self.global_step)
        return float(report["waveform"]["MAE"])

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
        if is_main_process():
            logger.info("Resumed from %s (epoch %d, step %d)", path, self.start_epoch, self.global_step)

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
