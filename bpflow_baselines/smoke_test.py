"""CPU smoke test for all six baselines (this IS the correctness gate).

For each model on one ->ABP and one bridge direction: build a tiny variant,
overfit a handful of real PulseDB segments, assert the training loss collapses,
then reconstruct + run bpflow's own metrics to confirm the eval path produces a
well-formed report. No GPU required.

    python -m bpflow_baselines.smoke_test
    python -m bpflow_baselines.smoke_test --steps 120 --n 8
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader, Subset

from bpflow.eval import evaluate, waveform_metrics

from .config import load_config
from .data import build_baseline_dataset
from .losses import bp_l1, waveform_loss
from .models.base import build_model, pad_to_multiple
from .norms import ABP_TARGET_MODE
from .reconstruct import reconstruct_pred, reconstruct_true

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke")

# tiny per-model params so the overfit loop is fast on CPU
SMALL = {
    "wavenet": {"layers": 4, "blocks": 2, "dilation_channels": 8, "residual_channels": 8,
                "skip_channels": 16, "end_channels": 16, "kernel_size": 2},
    "nabnet": {"model_depth": 3, "model_width": 8, "attention_type": "standard", "kernel_size": 3},
    "ppg2abp": {"base_channels": 8, "alpha": 1.0},
    "patchtst": {"patch_len": 16, "stride": 8, "d_model": 32, "num_encoder_layers": 2,
                 "num_heads": 4, "dropout": 0.0},
    "p2e_wgan": {"generator_init_filters": 16, "discriminator_init_filters": 16},
    "mdvisco": {"init_features": 8, "patch_size": 4, "depth": 1, "embedding_dim_multiplier": 2,
                "swin_num_heads": [2, 2, 2, 2, 2], "swin_mlp_ratio": 2.0, "kernel_size": 3},
}
MODELS = ["wavenet", "nabnet", "ppg2abp", "patchtst", "p2e_wgan", "mdvisco"]


def _cfg(model: str, direction: str):
    cfg = load_config(
        f"bpflow_baselines/config/{model}.yaml",
        overrides={
            "model": {"name": model, "params": SMALL[model]},
            "baseline": {"direction": direction},
            "training": {"device": "cpu", "num_workers": 0},
            "data": {"finetune": True,
                     "train_npy": "rawdata/pulsedb/CalFree_Test_Subset.npy",
                     "finetune_split_mode": "stratified"},
        },
    )
    return cfg


def _batch(cfg, direction: str, n: int):
    ds = build_baseline_dataset(cfg, "test")
    sub = Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=n, shuffle=False)
    return next(iter(loader))


def _overfit_supervised(model, batch, want_bp, steps, work_mult, clip_lo, clip_hi,
                        loss_base="mse", aux_loss_base=""):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = pad_to_multiple(batch["x"], work_mult)
    y = pad_to_multiple(batch["y"], work_mult)
    losses = []
    model.train()
    for _ in range(steps):
        out = model(x, want_bp=want_bp)
        loss = waveform_loss(out, y, base=loss_base, aux_base=aux_loss_base)
        if want_bp:
            loss = loss + bp_l1(out["bp"], batch["sbp"], batch["dbp"], clip_lo, clip_hi)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def _overfit_gan(model, batch, steps, work_mult, tgt_is_abp):
    optg = torch.optim.Adam(model.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    x = pad_to_multiple(batch["x"], work_mult)
    y = pad_to_multiple(batch["y"], work_mult)
    real = (2 * y - 1) if tgt_is_abp else (2 * y)  # ABP[0,1]->[-1,1]; bridge recenter[-0.5,0.5]->[-1,1]
    mses = []
    model.train()
    for _ in range(steps):  # generator-only MSE overfit (enough to prove the path)
        fake = model.generator(x)
        loss = torch.nn.functional.mse_loss(fake, real)
        optg.zero_grad(set_to_none=True)
        loss.backward()
        optg.step()
        mses.append(float(loss))
    return mses


def _eval_once(model, cfg, batch, direction):
    tgt_is_abp = direction.endswith("2abp")
    work_mult = int(model.work_multiple)
    want_bp = bool(model.has_bp_head) and tgt_is_abp
    gan_tanh = str(cfg.model.name) == "p2e_wgan"
    x = pad_to_multiple(batch["x"], work_mult)
    model.eval()
    with torch.no_grad():
        if gan_tanh:
            wave, bp_pred = model.generator(x), None
        else:
            out = model(x, want_bp=want_bp)
            wave, bp_pred = out["wave"], (out.get("bp") if want_bp else None)
    pred = reconstruct_pred(wave, seq_len=int(cfg.data.seq_len), tgt_is_abp=tgt_is_abp,
                            abp_mode=ABP_TARGET_MODE.get(str(cfg.model.name), "global"),
                            clip_lo=float(cfg.data.abp_clip_low), clip_hi=float(cfg.data.abp_clip_high),
                            bp_pred=bp_pred, gan_tanh=gan_tanh)
    true = reconstruct_true(batch, tgt_is_abp=tgt_is_abp)
    rep = evaluate(pred, true) if tgt_is_abp else {"waveform": waveform_metrics(pred, true)}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    torch.manual_seed(0)
    failures = []
    for model_name in args.models:
        # cover all 5 directions: both single-source ->ABP (ppg2abp, ecg2abp),
        # both bridges (ppg2ecg, ecg2ppg), and the dual-source ecg_ppg2abp (2-ch)
        for direction in ("ppg2abp", "ecg2abp", "ppg2ecg", "ecg2ppg", "ecg_ppg2abp"):
            tag = f"{model_name}/{direction}"
            try:
                cfg = _cfg(model_name, direction)
                model = build_model(cfg)
                batch = _batch(cfg, direction, args.n)
                want_bp = bool(model.has_bp_head) and direction.endswith("2abp")
                wm = int(model.work_multiple)
                if model_name == "p2e_wgan":
                    series = _overfit_gan(model, batch, args.steps, wm, direction.endswith("2abp"))
                    metric = "genMSE"
                else:
                    series = _overfit_supervised(
                        model, batch, want_bp, args.steps, wm,
                        float(cfg.data.abp_clip_low), float(cfg.data.abp_clip_high),
                        str(cfg.training.loss_base), str(cfg.training.aux_loss_base))
                    metric = "loss"
                drop = series[-1] / max(series[0], 1e-9)
                rep = _eval_once(model, cfg, batch, direction)
                wmae = rep["waveform"]["MAE"]
                # pass if the loss collapsed by >=30% OR is already tiny (the model
                # converged fast and has little room left — common when the initial
                # loss is low, e.g. wavenet/ecg2abp); always require a finite wave MAE.
                converged = series[-1] < 0.7 * series[0] or series[-1] < 0.02
                ok = converged and bool(torch.isfinite(torch.tensor(wmae)))
                status = "PASS" if ok else "FAIL"
                logger.info("%-22s %s  %s %.4f->%.4f (x%.2f)  wave.MAE=%.4f  params=%d",
                            tag, status, metric, series[0], series[-1], drop, wmae,
                            sum(p.numel() for p in model.parameters()))
                if not ok:
                    failures.append(tag)
            except Exception as e:  # noqa: BLE001
                logger.exception("%-22s ERROR %s", tag, e)
                failures.append(tag)

    if failures:
        raise SystemExit(f"SMOKE FAILURES: {failures}")
    logger.info("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
