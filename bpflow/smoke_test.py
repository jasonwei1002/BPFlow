"""CPU overfit smoke test for BPFlow.

Verifies the full learning loop end-to-end WITHOUT a GPU: build model + flow
matching, overfit a small fixed set of real PulseDB segments for a few hundred
steps, and check that (1) the loss drops sharply and (2) sampled ABP
reconstructs the overfit targets far better than the untrained model. Also
dumps a gen-vs-GT plot for eyeballing.

Run:  python -m bpflow.smoke_test [--config bpflow/config/smoke.yaml]
"""

import argparse
import logging
import math
from pathlib import Path

import torch

from .data import TASK_SPEC, build_dataset
from .eval import evaluate, format_report
from .model import build_model
from .sampling import build_flow_matching, flow_matching_loss, sample_target
from .trainer_utils import load_config, pick_device, set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def _sample_mmhg(model, fm, ecg, ppg, target_idx, cond_present, gen, device, cfg) -> torch.Tensor:
    """Eval-mode TARGET sample (mmHg for ABP) for the overfit set."""
    model.eval()
    out = sample_target(
        model, fm, ecg, ppg, target_idx, cond_present, generator=gen, device=device,
        abp_mean=float(cfg.data.abp_mean), abp_std=float(cfg.data.abp_std),
        cond_recenter=bool(cfg.data.cond_recenter),
    )
    model.train()
    return out


def run_smoke(config_path: str, n_samples: int = 4, n_steps: int = 600, quick: bool = False) -> bool:
    set_seed(0)
    # This tiny model is dominated by CPU thread-pool overhead, not compute: measured
    # on an M-series CPU, 1 thread (~22 ms/step) beats 5 (~136 ms) and 15 (~175 ms), and
    # n_samples=4 (batch 8) avoids the BLAS parallel cliff that batch 16 trips (~4x). So
    # the full gate runs in ~13s instead of ~82s. Single-thread also makes the reductions
    # deterministic -> a more reproducible pass/fail. Override with --n-samples for more.
    torch.set_num_threads(1)
    # --quick: dev inner-loop check. Run the SAME pipeline (data -> forward -> loss ->
    # backward -> sample) for a handful of steps and assert nothing NaNs/crashes. It
    # DROPS the loss-collapse + recon gates (those need ~600 steps), so it catches
    # crashes/NaNs/shape bugs but NOT silent non-learning bugs -> run the default full
    # gate before committing. ~2-3s vs ~16s.
    if quick:
        n_steps = min(n_steps, 20)
    cfg = load_config(config_path)
    device = pick_device("cpu")  # smoke is CPU-only by definition
    logger.info("Smoke: model=%s P=%d hidden=%d depth=%d steps=%d",
                cfg.model.name, cfg.model.patch_size, cfg.model.hidden_dim, cfg.model.depth, n_steps)

    ds = build_dataset(cfg, "train")
    items = [ds[i] for i in range(n_samples)]
    abp = torch.stack([it["abp_patches"] for it in items]).to(device)
    ecg = torch.stack([it["ecg_patches"] for it in items]).to(device)
    ppg = torch.stack([it["ppg_patches"] for it in items]).to(device)
    abp_gt = torch.stack([it["abp_raw"] for it in items])  # (B,L) mmHg
    # Deterministic MULTI-TARGET overfit: train EVERY segment on BOTH ecg_ppg2abp
    # and ppg2ecg (the batch is duplicated, one copy per task), so a target≠ABP
    # trains AND every segment's ABP target is learned -> the ABP recon threshold is
    # stable. (Splitting tasks across segments would leave half the ABP unseen.)
    tA, cA = TASK_SPEC["ecg_ppg2abp"]  # full-info ABP recon (easiest -> stable threshold)
    tB, cB = TASK_SPEC["ppg2ecg"]       # a target≠ABP task (verifies multi-target routing)
    abp_tr = torch.cat([abp, abp], dim=0)
    ecg_tr = torch.cat([ecg, ecg], dim=0)
    ppg_tr = torch.cat([ppg, ppg], dim=0)
    target_idx = torch.tensor([tA] * n_samples + [tB] * n_samples, dtype=torch.long, device=device)
    cond_present = torch.tensor(
        [cA] * n_samples + [cB] * n_samples, dtype=torch.float32, device=device
    )
    # recon is scored on ecg_ppg2abp for the original segments (ABP, deterministic mmHg)
    eval_ti = torch.zeros(n_samples, dtype=torch.long, device=device)
    eval_cp = torch.tensor([cA] * n_samples, dtype=torch.float32, device=device)

    model = build_model(cfg).to(device)
    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), betas=(0.9, 0.95))

    # baseline (untrained) reconstruction — only needed for the recon gate (full mode)
    mae0 = float("nan")
    if not quick:
        gen.manual_seed(123)
        pred0 = _sample_mmhg(model, fm, ecg, ppg, eval_ti, eval_cp, gen, device, cfg)
        mae0 = (pred0 - abp_gt).abs().mean().item()

    losses = []
    model.train()
    for step in range(n_steps):
        loss = flow_matching_loss(
            model, fm, abp_tr, ecg_tr, ppg_tr, target_idx, cond_present, generator=gen,
            logit_mean=float(cfg.sampling.logit_mean), logit_scale=float(cfg.sampling.logit_scale),
            prediction_type=str(cfg.sampling.prediction_type), loss_type=str(cfg.training.loss_type),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.clip_grad_norm))
        opt.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == n_steps - 1:
            logger.info("step %d loss %.4f", step, losses[-1])

    init_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10

    gen.manual_seed(123)
    pred1 = _sample_mmhg(model, fm, ecg, ppg, eval_ti, eval_cp, gen, device, cfg)

    if quick:
        # crash-only verdict: the full pipeline ran; just assert no NaN/inf slipped
        # through and the sample has the expected (B, L) shape. NO learning gate.
        losses_finite = all(math.isfinite(x) for x in losses)
        sample_finite = bool(torch.isfinite(pred1).all().item())
        shape_ok = tuple(pred1.shape) == tuple(abp_gt.shape)
        logger.info("=" * 60)
        logger.info("QUICK: %d steps, loss %.4f -> %.4f | losses_finite=%s sample_finite=%s shape=%s",
                    n_steps, losses[0], losses[-1], losses_finite, sample_finite, tuple(pred1.shape))
        logger.info("=" * 60)
        if losses_finite and sample_finite and shape_ok:
            logger.info("SMOKE QUICK PASS ✓  (pipeline ran end-to-end, losses + sample finite)")
            return True
        logger.error("SMOKE QUICK FAIL ✗  losses_finite=%s sample_finite=%s shape_ok=%s",
                     losses_finite, sample_finite, shape_ok)
        return False

    mae1 = (pred1 - abp_gt).abs().mean().item()
    report = evaluate(pred1, abp_gt)

    # plot gen vs gt for the first 4 segments
    out_dir = Path(cfg.training.output_dir) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = min(4, n_samples)
        fig, axes = plt.subplots(k, 1, figsize=(10, 2.2 * k))
        if k == 1:
            axes = [axes]
        for j in range(k):
            axes[j].plot(abp_gt[j].numpy(), label="GT", lw=1.0)
            axes[j].plot(pred1[j].numpy(), label="gen", lw=1.0, alpha=0.8)
            axes[j].set_ylabel("mmHg")
            axes[j].legend(loc="upper right", fontsize=8)
        axes[0].set_title("BPFlow smoke: overfit reconstruction (gen vs GT)")
        fig.tight_layout()
        fig.savefig(out_dir / "smoke_recon.png", dpi=110)
        plt.close(fig)
        logger.info("Saved plot -> %s", out_dir / "smoke_recon.png")
    except Exception as e:  # plotting is optional
        logger.warning("plot skipped: %s", e)

    logger.info("=" * 60)
    logger.info("loss: init %.4f -> final %.4f  (ratio %.3f)", init_loss, final_loss, final_loss / init_loss)
    logger.info("recon MAE (mmHg): untrained %.2f -> trained %.2f", mae0, mae1)
    logger.info("\n%s", format_report(report))
    logger.info("=" * 60)

    # success criteria: loss collapsed AND trained recon clearly beats untrained.
    # recon threshold 0.7 (not 0.6): the tiny CPU model splits capacity across the
    # two overfit tasks (ABP recon + ppg2ecg), so ABP recon plateaus ~21 mmHg
    # (untrained ~33) in 600 steps at the default n_samples=4 (ratio ~0.64). A broken
    # pipeline stays ~1.0, real learning hits ~0.64 -> 0.7 still cleanly separates the
    # two; this is a pipeline gate, not a quality bar.
    ok_loss = final_loss < 0.5 * init_loss
    ok_recon = mae1 < 0.7 * mae0
    if ok_loss and ok_recon:
        logger.info("SMOKE PASS ✓  (loss collapsed, reconstruction improved)")
        return True
    logger.error("SMOKE FAIL ✗  ok_loss=%s ok_recon=%s", ok_loss, ok_recon)
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="BPFlow CPU overfit smoke test")
    ap.add_argument("--config", default="bpflow/config/smoke.yaml")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--quick", action="store_true",
                    help="fast crash-only check (<=20 steps, no loss/recon gate) for the "
                         "dev inner loop; ~2-3s. Run the default full gate before committing.")
    args = ap.parse_args()
    ok = run_smoke(args.config, args.n_samples, args.n_steps, quick=args.quick)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
