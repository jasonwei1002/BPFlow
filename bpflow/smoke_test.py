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
from pathlib import Path

import torch

from .data import build_dataset
from .eval import evaluate, format_report
from .model import build_model
from .sampling import build_flow_matching, flow_matching_loss, sample_abp
from .trainer_utils import load_config, pick_device, set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def _sample_mmhg(model, fm, cond, gen, device, cfg, cond_mask=None) -> torch.Tensor:
    """Eval-mode ABP sample in mmHg for the overfit set."""
    model.eval()
    out = sample_abp(
        model, fm, cond, generator=gen, device=device,
        abp_mean=float(cfg.data.abp_mean), abp_std=float(cfg.data.abp_std),
        cond_mask=cond_mask,
    )
    model.train()
    return out


def run_smoke(config_path: str, n_samples: int = 8, n_steps: int = 600) -> bool:
    set_seed(0)
    cfg = load_config(config_path)
    device = pick_device("cpu")  # smoke is CPU-only by definition
    logger.info("Smoke: model=%s P=%d hidden=%d depth=%d steps=%d",
                cfg.model.name, cfg.model.patch_size, cfg.model.hidden_dim, cfg.model.depth, n_steps)

    ds = build_dataset(cfg, "train")
    items = [ds[i] for i in range(n_samples)]
    abp = torch.stack([it["abp_patches"] for it in items]).to(device)
    cond = torch.stack([it["cond_patches"] for it in items]).to(device)
    cond_mask = torch.stack([it["cond_mask"] for it in items]).to(device)
    abp_gt = torch.stack([it["abp_raw"] for it in items])  # (B,L) mmHg

    model = build_model(cfg).to(device)
    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), betas=(0.9, 0.95))

    # baseline (untrained) reconstruction
    gen.manual_seed(123)
    pred0 = _sample_mmhg(model, fm, cond, gen, device, cfg, cond_mask)
    mae0 = (pred0 - abp_gt).abs().mean().item()

    losses = []
    model.train()
    for step in range(n_steps):
        loss = flow_matching_loss(
            model, fm, abp, cond, generator=gen,
            logit_mean=float(cfg.sampling.logit_mean), logit_scale=float(cfg.sampling.logit_scale),
            prediction_type=str(cfg.sampling.prediction_type), loss_type=str(cfg.training.loss_type),
            cond_mask=cond_mask,
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
    pred1 = _sample_mmhg(model, fm, cond, gen, device, cfg, cond_mask)
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

    # success criteria: loss collapsed AND trained recon clearly beats untrained
    ok_loss = final_loss < 0.5 * init_loss
    ok_recon = mae1 < 0.6 * mae0
    if ok_loss and ok_recon:
        logger.info("SMOKE PASS ✓  (loss collapsed, reconstruction improved)")
        return True
    logger.error("SMOKE FAIL ✗  ok_loss=%s ok_recon=%s", ok_loss, ok_recon)
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="BPFlow CPU overfit smoke test")
    ap.add_argument("--config", default="bpflow/config/smoke.yaml")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=600)
    args = ap.parse_args()
    ok = run_smoke(args.config, args.n_samples, args.n_steps)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
