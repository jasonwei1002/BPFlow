"""CPU smoke for the scalar (cuff) ANIL meta-learning core (bpflow.meta).

Builds real same-subject episodes from Train_Subset (npy + sibling CSV subject_id
+ cuff sbp/dbp), runs a few outer meta-steps on a tiny model, then checks the two
things that must hold for the cuff mechanic to be real:

  1. the mean post-adaptation query loss drops over meta-training, and
  2. on HELD-OUT subjects, adapting phi on the support's CUFF SBP/DBP scalars
     lowers the query BPHead SBP/DBP error vs the unadapted phi=0 -- i.e. phi is
     calibrated from scalars alone (no support ABP), as a real cuff would.

    python -m bpflow.meta_smoke
    python -m bpflow.meta_smoke --steps 400 --ks 5 --kq 5
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .meta import adapt_context_scalar, meta_train_step
from .meta_data import load_bp_z, stack_segments, subject_groups
from .model import build_model
from .sampling import build_flow_matching
from .trainer_utils import load_config, pick_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _episode(arr, bp_z, idxs, ks, kq, cfg, device, rng):
    """Disjoint support/query draw from one subject -> 6-tuple episode."""
    P = int(cfg.model.patch_size)
    pick = idxs[rng.permutation(len(idxs))[: ks + kq]]
    sup, qry = pick[:ks], pick[ks: ks + kq]
    abp_s, cond_s, _ = stack_segments(arr, sup, cfg.data, P)
    abp_q, cond_q, _ = stack_segments(arr, qry, cfg.data, P)
    bp_s = torch.from_numpy(bp_z[sup]).to(device)
    bp_q = torch.from_numpy(bp_z[qry]).to(device)
    return (abp_s.to(device), cond_s.to(device), bp_s,
            abp_q.to(device), cond_q.to(device), bp_q)


def run_meta_smoke(config_path: str, steps: int = 400, ks: int = 5, kq: int = 5,
                   meta_bs: int = 4, k_inner: int = 3, inner_lr: float = 0.1, n_eval: int = 8) -> bool:
    set_seed(0)
    cfg = load_config(config_path)
    import omegaconf
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.model.use_context = True
    cfg.model.context_dim = 16
    cfg.meta.inner_objective = "scalar"
    cfg.meta.k_inner = k_inner
    cfg.meta.inner_lr = inner_lr
    cfg.meta.bp_loss_weight = 0.1
    omegaconf.OmegaConf.set_struct(cfg, True)
    device = pick_device("cpu")

    arr = np.load(str(cfg.data.train_npy), mmap_mode="r")
    bp_z = load_bp_z(str(cfg.data.train_npy), cfg)
    groups = subject_groups(str(cfg.data.train_npy), min_segs=ks + kq)
    subj = list(groups.keys())
    rng = np.random.default_rng(0)
    rng.shuffle(subj)
    train_subj, eval_subj = subj[n_eval:], subj[:n_eval]
    logger.info("Meta smoke (scalar/cuff): %d train / %d eval subjects (Ks=%d Kq=%d, steps=%d)",
                len(train_subj), len(eval_subj), ks, kq, steps)

    model = build_model(cfg).to(device)
    fm = build_flow_matching(cfg)
    gen = torch.Generator(device=device).manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), betas=(0.9, 0.95))

    first, last = [], []
    for step in range(steps):
        chosen = [train_subj[i] for i in rng.permutation(len(train_subj))[:meta_bs]]
        episodes = [_episode(arr, bp_z, groups[s], ks, kq, cfg, device, rng) for s in chosen]
        ql = meta_train_step(model, fm, episodes, cfg=cfg, generator=gen, optimizer=opt)
        (first if step < 20 else last if step >= steps - 20 else []).append(ql)
        if step % 50 == 0 or step == steps - 1:
            logger.info("meta step %d  query_loss %.4f", step, ql)
    init_q, final_q = float(np.mean(first)), float(np.mean(last))

    # held-out: does adapting phi on the CUFF scalars cut the query SBP/DBP error?
    model.eval()
    bp_err0, bp_err1, phi_norms, improved = [], [], [], 0
    for s in eval_subj:
        _, cond_s, bp_s, _, cond_q, bp_q = _episode(arr, bp_z, groups[s], ks, kq, cfg, device, rng)
        phi = adapt_context_scalar(model, cond_s, bp_s, cfg=cfg, k_inner=k_inner, inner_lr=inner_lr)
        phi_norms.append(float(phi.norm()))
        with torch.no_grad():
            e0 = float(F.mse_loss(model.predict_bp(cond_q, torch.zeros_like(phi)), bp_q))
            e1 = float(F.mse_loss(model.predict_bp(cond_q, phi), bp_q))
        bp_err0.append(e0); bp_err1.append(e1); improved += int(e1 < e0)
    mu0, mu1 = float(np.mean(bp_err0)), float(np.mean(bp_err1))

    logger.info("=" * 60)
    logger.info("meta query loss: init(20) %.4f -> final(20) %.4f  (ratio %.3f)",
                init_q, final_q, final_q / init_q)
    logger.info("adapted phi norm (mean) %.4f  (must be > 0: phi actually moved)", float(np.mean(phi_norms)))
    logger.info("held-out query SBP/DBP MSE(z):  phi=0 %.4f  ->  cuff-adapted %.4f  (delta %.4f); improved %d/%d",
                mu0, mu1, mu1 - mu0, improved, len(eval_subj))
    ok_train = final_q < init_q
    ok_adapt = (mu1 < mu0) and (improved > len(eval_subj) // 2)
    logger.info("=" * 60)
    if ok_train and ok_adapt:
        logger.info("META SMOKE PASS ✓  (meta loss dropped; cuff-scalar adaptation cuts query BP error)")
    else:
        logger.error("META SMOKE FAIL ✗  ok_train=%s ok_adapt=%s", ok_train, ok_adapt)
    return ok_train and ok_adapt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="bpflow/config/smoke.yaml")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--kq", type=int, default=5)
    ap.add_argument("--meta-bs", type=int, default=4)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--inner-lr", type=float, default=0.1)
    args = ap.parse_args()
    ok = run_meta_smoke(args.config, steps=args.steps, ks=args.ks, kq=args.kq,
                        meta_bs=args.meta_bs, k_inner=args.k_inner, inner_lr=args.inner_lr)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
