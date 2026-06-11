"""ANIL/CAVIA first-order meta-learning for sample-efficient subject adaptation.

The per-subject context vector ``phi`` (``model.context_dim``) is the ONLY thing
adapted in the inner loop; the body + ``context_encoder`` (+ ``bp_head``) are the
meta-learned parameters and stay frozen inside the inner loop. First-order
(FOMAML/Reptile family): the inner loop uses ``torch.autograd.grad(...,
create_graph=False)`` so the outer update never differentiates through the inner
steps -- DDP- and ``torch.compile``-safe.

Inner objective (``meta.inner_objective``):
  - "scalar"   — cuff-only, the realistic deployment: adapt phi by matching the
    support's cuff [SBP, DBP] through the cheap differentiable ``BPHead``; the
    support carries ECG/PPG + scalars, NO ABP waveform.
  - "waveform" — reference-ABP: adapt phi by the flow-matching loss on the
    support's ABP (needs the support waveform).

DDP: the meta loop runs on the raw (un-wrapped) model and all-reduces gradients
MANUALLY once per outer step (``reduce_grads``). This is simpler and more correct
than DDP's autograd hooks here -- the per-episode structure does many backwards,
and ``bp_head`` is used outside the generative forward (DDP wouldn't track it).

Episode (6-tuple): (abp_s, cond_s, bp_s, abp_q, cond_q, bp_q); bp_* are z-scored
[SBP, DBP] (n, 2). ``abp_s`` is unused by the scalar objective.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .sampling import flow_matching_loss, sample_abp

Episode = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def adapt_context(
    model_raw: torch.nn.Module,
    fm,
    abp_support: torch.Tensor,
    cond_support: torch.Tensor,
    *,
    cfg,
    k_inner: int,
    inner_lr: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Waveform inner loop: adapt phi by the flow-matching loss on support ABP."""
    dim = int(cfg.model.context_dim)
    phi = torch.zeros(dim, device=abp_support.device, requires_grad=True)
    for _ in range(int(k_inner)):
        loss = flow_matching_loss(
            model_raw, fm, abp_support, cond_support,
            generator=generator,
            logit_mean=float(cfg.sampling.logit_mean),
            logit_scale=float(cfg.sampling.logit_scale),
            prediction_type=str(cfg.sampling.prediction_type),
            loss_type=str(cfg.training.loss_type),
            context=phi,
        )
        (grad,) = torch.autograd.grad(loss, phi)
        phi = (phi - float(inner_lr) * grad).detach().requires_grad_(True)
    return phi.detach()


def adapt_context_scalar(
    model_raw: torch.nn.Module,
    cond_support: torch.Tensor,
    bp_support: torch.Tensor,
    *,
    cfg,
    k_inner: int,
    inner_lr: float,
) -> torch.Tensor:
    """Cuff inner loop: adapt phi by matching the support's [SBP, DBP] via BPHead.

    No ABP waveform is used -- only ECG/PPG (``cond_support``) and the cuff scalars
    (``bp_support``, z-scored). Cheap (one BPHead forward per step) and first-order.
    """
    dim = int(cfg.model.context_dim)
    phi = torch.zeros(dim, device=cond_support.device, requires_grad=True)
    for _ in range(int(k_inner)):
        pred = model_raw.predict_bp(cond_support, phi)  # (Ks, 2)
        loss = F.mse_loss(pred, bp_support)
        (grad,) = torch.autograd.grad(loss, phi)
        phi = (phi - float(inner_lr) * grad).detach().requires_grad_(True)
    return phi.detach()


def _adapt(model_raw, fm, ep: Episode, *, cfg, generator) -> torch.Tensor:
    """Run the configured inner objective for one episode, return the adapted phi."""
    abp_s, cond_s, bp_s, _abp_q, _cond_q, _bp_q = ep
    k_inner, inner_lr = int(cfg.meta.k_inner), float(cfg.meta.inner_lr)
    if str(cfg.meta.inner_objective) == "scalar":
        return adapt_context_scalar(model_raw, cond_s, bp_s, cfg=cfg, k_inner=k_inner, inner_lr=inner_lr)
    return adapt_context(model_raw, fm, abp_s, cond_s, cfg=cfg, k_inner=k_inner,
                         inner_lr=inner_lr, generator=generator)


def meta_train_step(
    model_raw: torch.nn.Module,
    fm,
    episodes: List[Episode],
    *,
    cfg,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    reduce_grads=None,
) -> float:
    """One outer ANIL meta-step over a meta-batch of subject episodes.

    Per episode: adapt phi on support (first-order inner), then accumulate the
    query loss's gradient into the body + context_encoder (+ bp_head). The query
    loss is the generative flow-matching loss on the query ABP plus (scalar mode)
    a ``bp_loss_weight`` term keeping BPHead accurate so phi's calibration stays
    meaningful. After the meta-batch: optional manual grad all-reduce (DDP), then
    one optimizer step. Returns the mean query loss.
    """
    optimizer.zero_grad(set_to_none=True)
    n = len(episodes)
    scalar = str(cfg.meta.inner_objective) == "scalar"
    lam = float(cfg.meta.bp_loss_weight)
    total = 0.0
    for ep in episodes:
        _abp_s, _cond_s, _bp_s, abp_q, cond_q, bp_q = ep
        phi = _adapt(model_raw, fm, ep, cfg=cfg, generator=generator)
        loss_q = flow_matching_loss(
            model_raw, fm, abp_q, cond_q,
            generator=generator,
            logit_mean=float(cfg.sampling.logit_mean),
            logit_scale=float(cfg.sampling.logit_scale),
            prediction_type=str(cfg.sampling.prediction_type),
            loss_type=str(cfg.training.loss_type),
            context=phi,
        )
        if scalar and lam > 0.0:
            loss_q = loss_q + lam * F.mse_loss(model_raw.predict_bp(cond_q, phi), bp_q)
        (loss_q / n).backward()
        total += float(loss_q.detach())
    if reduce_grads is not None:
        reduce_grads(model_raw)
    optimizer.step()
    return total / n


def kshot_evaluate(
    model_raw: torch.nn.Module,
    fm,
    cfg,
    arr,
    bp_z: np.ndarray,
    subj_to_idx: Dict[object, np.ndarray],
    subjects: List[object],
    ks_list: List[int],
    *,
    device: torch.device,
    generator: torch.Generator,
    max_subjects: int = -1,
    max_query: int = 40,
) -> Dict[int, dict]:
    """Subject-disjoint K-shot evaluation -> ``{K: waveform/clinical report}``.

    Per subject a fixed permutation fixes a held-out query block (identical across
    all K) and nested supports (prefix). For each K>0 the inner loop adapts phi on
    the first K segments (scalar = cuff SBP/DBP from ``bp_z``, waveform = support
    ABP); K=0 is the calibration-free baseline (phi=0). The model must already be
    in the desired eval/param state.
    """
    from .meta_data import stack_segments
    from .eval import evaluate

    d = cfg.data
    P = int(cfg.model.patch_size)
    scalar = str(cfg.meta.inner_objective) == "scalar"
    ks_list = sorted(set(int(k) for k in ks_list))
    max_k = max(ks_list)
    subs = subjects if max_subjects <= 0 else subjects[:max_subjects]
    rng = np.random.default_rng(int(cfg.training.seed))

    perms: Dict[object, np.ndarray] = {}
    for s in subs:
        idxs = subj_to_idx[s]
        if len(idxs) > max_k:
            perms[s] = idxs[rng.permutation(len(idxs))]

    reports: Dict[int, dict] = {}
    for k in ks_list:
        preds, gts = [], []
        for s, perm in perms.items():
            qry = perm[max_k: max_k + max_query] if max_query > 0 else perm[max_k:]
            if len(qry) == 0:
                continue
            if k > 0:
                sup = perm[:k]
                if scalar:
                    _, cond_s, _ = stack_segments(arr, sup, d, P)
                    bp_s = torch.from_numpy(bp_z[sup]).to(device)
                    phi = adapt_context_scalar(
                        model_raw, cond_s.to(device), bp_s,
                        cfg=cfg, k_inner=int(cfg.meta.k_inner), inner_lr=float(cfg.meta.inner_lr),
                    )
                else:
                    abp_s, cond_s, _ = stack_segments(arr, sup, d, P)
                    phi = adapt_context(
                        model_raw, fm, abp_s.to(device), cond_s.to(device),
                        cfg=cfg, k_inner=int(cfg.meta.k_inner), inner_lr=float(cfg.meta.inner_lr),
                        generator=generator,
                    )
            else:
                phi = torch.zeros(int(cfg.model.context_dim), device=device)
            _, cond_q, raw_q = stack_segments(arr, qry, d, P)
            pred = sample_abp(
                model_raw, fm, cond_q.to(device),
                generator=generator, device=device,
                abp_mean=float(d.abp_mean), abp_std=float(d.abp_std), context=phi,
            )
            preds.append(pred)
            gts.append(raw_q)
        if preds:
            reports[k] = evaluate(torch.cat(preds), torch.cat(gts))
    return reports
