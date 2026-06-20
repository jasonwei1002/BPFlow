"""Task-routing attention mask + mask-aware attention for multi-target BPFlow.

The vendored ``attention`` (``model/_vendor/transformer_layers.py``) is a
read-only snapshot that takes no mask. Symmetric multi-target routing (a
per-sample task = which modality is the noised TARGET and which are the clean
CONDITIONS) needs an additive mask over the concatenated [ABP|ECG|PPG] token
sequence, so this module supplies the mask builder and a mask-aware twin of
``attention``.

Routing (UniCardio-style, asymmetric — see plan/task_plan_multitask_translation.md, D3):
  - a TARGET-stream query attends to: its own stream + every present CONDITION stream
  - a CONDITION-stream query attends to: present CONDITION streams only (NOT the
    noised target — keep clean conditions from absorbing target noise)
  - an ABSENT-stream query attends to: itself only (its output is discarded; this
    only keeps the query row non-empty so softmax never sees an all-masked row)
The diagonal (a stream attends itself) is always open, guaranteeing no all-``-inf``
row -> no softmax NaN.
"""

from typing import Optional

import torch
import torch.nn.functional as F

NUM_STREAMS = 3  # concatenated joint order: 0 = ABP, 1 = ECG, 2 = PPG


@torch.compiler.disable
def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``_vendor.attention`` + an optional additive ``attn_mask`` on the scores.

    Mirrors the vendored kernel exactly (the contiguous() calls work around a
    cuDNN limitation, see the vendor note) but forwards ``attn_mask`` to SDPA.

    Runs EAGER under ``@torch.compiler.disable``: with an additive float mask, the
    SDPA output + head-merge view trips torch.compile's AOTAutograd alias
    reconstruction ("Cannot view a tensor ... as ..."), crashing the compiled
    multi-GPU forward at the first joint block. SDPA is already a single fused
    kernel, so running attention eagerly costs ~nothing; the rest of the block
    (norms, AdaLN, FFN, projections) still compiles. A graph break here is benign
    under DDPOptimizer. Verified correct in eager by the CPU smoke test.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # merge heads: (B, H, N, D) -> (B, N, H*D), contiguous before the view
    b, h, n, d = out.shape
    out = out.transpose(1, 2).contiguous().view(b, n, h * d)
    return out


def build_task_mask(
    target_idx: torch.Tensor,    # (B,) long in {0..NUM_STREAMS-1}
    cond_present: torch.Tensor,  # (B, NUM_STREAMS) in {0,1}: present-as-condition
    n_tokens: int,               # tokens per stream (N)
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``(B, 1, S*N, S*N)`` mask: ``0`` where attention is allowed, ``-inf``
    where blocked (S = NUM_STREAMS). ``cond_present`` for the target's own slot is
    ignored (a target is never its own condition)."""
    b = target_idx.shape[0]
    idx = torch.arange(NUM_STREAMS, device=device)
    qs = idx.view(1, NUM_STREAMS, 1)
    ks = idx.view(1, 1, NUM_STREAMS)
    tgt = target_idx.view(b, 1, 1).to(device)
    # the target slot is never treated as a condition, even if cond_present says so
    cond = cond_present.bool().to(device) & (idx.view(1, NUM_STREAMS) != target_idx.view(b, 1))
    condq = cond.view(b, NUM_STREAMS, 1)
    condk = cond.view(b, 1, NUM_STREAMS)
    is_tq = qs == tgt
    is_tk = ks == tgt
    allow = (
        (is_tq & (is_tk | condk))           # target query  -> target + present conditions
        | (condq & condk)                    # condition query -> present conditions
        | (qs == ks)                         # diagonal always open (absent self; no empty row)
    )  # (B, S, S) bool
    block = torch.zeros(b, NUM_STREAMS, NUM_STREAMS, dtype=dtype, device=device)
    block.masked_fill_(~allow, float("-inf"))
    # stream-level (B, S, S) -> token-level (B, S*N, S*N), then add a head axis
    add = block.repeat_interleave(n_tokens, dim=1).repeat_interleave(n_tokens, dim=2)
    return add.unsqueeze(1)  # (B, 1, S*N, S*N)
