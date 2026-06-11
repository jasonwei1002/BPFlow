# @lint-ignore-every LICENSELINT
# BPFlow generator: 3-stream joint-attention DiT (WavFlow MMDiT, adapted).
# The ABP latent, ECG, and PPG are three token streams that attend jointly
# (BPJointBlock) for `joint_depth` layers, then the latent alone passes through
# `depth - joint_depth` single-stream blocks. Unlike WavFlow, all three streams
# share one RoPE grid (the signals are sample-aligned) and ECG/PPG use the same
# conv kernel (both are time signals). ECG and PPG are embedded by separate
# projections (they are distinct modalities) and exchange information with the
# ABP latent through joint attention -- explicitly modelling cross-modal cues
# such as the ECG->PPG pulse-transit-time phase lag.

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ._wavflow_layers import (
    MLP,
    ChannelLastConv1d,
    ConvMLP,
    FinalBlock,
    MMDitSingleBlock,
    TimestepEmbedder,
    compute_rope_rotations,
)
from .blocks import BPJointBlock

log = logging.getLogger(__name__)

# Continuous demographic channels [age, height, weight, bmi, body_missing_flag].
# Must match data.transforms.DEMO_CONT_DIM (the dataset emits this layout).
_DEMO_CONT_DIM = 5

# A demographics sample: (continuous (B, 5), gender index (B,) long).
DemoInput = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class BPConditions:
    """Cached condition tensors that do not depend on the latent or timestep."""

    ecg_seq: torch.Tensor  # (B, N, hidden)
    ppg_seq: torch.Tensor  # (B, N, hidden)
    pooled: torch.Tensor  # (B, hidden)
    demo_emb: Optional[torch.Tensor] = None  # (B, hidden) global demo prior, or None
    context_emb: Optional[torch.Tensor] = None  # (B, hidden) per-subject ANIL context, or None


class DemoEncoder(nn.Module):
    """Encode structured demographics into a global conditioning vector.

    Continuous fields go through a linear map; gender (binary) through an
    embedding; their sum is refined by an MLP. The final layer is zero-init so
    demographics start as a no-op (global_c unchanged) and are learned in.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cont_proj = nn.Linear(_DEMO_CONT_DIM, hidden_dim)
        self.gender_emb = nn.Embedding(2, hidden_dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, cont: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        h = self.cont_proj(cont) + self.gender_emb(gender)  # (B, hidden)
        return self.mlp(h)


class ContextEncoder(nn.Module):
    """Map a low-dim per-subject ANIL/CAVIA context vector into a global add-on.

    The context ``phi`` (context_dim,) or (B, context_dim) is the ONLY per-subject
    parameter adapted in the meta inner loop (the body is frozen there); this MLP
    that turns it into a global_c contribution is part of the meta-learned body.
    Zero biases (default init) make emb(phi=0)=0 -> phi=0 is the unadapted
    population default (a no-op add-on), while the weights stay nonzero so
    d(emb)/d(phi) != 0 and the inner loop can actually move phi (see
    initialize_weights for why this must NOT be zero-init'd).
    """

    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(context_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(phi))  # (..., hidden)


class BPHead(nn.Module):
    """Predict [SBP_z, DBP_z] from pooled ECG/PPG features + the per-subject context.

    The cheap, fully-differentiable scalar predictor the meta inner loop minimizes
    against the support's cuff readings, so phi is calibrated from SBP/DBP scalars
    alone -- no support ABP waveform needed (matches a real cuff). Sharing phi with
    the generative path is what ties "phi that fits the scalars" to "phi that makes
    the flow model generate ABP at that BP level".
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)  # (B, 2) = [SBP_z, DBP_z]


def _patch_embedder(in_dim: int, hidden: int, mlp_kernel: int) -> nn.Sequential:
    return nn.Sequential(
        ChannelLastConv1d(in_dim, hidden, kernel_size=7, padding=3),
        nn.SELU(),
        ConvMLP(hidden, hidden * 4, kernel_size=mlp_kernel, padding=mlp_kernel // 2),
    )


class BPFlowModel(nn.Module):
    """ECG+PPG -> ABP conditional flow-matching model (3-stream joint DiT).

    Works in normalized patch space: the ABP target and the ECG/PPG conditions
    are patchified on a shared grid (latent_seq_len = seq_len // patch_size).
    latent_dim = patch_size; the condition arrives as (B, N, 2P) and is split
    back into per-modality ECG/PPG (B, N, P) streams inside the model.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        patch_size: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        joint_depth: int,
        mlp_ratio: float = 4.0,
        use_demo: bool = False,
        use_context: bool = False,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError(f"seq_len {seq_len} must be divisible by patch_size {patch_size}")
        head_dim = hidden_dim // num_heads
        if hidden_dim % num_heads != 0 or head_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads} "
                f"and head_dim {head_dim} must be even (RoPE)"
            )
        if not 0 < joint_depth <= depth:
            raise ValueError(f"joint_depth {joint_depth} must be in (0, depth={depth}]")

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.latent_dim = patch_size  # ABP target patch dim
        self.cond_dim = 2 * patch_size  # condition delivered as (B,N,2P) then split
        self.latent_seq_len = seq_len // patch_size

        # Per-stream patch embedders (ECG and PPG each get their own).
        self.abp_input_proj = _patch_embedder(self.latent_dim, hidden_dim, mlp_kernel=7)
        self.ecg_input_proj = _patch_embedder(patch_size, hidden_dim, mlp_kernel=3)
        self.ppg_input_proj = _patch_embedder(patch_size, hidden_dim, mlp_kernel=3)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        self.t_embed = TimestepEmbedder(hidden_dim, frequency_embedding_size=256, max_period=10000)

        self.joint_blocks = nn.ModuleList(
            [
                BPJointBlock(hidden_dim, num_heads, mlp_ratio, pre_only=(i == joint_depth - 1))
                for i in range(joint_depth)
            ]
        )
        self.fused_blocks = nn.ModuleList(
            [
                MMDitSingleBlock(
                    hidden_dim, num_heads, mlp_ratio, pre_only=False, kernel_size=3, padding=1
                )
                for _ in range(depth - joint_depth)
            ]
        )
        self.final_layer = FinalBlock(hidden_dim, self.latent_dim)
        # optional demographic global-condition encoder (zero-init -> no-op start)
        self.use_demo = use_demo
        if use_demo:
            self.demo_encoder = DemoEncoder(hidden_dim)
        # optional ANIL/CAVIA per-subject context encoder (zero-init -> no-op start)
        self.use_context = use_context
        self.context_dim = context_dim
        if use_context:
            self.context_encoder = ContextEncoder(context_dim, hidden_dim)
            # cheap differentiable SBP/DBP predictor for scalar (cuff) inner-loop adaptation
            self.bp_head = BPHead(hidden_dim)

        self.initialize_weights()
        latent_rot = compute_rope_rotations(self.latent_seq_len, head_dim, 10000, device="cpu")
        self.latent_rot = nn.Buffer(latent_rot, persistent=False)

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        def _zero_adaln(blk: MMDitSingleBlock) -> None:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)

        for jb in self.joint_blocks:
            _zero_adaln(jb.latent_block)
            _zero_adaln(jb.ecg_block)
            _zero_adaln(jb.ppg_block)
        for fb in self.fused_blocks:
            _zero_adaln(fb)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)
        if self.use_demo:
            # zero-init the encoder output so demographics start as a no-op
            nn.init.constant_(self.demo_encoder.mlp[-1].weight, 0)
            nn.init.constant_(self.demo_encoder.mlp[-1].bias, 0)
        # NOTE: the context encoder is deliberately NOT zero-init'd. Zero biases
        # (from _basic_init) already make emb(phi=0)=0 (proj(0)=0, SiLU(0)=0), so
        # phi=0 is the unadapted population default / no-op. But the WEIGHTS must
        # stay nonzero, else d(emb)/d(phi)==0 and the inner loop can never move phi
        # (a dead branch). Nonzero weights give d(emb)/d(phi)|_0 = W_last·0.5·W_proj.

    def _embed_cond(self, ecg_p: torch.Tensor, ppg_p: torch.Tensor) -> BPConditions:
        ecg_seq = self.ecg_input_proj(ecg_p)  # (B, N, H)
        ppg_seq = self.ppg_input_proj(ppg_p)  # (B, N, H)
        pooled = self.global_cond_mlp((ecg_seq + ppg_seq).mean(dim=1))  # (B, H)
        return BPConditions(ecg_seq=ecg_seq, ppg_seq=ppg_seq, pooled=pooled)

    def _demo_emb(self, demo: Optional[DemoInput]) -> Optional[torch.Tensor]:
        """Encode a demographics sample, or None when demo conditioning is off."""
        if not self.use_demo or demo is None:
            return None
        cont, gender = demo
        return self.demo_encoder(cont, gender)  # (B, hidden)

    def _context_emb(self, context: Optional[torch.Tensor], bs: int) -> Optional[torch.Tensor]:
        """Encode the per-subject ANIL context ``phi`` into a (B, hidden) add-on.

        ``context`` is (context_dim,) — one vector shared by the whole batch (one
        subject per episode) — or (B, context_dim). Returns None when context
        conditioning is off. Differentiable in ``phi`` so the inner loop can
        adapt it; the expand keeps grads flowing to the single shared vector.
        """
        if not self.use_context or context is None:
            return None
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(bs, -1)  # (B, context_dim)
        return self.context_encoder(context)  # (B, hidden)

    def preprocess_conditions(
        self,
        cond_patches: torch.Tensor,
        demo: Optional[DemoInput] = None,
        context: Optional[torch.Tensor] = None,
    ) -> BPConditions:
        # cond_patches (B,N,2P) is channel-major [ECG(P), PPG(P)] -> split back.
        p = self.patch_size
        conds = self._embed_cond(cond_patches[..., :p], cond_patches[..., p:])
        conds.demo_emb = self._demo_emb(demo)
        conds.context_emb = self._context_emb(context, cond_patches.shape[0])
        return conds

    def predict_flow(
        self, latent_patches: torch.Tensor, t: torch.Tensor, conditions: BPConditions
    ) -> torch.Tensor:
        latent = self.abp_input_proj(latent_patches)  # (B, N, H)
        global_c = self.t_embed(t).unsqueeze(1) + conditions.pooled.unsqueeze(1)  # (B,1,H)
        if conditions.demo_emb is not None:
            global_c = global_c + conditions.demo_emb.unsqueeze(1)
        if conditions.context_emb is not None:
            global_c = global_c + conditions.context_emb.unsqueeze(1)
        ecg, ppg = conditions.ecg_seq, conditions.ppg_seq
        for block in self.joint_blocks:
            latent, ecg, ppg = block(latent, ecg, ppg, global_c, self.latent_rot)
        for block in self.fused_blocks:
            latent = block(latent, global_c, self.latent_rot)
        return self.final_layer(latent, global_c)  # (B, N, P)

    def forward(
        self,
        latent_patches: torch.Tensor,
        cond_patches: torch.Tensor,
        t: torch.Tensor,
        demo: Optional[DemoInput] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_flow(
            latent_patches, t, self.preprocess_conditions(cond_patches, demo, context)
        )

    def predict_bp(self, cond_patches: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Predict (B,2)=[SBP_z, DBP_z] from ECG/PPG + the context phi.

        Cheap (no ODE) and differentiable in phi, so the scalar meta inner loop can
        calibrate phi against the support's cuff SBP/DBP. Reuses the same pooled
        ECG/PPG features + context_emb the generative path conditions on.
        """
        conds = self.preprocess_conditions(cond_patches, None, context)
        feat = conds.pooled
        if conds.context_emb is not None:
            feat = feat + conds.context_emb
        return self.bp_head(feat)

    def ode_wrapper(
        self,
        t: torch.Tensor,
        latent: torch.Tensor,
        conditions: BPConditions,
    ) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        return self.predict_flow(latent, t, conditions)

    @property
    def device(self) -> torch.device:
        return self.latent_rot.device

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
