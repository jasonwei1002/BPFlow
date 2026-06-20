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
from .attention_mask import build_task_mask
from .blocks import BPJointBlock

log = logging.getLogger(__name__)

# Continuous demographic channels [age, height, weight, bmi, body_missing_flag].
# Must match data.transforms.DEMO_CONT_DIM (the dataset emits this layout).
_DEMO_CONT_DIM = 5

# A demographics sample: (continuous (B, 5), gender index (B,) long).
DemoInput = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class BPConditions:
    """Per-task condition tensors that are constant across ODE steps.

    For the symmetric multi-target setup, the only thing that changes across ODE
    steps is the noised TARGET; everything here (the clean-condition / absent
    contribution per stream, the global pooled vector, the task attention mask,
    the per-sample target index + role one-hot) is precomputed once.
    """

    stream_base: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # cond+absent part per stream
    is_target: torch.Tensor  # (B, 3) one-hot float: which stream is the noised target
    target_idx: torch.Tensor  # (B,) long
    pooled: torch.Tensor  # (B, hidden) global condition (present conditions only)
    attn_mask: torch.Tensor  # (B, 1, 3N, 3N) additive task-routing mask
    demo_emb: Optional[torch.Tensor] = None  # (B, hidden) global demo prior, or None


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

        # Symmetric multi-target embedders (stream order 0=ABP, 1=ECG, 2=PPG).
        # A stream can be the noised TARGET (embedded by noised_in[s]) or a clean
        # CONDITION (cond_in); an absent stream is filled with a learned
        # absent_token[s] and masked out of attention. ABP is never a condition in
        # the task set -> it has no cond_in.
        self.noised_in = nn.ModuleList(
            [
                _patch_embedder(patch_size, hidden_dim, mlp_kernel=7),  # ABP target
                _patch_embedder(patch_size, hidden_dim, mlp_kernel=3),  # ECG target
                _patch_embedder(patch_size, hidden_dim, mlp_kernel=3),  # PPG target
            ]
        )
        self.cond_in = nn.ModuleDict(
            {
                "ecg": _patch_embedder(patch_size, hidden_dim, mlp_kernel=3),
                "ppg": _patch_embedder(patch_size, hidden_dim, mlp_kernel=3),
            }
        )
        # learned per-stream "absent" fillers; referenced (zero-gated) every forward
        # via the role one-hot, so DDP find_unused_parameters=False stays valid.
        self.absent_token = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(3)]
        )
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        self.t_embed = TimestepEmbedder(hidden_dim, frequency_embedding_size=256, max_period=10000)

        # pre_only=False on every joint block: the per-sample target varies across a
        # batch, so all three streams must stay updated through every joint layer.
        self.joint_blocks = nn.ModuleList(
            [
                BPJointBlock(hidden_dim, num_heads, mlp_ratio, pre_only=False)
                for _ in range(joint_depth)
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
        # per-modality flow-decode heads (the gathered target stream -> its flow).
        self.heads = nn.ModuleList([FinalBlock(hidden_dim, self.latent_dim) for _ in range(3)])
        # optional demographic global-condition encoder (zero-init -> no-op start)
        self.use_demo = use_demo
        if use_demo:
            self.demo_encoder = DemoEncoder(hidden_dim)

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
        # distinct, small per-stream "absent" fillers (not zero → a real code)
        for tok in self.absent_token:
            nn.init.normal_(tok, std=0.02)

        def _zero_adaln(blk: MMDitSingleBlock) -> None:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)

        for jb in self.joint_blocks:
            _zero_adaln(jb.latent_block)
            _zero_adaln(jb.ecg_block)
            _zero_adaln(jb.ppg_block)
        for fb in self.fused_blocks:
            _zero_adaln(fb)
        # zero-init every decode head so the model starts as a no-op (flow ≈ 0).
        for head in self.heads:
            nn.init.constant_(head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(head.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(head.conv.weight, 0)
            nn.init.constant_(head.conv.bias, 0)
        if self.use_demo:
            # zero-init the encoder output so demographics start as a no-op
            nn.init.constant_(self.demo_encoder.mlp[-1].weight, 0)
            nn.init.constant_(self.demo_encoder.mlp[-1].bias, 0)

    def _demo_emb(self, demo: Optional[DemoInput]) -> Optional[torch.Tensor]:
        """Encode a demographics sample, or None when demo conditioning is off."""
        if not self.use_demo or demo is None:
            return None
        cont, gender = demo
        return self.demo_encoder(cont, gender)  # (B, hidden)

    def preprocess_conditions(
        self,
        ecg_clean: torch.Tensor,
        ppg_clean: torch.Tensor,
        target_idx: torch.Tensor,
        cond_present: torch.Tensor,
        demo: Optional[DemoInput] = None,
    ) -> BPConditions:
        """Precompute the per-task pieces constant across ODE steps.

        ``ecg_clean`` / ``ppg_clean`` (B, N, P) are the recentered clean condition
        patches (always passed; gated to 0 when that modality is the target/absent,
        so a clean target never leaks). ``target_idx`` (B,) in {0,1,2}, and
        ``cond_present`` (B, 3) marks which streams condition the model.
        """
        b = ecg_clean.shape[0]
        dev, dt = ecg_clean.device, ecg_clean.dtype
        sidx = torch.arange(3, device=dev).view(1, 3)
        target_idx = target_idx.to(dev).long()
        is_target = (sidx == target_idx.view(b, 1)).to(dt)  # (B, 3)
        cp = cond_present.to(dev, dt)
        is_cond = cp * (1.0 - is_target)  # present AND not the target -> condition
        is_absent = (1.0 - is_target) * (1.0 - cp)  # neither target nor condition
        # condition embeddings (ECG / PPG only; ABP is never a condition)
        ecg_c = self.cond_in["ecg"](ecg_clean)  # (B, N, H)
        ppg_c = self.cond_in["ppg"](ppg_clean)  # (B, N, H)

        def base(s: int, cond_emb: Optional[torch.Tensor]) -> torch.Tensor:
            out = is_absent[:, s].view(b, 1, 1) * self.absent_token[s].view(1, 1, -1)
            if cond_emb is not None:
                out = out + is_cond[:, s].view(b, 1, 1) * cond_emb
            return out  # (B, 1or N, H), broadcasts over tokens when added to noised

        stream_base = (base(0, None), base(1, ecg_c), base(2, ppg_c))
        # global condition pooled over present conditions only (>=1 by construction)
        cond_sum = is_cond[:, 1].view(b, 1) * ecg_c.mean(dim=1) + is_cond[:, 2].view(b, 1) * ppg_c.mean(dim=1)
        num_cond = is_cond[:, 1:].sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = self.global_cond_mlp(cond_sum / num_cond)
        attn_mask = build_task_mask(target_idx, cp, self.latent_seq_len, dtype=dt, device=dev)
        return BPConditions(
            stream_base=stream_base,
            is_target=is_target,
            target_idx=target_idx,
            pooled=pooled,
            attn_mask=attn_mask,
            demo_emb=self._demo_emb(demo),
        )

    def predict_flow(
        self, noised_target: torch.Tensor, t: torch.Tensor, conditions: BPConditions
    ) -> torch.Tensor:
        b = noised_target.shape[0]
        it = conditions.is_target
        # 3 stream embeddings = cached (cond/absent) base + this-step's noised-target
        # embedding, gated by the role one-hot. noised_in[s] is applied for every s
        # (gated) so all embedders stay in the autograd graph (DDP FUP=False).
        streams = [
            conditions.stream_base[s] + it[:, s].view(b, 1, 1) * self.noised_in[s](noised_target)
            for s in range(3)
        ]
        latent, ecg, ppg = streams
        global_c = self.t_embed(t).unsqueeze(1) + conditions.pooled.unsqueeze(1)  # (B,1,H)
        if conditions.demo_emb is not None:
            global_c = global_c + conditions.demo_emb.unsqueeze(1)
        for block in self.joint_blocks:
            latent, ecg, ppg = block(latent, ecg, ppg, global_c, self.latent_rot, conditions.attn_mask)
        # gather the per-sample target stream, refine, decode with its head
        stacked = torch.stack([latent, ecg, ppg], dim=1)  # (B, 3, N, H)
        n, h = stacked.shape[2], stacked.shape[3]
        gi = conditions.target_idx.view(b, 1, 1, 1)
        tgt = stacked.gather(1, gi.expand(b, 1, n, h)).squeeze(1)  # (B, N, H)
        for block in self.fused_blocks:
            tgt = block(tgt, global_c, self.latent_rot)
        # apply every head (keeps all in the graph), then gather the target's output
        outs = torch.stack([head(tgt, global_c) for head in self.heads], dim=1)  # (B,3,N,P)
        p = outs.shape[3]
        return outs.gather(1, gi.expand(b, 1, n, p)).squeeze(1)  # (B, N, P)

    def forward(
        self,
        noised_target: torch.Tensor,
        ecg_clean: torch.Tensor,
        ppg_clean: torch.Tensor,
        t: torch.Tensor,
        target_idx: torch.Tensor,
        cond_present: torch.Tensor,
        demo: Optional[DemoInput] = None,
    ) -> torch.Tensor:
        return self.predict_flow(
            noised_target,
            t,
            self.preprocess_conditions(ecg_clean, ppg_clean, target_idx, cond_present, demo),
        )

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
