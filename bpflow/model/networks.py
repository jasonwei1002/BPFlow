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


@dataclass
class BPConditions:
    """Cached condition tensors that do not depend on the latent or timestep."""

    ecg_seq: torch.Tensor  # (B, N, hidden)
    ppg_seq: torch.Tensor  # (B, N, hidden)
    pooled: torch.Tensor  # (B, hidden)


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
        # learned null conditions for optional classifier-free guidance
        self.empty_ecg = nn.Parameter(torch.zeros(1, 1, patch_size))
        self.empty_ppg = nn.Parameter(torch.zeros(1, 1, patch_size))

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
        nn.init.constant_(self.empty_ecg, 0)
        nn.init.constant_(self.empty_ppg, 0)

    def _embed_cond(self, ecg_p: torch.Tensor, ppg_p: torch.Tensor) -> BPConditions:
        ecg_seq = self.ecg_input_proj(ecg_p)  # (B, N, H)
        ppg_seq = self.ppg_input_proj(ppg_p)  # (B, N, H)
        pooled = self.global_cond_mlp((ecg_seq + ppg_seq).mean(dim=1))  # (B, H)
        return BPConditions(ecg_seq=ecg_seq, ppg_seq=ppg_seq, pooled=pooled)

    def preprocess_conditions(self, cond_patches: torch.Tensor) -> BPConditions:
        # cond_patches (B,N,2P) is channel-major [ECG(P), PPG(P)] -> split back.
        p = self.patch_size
        return self._embed_cond(cond_patches[..., :p], cond_patches[..., p:])

    def predict_flow(
        self, latent_patches: torch.Tensor, t: torch.Tensor, conditions: BPConditions
    ) -> torch.Tensor:
        latent = self.abp_input_proj(latent_patches)  # (B, N, H)
        global_c = self.t_embed(t).unsqueeze(1) + conditions.pooled.unsqueeze(1)  # (B,1,H)
        ecg, ppg = conditions.ecg_seq, conditions.ppg_seq
        for block in self.joint_blocks:
            latent, ecg, ppg = block(latent, ecg, ppg, global_c, self.latent_rot)
        for block in self.fused_blocks:
            latent = block(latent, global_c, self.latent_rot)
        return self.final_layer(latent, global_c)  # (B, N, P)

    def forward(
        self, latent_patches: torch.Tensor, cond_patches: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_flow(latent_patches, t, self.preprocess_conditions(cond_patches))

    def get_empty_conditions(self, bs: int) -> BPConditions:
        ecg_p = self.empty_ecg.expand(bs, self.latent_seq_len, -1)
        ppg_p = self.empty_ppg.expand(bs, self.latent_seq_len, -1)
        return self._embed_cond(ecg_p, ppg_p)

    def ode_wrapper(
        self,
        t: torch.Tensor,
        latent: torch.Tensor,
        conditions: BPConditions,
        empty_conditions: BPConditions,
        cfg_strength: float,
    ) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if cfg_strength > 1.0:
            return cfg_strength * self.predict_flow(latent, t, conditions) + (
                1 - cfg_strength
            ) * self.predict_flow(latent, t, empty_conditions)
        return self.predict_flow(latent, t, conditions)

    @property
    def device(self) -> torch.device:
        return self.latent_rot.device

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
