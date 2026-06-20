"""3-stream joint-attention block for BPFlow.

The ABP latent, ECG, and PPG token streams each produce Q/K/V (reusing the
verified ``MMDitSingleBlock.pre_attention``), which are concatenated along the
token axis and run through ONE joint self-attention so all three streams attend
to each other. All three share the same RoPE table because they live on the
same length-N time grid (the alignment prior), so same-time tokens get the same
rotary phase. When ``pre_only=True`` the condition streams contribute to the
joint attention but are not themselves updated (used for the final joint layer).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ._wavflow_layers import MMDitSingleBlock
from .attention_mask import masked_attention


class BPJointBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False) -> None:
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=False, kernel_size=3, padding=1
        )
        self.ecg_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=3, padding=1
        )
        self.ppg_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=3, padding=1
        )

    def forward(
        self,
        latent: torch.Tensor,
        ecg: torch.Tensor,
        ppg: torch.Tensor,
        c: torch.Tensor,
        rot: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_qkv, x_mod = self.latent_block.pre_attention(latent, c, rot)
        e_qkv, e_mod = self.ecg_block.pre_attention(ecg, c, rot)
        p_qkv, p_mod = self.ppg_block.pre_attention(ppg, c, rot)

        ll, el = latent.shape[1], ecg.shape[1]
        joint_qkv = [torch.cat([x_qkv[i], e_qkv[i], p_qkv[i]], dim=2) for i in range(3)]
        # attn_mask=None -> identical to the vendored `attention`; a task mask routes
        # condition->target token flow for the multi-target setup.
        attn = masked_attention(*joint_qkv, attn_mask=attn_mask)
        x_out = attn[:, :ll]
        e_out = attn[:, ll : ll + el]
        p_out = attn[:, ll + el :]

        latent = self.latent_block.post_attention(latent, x_out, x_mod)
        if not self.pre_only:
            ecg = self.ecg_block.post_attention(ecg, e_out, e_mod)
            ppg = self.ppg_block.post_attention(ppg, p_out, p_mod)
        return latent, ecg, ppg
