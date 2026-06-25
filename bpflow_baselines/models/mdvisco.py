"""MD-ViSCo stage-1 baseline (UNet + Swin Transformer bottleneck + AdaIN).

Self-contained port of MD-ViSCo's stage-1 approximation network
``UNetSwinUnet`` (1D U-Net encoder/decoder with a Swin Transformer bottleneck
modulated by Adaptive Instance Normalization). The Hydra / ``SingleStageModel``
plumbing and the stage-2 refinement stack (BPModel / VitalEncoder /
PatchTSMixer / DistilBert) are dropped; only the waveform generator is kept.

The AdaIN style vector ``s`` originally came from a per-sample ``tgt_idxs``
one-hot. Here each baseline serves a single fixed direction, so the factory
bakes a constant one-hot style (target modality over [abp, ecg, ppg]) and the
forward signature becomes ``forward(x, want_bp=False)`` matching the project
baseline contract.

For ``*2abp`` directions a small convolutional SBP/DBP regression head is added
(stage-2 stand-in); other directions are stage-1 only.

Architecture reference: MD-ViSCo, IEEE J. Biomed. Health Inform. (2026),
DOI 10.1109/JBHI.2025.3639315; code https://github.com/fr-meyer/MD-ViSCo (MIT).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

from .base import (
    BaselineModule,
    next_multiple,
    register_model,
)
from ..norms import num_source_channels

logger = logging.getLogger(__name__)

# Style index per modality, matching the project channel order [abp, ecg, ppg].
_MODALITY_TO_STYLE: Dict[str, int] = {"abp": 0, "ecg": 1, "ppg": 2}


# ---------------------------------------------------------------------------
# AdaIN building blocks
# ---------------------------------------------------------------------------
class AdaIN(nn.Module):
    """Adaptive Instance Normalization driven by a style vector."""

    def __init__(self, style_dim: int, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Modulate ``x`` (B, C, L) with style ``s`` (B, style_dim)."""
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    """Residual block with AdaIN normalization and optional dual-path upsampling."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        k: int = 3,
        style_dim: int = 64,
        actv: Optional[nn.Module] = None,
        upsample: bool = False,
        upsample_scale: int = 2,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.k = k
        self.actv = (
            actv if actv is not None else nn.LeakyReLU(leaky_relu_negative_slope)
        )
        self.upsample = upsample
        self.upsample_scale = upsample_scale
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int) -> None:
        if self.upsample:
            self.conv1 = nn.Conv1d(dim_in * 2, dim_out, self.k, 1, self.k // 2)
        else:
            self.conv1 = nn.Conv1d(dim_in, dim_out, self.k, 1, self.k // 2)
        self.conv2 = nn.Conv1d(dim_out, dim_out, self.k, 1, self.k // 2)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.upsample:
            self.transpose_residual = nn.ConvTranspose1d(
                in_channels=dim_in,
                out_channels=dim_in,
                kernel_size=self.upsample_scale,
                stride=self.upsample_scale,
                padding=0,
                output_padding=0,
                bias=False,
            )

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x_up = F.interpolate(x, scale_factor=self.upsample_scale, mode="nearest")
            x_trans = self.transpose_residual(x)
            x = torch.cat([x_up, x_trans], dim=1)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self._residual(x, s)


# ---------------------------------------------------------------------------
# Swin Transformer primitives
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    """Two-layer MLP used inside Swin blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split (B, W, C) into non-overlapping windows (B*nW, window_size, C)."""
    b, w, c = x.shape
    x = x.view(b, w // window_size, window_size, c)
    windows = x.contiguous().view(-1, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, w: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`, reconstructing (B, W, C)."""
    b = int(windows.shape[0] / (w / window_size))
    x = windows.view(b, w // window_size, window_size, -1)
    x = x.contiguous().view(b, w, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads)
        )
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """One Swin Transformer block (W-MSA / SW-MSA + MLP) with optional AdaIN norm."""

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.Module,
        style_dim: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.style_dim = style_dim

        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution

        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        if style_dim is not None:
            self.norm1 = norm_layer(style_dim, dim)
            self.norm2 = norm_layer(style_dim, dim)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            w = self.input_resolution
            img_mask = torch.zeros((1, w, 1))
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            for cnt, w_slice in enumerate(w_slices):
                img_mask[:, w_slice, :] = cnt
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, -100.0
            ).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def _norm(
        self, norm: nn.Module, x: torch.Tensor, s: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = norm(x, s) if self.style_dim is not None else norm(x)
        return x.transpose(1, 2)

    def forward(
        self, x: torch.Tensor, s: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w = self.input_resolution
        b, seq_len, c = x.shape
        if seq_len != w:
            raise ValueError("input feature has wrong size")

        shortcut = x
        x = self._norm(self.norm1, x, s)
        x = x.view(b, w, c)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=1)
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
        x = x.view(b, w, c)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self._norm(self.norm2, x, s)))
        return x


class PatchMerging(nn.Module):
    """Downsampling: merge adjacent tokens, halve length, keep channel ramp."""

    def __init__(
        self,
        input_resolution: int,
        dim: int,
        norm_layer: type = nn.LayerNorm,
        style_dim: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        norm_style_dim = style_dim if style_dim is not None else 64
        # NOTE: faithful to MD-ViSCo upstream (src/model/mdvisco.py:1823). The 2-arg
        # form (style_dim, features) targets the AdaIN norm; when norm_layer is the
        # encoder's nn.InstanceNorm1d, the 2nd arg lands in `eps` (an upstream quirk).
        # Kept verbatim so the baseline reproduces MD-ViSCo's actual behavior.
        self.norm = norm_layer(norm_style_dim, 2 * dim)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        w = self.input_resolution
        b, seq_len, c = x.shape
        if seq_len != w:
            raise ValueError("input feature has wrong size")
        if w % 2 != 0:
            raise ValueError(f"x size ({w}) is not even")
        x = x.view(b, w, c)
        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], -1)
        x = x.view(b, -1, 2 * c)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    """Upsampling: double token count, halve channels (AdaIN normalized)."""

    def __init__(
        self,
        input_resolution: int,
        dim: int,
        dim_scale: int = 2,
        norm_layer: type = nn.LayerNorm,
        style_dim: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = (
            nn.Linear(dim, dim // dim_scale, bias=False)
            if dim_scale >= 2
            else nn.Identity()
        )
        norm_style_dim = style_dim if style_dim is not None else 64
        self.norm = norm_layer(norm_style_dim, dim // (dim_scale * 2))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        w = self.input_resolution
        x = self.expand(x)
        b, seq_len, c = x.shape
        if seq_len != w:
            raise ValueError("input feature has wrong size")
        x = x.view(b, w, c)
        x = rearrange(x, "b w (p1 c)-> b (w p1) c", p1=2, c=c // 2)
        x = x.view(b, -1, c // 2)
        x = self.norm(x.transpose(1, 2), s).transpose(1, 2)
        return x


class FinalPatchExpandX4(nn.Module):
    """Final upsampling that expands tokens by ``dim_scale`` (== patch_size)."""

    def __init__(
        self,
        input_resolution: int,
        dim: int,
        dim_scale: int = 4,
        norm_layer: type = nn.Module,
        style_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim // dim_scale
        self.norm = norm_layer(style_dim, self.output_dim)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        w = self.input_resolution
        x = self.expand(x)
        b, seq_len, c = x.shape
        if seq_len != w:
            raise ValueError("input feature has wrong size")
        x = x.view(b, w, c)
        x = rearrange(
            x, "b w (p1 c)-> b (w p1) c", p1=self.dim_scale, c=c // self.dim_scale
        )
        # NOTE: faithful to MD-ViSCo upstream (src/model/mdvisco.py:2064), which
        # also uses .view() here (not transpose). Kept verbatim for reproduction.
        x = x.view(b, self.output_dim, -1)
        x = self.norm(x, s)
        return x


class BasicLayer(nn.Module):
    """One Swin encoder stage: stacked blocks + optional patch merging."""

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Any = 0.0,
        norm_layer: type = nn.Module,
        downsample: Optional[type] = None,
        style_dim: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    style_dim=style_dim,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, style_dim=style_dim
            )
        else:
            self.downsample = None

    def forward(
        self, x: torch.Tensor, s: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, s)
        if self.downsample is not None:
            x = self.downsample(x, s)
        return x


class BasicLayerUp(nn.Module):
    """One Swin decoder stage: stacked blocks + optional patch expansion."""

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Any = 0.0,
        norm_layer: type = nn.Module,
        upsample: Optional[bool] = None,
        style_dim: Optional[int] = 64,
        dim_scale: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    style_dim=style_dim,
                )
                for i in range(depth)
            ]
        )
        if upsample is not None:
            self.upsample: Optional[PatchExpand] = PatchExpand(
                input_resolution,
                dim=dim,
                dim_scale=dim_scale,
                norm_layer=norm_layer,
                style_dim=style_dim,
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, s)
        if self.upsample is not None:
            x = self.upsample(x, s)
        return x


class PatchEmbed(nn.Module):
    """Conv patch embedding (B, C, W) -> (B, W/patch, embed_dim)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[type] = None,
        style_dim: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = img_size // patch_size
        self.num_patches = self.patch_resolution
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w = x.shape
        if self.img_size != w:
            raise ValueError(
                f"Input signal length ({w}) doesn't match model ({self.img_size})."
            )
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
            x = x.transpose(1, 2)
        return x


class SwinTransformerSysAdaIn(nn.Module):
    """Swin-Unet bottleneck with AdaIN-conditioned decoder.

    Mirrors MD-ViSCo's ``SwinTransformerSysAdaIn``: a Swin encoder
    (InstanceNorm) feeds a Swin decoder (AdaIN), then a final patch expand
    restores resolution. The number of encoder/decoder stages equals
    ``len(depths)``; for the default single-element ``upsample_scale`` there is
    exactly one stage and no patch merging/expansion.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        depths_decoder: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer_encoder: type = nn.Module,
        norm_layer_decoder: type = nn.Module,
        ape: bool = False,
        patch_norm: bool = True,
        final_upsample: str = "expand_first",
        style_dim: int = 64,
    ) -> None:
        super().__init__()
        depths = [2, 2, 2, 2] if depths is None else list(depths)
        depths_decoder = (
            [1, 2, 2, 2] if depths_decoder is None else list(depths_decoder)
        )
        num_heads = [3, 6, 12, 24] if num_heads is None else list(num_heads)

        self.style_dim = style_dim
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_layers_decoder = len(depths_decoder)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer_encoder if self.patch_norm else None,
            style_dim=style_dim,
        )
        num_patches = self.patch_embed.num_patches
        patch_resolution = self.patch_embed.patch_resolution
        self.patch_resolution = patch_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            float(x.item()) for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path_slice = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])]
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=patch_resolution // (2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=(
                    drop_path_slice if len(drop_path_slice) > 1 else drop_path_slice[0]
                ),
                norm_layer=norm_layer_encoder,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                style_dim=None,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers_decoder):
            drop_path_slice = dpr[
                sum(depths[: (self.num_layers_decoder - 1 - i_layer)]) : sum(
                    depths[: (self.num_layers_decoder - 1 - i_layer) + 1]
                )
            ]
            base_dim = int(embed_dim * 2 ** (self.num_layers_decoder - 1 - i_layer))
            resolution = patch_resolution // (
                2 ** (self.num_layers_decoder - 1 - i_layer)
            )
            do_upsample = True if (i_layer < self.num_layers_decoder - 1) else None
            if i_layer == 0:
                layer_up = BasicLayerUp(
                    dim=base_dim,
                    input_resolution=resolution,
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers_decoder - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_slice
                        if len(drop_path_slice) > 1
                        else drop_path_slice[0]
                    ),
                    norm_layer=norm_layer_decoder,
                    upsample=do_upsample,
                    style_dim=style_dim,
                    dim_scale=1,
                )
            else:
                layer_up = BasicLayerUp(
                    dim=2 * base_dim,
                    input_resolution=resolution,
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers_decoder - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_slice
                        if len(drop_path_slice) > 1
                        else drop_path_slice[0]
                    ),
                    norm_layer=norm_layer_decoder,
                    upsample=do_upsample,
                    style_dim=style_dim,
                )
            self.layers_up.append(layer_up)

        self.norm = norm_layer_encoder(self.num_features)
        if len(self.layers_up) > 1:
            self.norm_up = norm_layer_decoder(self.style_dim, self.embed_dim * 2)
            final_dim = embed_dim * 2
        else:
            self.norm_up = norm_layer_decoder(self.style_dim, self.embed_dim)
            final_dim = embed_dim
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpandX4(
                input_resolution=img_size // patch_size,
                dim_scale=patch_size,
                dim=final_dim,
                norm_layer=AdaIN,
                style_dim=style_dim,
            )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                x_downsample.insert(0, x)
            x = layer(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x, x_downsample

    def forward_up_features(
        self, x: torch.Tensor, x_downsample: List[torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        for inx, layer_up in enumerate(self.layers_up):
            x = layer_up(x, s)
            if inx != len(self.layers_up) - 1:
                x = torch.cat([x, x_downsample[inx]], -1)
        x = self.norm_up(x.transpose(1, 2), s).transpose(1, 2)
        return x

    def up_x4(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        w = self.patch_resolution
        b, seq_len, c = x.shape
        if seq_len != w:
            raise ValueError("input features has wrong size")
        if self.final_upsample == "expand_first":
            x = self.up(x, s)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample, s)
        x = self.up_x4(x, s)
        return x


# ---------------------------------------------------------------------------
# U-Net + Swin generator (stage 1)
# ---------------------------------------------------------------------------
class UNetSwinUnet(nn.Module):
    """1D U-Net with a Swin Transformer bottleneck and AdaIN-styled decoder.

    Faithful to MD-ViSCo's stage-1 generator, with the batch-dict / Hydra
    plumbing removed: forward takes a raw waveform ``x`` (B, C, L) and a fixed
    style vector ``s`` (B, style_dim).
    """

    def __init__(
        self,
        input_length: int,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        kernel_size: int = 3,
        style_dim: int = 3,
        upsample_scale: Optional[List[int]] = None,
        patch_size: int = 4,
        depth: int = 1,
        embedding_dim_multiplier: int = 4,
        swin_num_heads: Optional[List[int]] = None,
        swin_mlp_ratio: float = 4.0,
        swin_drop_rate: float = 0.0,
        swin_attn_drop_rate: float = 0.0,
        swin_drop_path_rate: float = 0.1,
        leaky_relu_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if upsample_scale is None:
            upsample_scale = [4]
        if swin_num_heads is None:
            swin_num_heads = [32, 32, 32, 32, 32]

        self.input_length = input_length
        features = init_features
        self.features = init_features
        self.conv_init_features = nn.Conv1d(in_channels, features, 3, 1, 1)

        self.encoder = nn.ModuleList()
        self.depth = depth
        w_size = input_length
        for i in range(depth):
            self.encoder.append(
                UNetSwinUnet._block(
                    features, features, k=kernel_size, name=f"enc{i}_1"
                )
            )
            self.encoder.append(
                nn.Conv1d(features, features, kernel_size=2, stride=2)
            )
            self.encoder.append(
                UNetSwinUnet._block(
                    features * 2, features * 2, k=kernel_size, name=f"enc{i}_2"
                )
            )
            features = features * 2
            w_size = w_size // 2

        in_chans = features
        embed_dim = in_chans * embedding_dim_multiplier
        window_size = patch_size

        self.bottleneck = SwinTransformerSysAdaIn(
            img_size=w_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=embed_dim,
            embed_dim=embed_dim,
            depths=upsample_scale,
            depths_decoder=upsample_scale,
            num_heads=swin_num_heads,
            window_size=window_size,
            mlp_ratio=swin_mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=swin_drop_rate,
            attn_drop_rate=swin_attn_drop_rate,
            drop_path_rate=swin_drop_path_rate,
            norm_layer_encoder=nn.InstanceNorm1d,
            norm_layer_decoder=AdaIN,
            ape=False,
            patch_norm=True,
            final_upsample="expand_first",
            style_dim=style_dim,
        )

        self.decoder = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.decoder.append(
                    AdainResBlk(
                        dim_in=embed_dim // patch_size,
                        dim_out=features // 2,
                        upsample=True,
                        style_dim=style_dim,
                        upsample_scale=2,
                        k=3,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
                self.decoder.append(
                    AdainResBlk(
                        dim_in=features,
                        dim_out=features // 2,
                        upsample=False,
                        style_dim=style_dim,
                        upsample_scale=2,
                        k=3,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
            else:
                self.decoder.append(
                    AdainResBlk(
                        dim_in=features,
                        dim_out=features // 2,
                        upsample=True,
                        style_dim=style_dim,
                        upsample_scale=2,
                        k=3,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
                self.decoder.append(
                    AdainResBlk(
                        dim_in=features,
                        dim_out=features // 2,
                        upsample=False,
                        style_dim=style_dim,
                        upsample_scale=2,
                        k=3,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
            features = features // 2

        self.last = nn.Sequential(
            nn.InstanceNorm1d(features, affine=True),
            nn.LeakyReLU(leaky_relu_negative_slope),
            nn.Conv1d(features, out_channels, kernel_size=1, padding=0, bias=False),
        )

    @staticmethod
    def _block(in_channels: int, features: int, k: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(in_channels, features, k, padding=k // 2, bias=False),
                    ),
                    (name + "norm1", nn.InstanceNorm1d(features)),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(features, features, k, padding=k // 2, bias=False),
                    ),
                    (name + "norm2", nn.InstanceNorm1d(features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        enc = self.conv_init_features(x)
        encoder: List[torch.Tensor] = []
        for i in range(0, len(self.encoder), 3):
            encoder.insert(0, enc)
            enc = self.encoder[i](enc)
            x_down = F.max_pool1d(enc, 2)
            x_conv_pool = self.encoder[i + 1](enc)
            enc = torch.cat([x_down, x_conv_pool], dim=1)
            enc = self.encoder[i + 2](enc)

        bottleneck = self.bottleneck(enc, s)

        dec = bottleneck
        for idx, i in enumerate(range(0, len(self.decoder), 2)):
            dec = self.decoder[i](dec, s)
            if idx < len(encoder):
                dec = torch.cat([dec, encoder[idx]], dim=1)
            dec = self.decoder[i + 1](dec, s)
        out = self.last(dec)
        return out


# ---------------------------------------------------------------------------
# BP head (stage-2 stand-in for ->ABP directions)
# ---------------------------------------------------------------------------
class BPHead(nn.Module):
    """Small conv encoder + global pool + MLP mapping a waveform to (SBP, DBP).

    Output is linear (no activation); trained in global-min-max [0,1] (via bp_l1),
    de-normalized to mmHg in reconstruct.py — NOT raw mmHg.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(hidden * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).squeeze(-1)
        return self.mlp(h)


# ---------------------------------------------------------------------------
# Baseline wrapper
# ---------------------------------------------------------------------------
class MDViSCoBaseline(BaselineModule):
    """MD-ViSCo stage-1 generator wrapped to the project baseline contract."""

    def __init__(
        self,
        input_length: int,
        work_multiple: int,
        style_index: int,
        style_dim: int,
        has_bp_head: bool,
        unet_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.work_multiple = work_multiple
        self.has_bp_head = has_bp_head
        self.style_dim = style_dim

        self.generator = UNetSwinUnet(
            input_length=input_length, style_dim=style_dim, **unet_kwargs
        )
        # Constant one-hot style for this single-direction model (buffer: moves
        # with .to(device) and is excluded from optimizer / weight decay).
        style = torch.zeros(1, style_dim)
        style[0, style_index] = 1.0
        self.register_buffer("style", style)

        # forward() feeds the raw source x (B, C, L) — not the generated wave —
        # to bp_head, so it must match the UNet's input channel count.
        bp_in = int(unet_kwargs.get("in_channels", 1))
        self.bp_head = BPHead(in_channels=bp_in) if has_bp_head else None

    def forward(
        self, x: torch.Tensor, want_bp: bool = False
    ) -> Dict[str, object]:
        s = self.style.to(dtype=x.dtype).expand(x.size(0), self.style_dim)
        wave = self.generator(x, s)

        bp: Optional[torch.Tensor] = None
        if want_bp and self.bp_head is not None:
            bp = self.bp_head(x)

        return {"wave": wave, "wave_aux": [], "bp": bp}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _swin_work_multiple(patch_size: int, depth: int, num_swin_layers: int) -> int:
    """Smallest length unit so every internal length stays integral & divisible.

    The waveform length must (a) survive ``depth`` U-Net halvings, (b) be
    divisible by ``patch_size`` at the patch embedding, and (c) keep each Swin
    stage's token count divisible by ``window_size`` (== ``patch_size``) after
    ``num_swin_layers - 1`` patch-merging halvings.
    """
    return patch_size * (2**depth) * patch_size * (2 ** (num_swin_layers - 1))


@register_model("mdvisco")
def factory(params: dict, seq_len: int, direction: str) -> BaselineModule:
    """Build the MD-ViSCo stage-1 baseline for a single conversion direction.

    Args:
        params: Structural hyperparameters (see defaults below).
        seq_len: True (unpadded) signal length.
        direction: e.g. ``"ecg2abp"`` / ``"ppg2ecg"``; ``*2abp`` enables the
            BP head.

    Returns:
        A configured :class:`MDViSCoBaseline`.
    """
    p = dict(params or {})

    init_features = int(p.get("init_features", 64))
    patch_size = int(p.get("patch_size", 4))
    depth = int(p.get("depth", 1))
    embedding_dim_multiplier = int(p.get("embedding_dim_multiplier", 4))
    swin_num_heads = list(p.get("swin_num_heads", [32, 32, 32, 32, 32]))
    swin_mlp_ratio = float(p.get("swin_mlp_ratio", 4.0))
    swin_drop_path_rate = float(p.get("swin_drop_path_rate", 0.1))
    kernel_size = int(p.get("kernel_size", 3))
    style_dim = int(p.get("style_dim", 3))
    upsample_scale = list(p.get("upsample_scale", [4]))

    num_swin_layers = len(upsample_scale)
    if len(swin_num_heads) < num_swin_layers:
        raise ValueError(
            "swin_num_heads must have at least len(upsample_scale) entries"
        )

    work_multiple = _swin_work_multiple(patch_size, depth, num_swin_layers)
    work_length = next_multiple(seq_len, work_multiple)

    # Target modality drives the fixed AdaIN style (abp=0, ecg=1, ppg=2).
    target = direction.split("2")[-1]
    style_index = _MODALITY_TO_STYLE.get(target, 0)
    if style_index >= style_dim:
        raise ValueError(
            f"style_index {style_index} out of range for style_dim {style_dim}"
        )

    tgt_is_abp = direction.endswith("2abp")

    unet_kwargs: Dict[str, Any] = {
        "in_channels": int(p.get("in_channels", num_source_channels(direction))),
        "out_channels": 1,
        "init_features": init_features,
        "kernel_size": kernel_size,
        "upsample_scale": upsample_scale,
        "patch_size": patch_size,
        "depth": depth,
        "embedding_dim_multiplier": embedding_dim_multiplier,
        "swin_num_heads": swin_num_heads,
        "swin_mlp_ratio": swin_mlp_ratio,
        "swin_drop_path_rate": swin_drop_path_rate,
    }

    return MDViSCoBaseline(
        input_length=work_length,
        work_multiple=work_multiple,
        style_index=style_index,
        style_dim=style_dim,
        has_bp_head=tgt_is_abp,
        unet_kwargs=unet_kwargs,
    )
