"""PPG2ABP cascade baseline: UNetDS64 (stage 1, deep supervision) -> MultiResUNet1D (stage 2).

Ported from MD-ViSCo/src/model/ppg2abp.py, stripped of Hydra / SingleStageModel /
batch_dict conventions. Self-contained; no runtime dependency on MD-ViSCo.

References:
    "PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood
    Pressure (ABP) Waveforms using Fully Convolutional Neural Networks"
    Ibtehaz et al., Bioengineering 2022. https://www.mdpi.com/2306-5354/9/11/692
    Original implementation: https://github.com/nibtehaz/PPG2ABP (MIT)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BaselineModule, register_model
from ..norms import num_source_channels


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------

def _conv_bn(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    activation: bool = True,
) -> nn.Sequential:
    """Conv1d -> BatchNorm1d -> optional ReLU."""
    padding = kernel_size // 2
    layers: list[nn.Module] = [
        nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
        nn.BatchNorm1d(out_ch),
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _UNetDS64Core(nn.Module):
    """Deeply supervised U-Net (all-in-one, no extra base class).

    Outputs: (out, level1, level2, level3, level4) each (B, 1, L_i).
    level4 is the bottleneck deep-supervision output (smallest resolution).
    out is the full-resolution prediction.
    """

    def __init__(self, base_channels: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        x = base_channels

        # Encoder
        self.conv1 = nn.Sequential(_conv_bn(in_channels, x), _conv_bn(x, x))
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Sequential(_conv_bn(x, x * 2), _conv_bn(x * 2, x * 2))
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Sequential(_conv_bn(x * 2, x * 4), _conv_bn(x * 4, x * 4))
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Sequential(_conv_bn(x * 4, x * 8), _conv_bn(x * 8, x * 8))
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Sequential(_conv_bn(x * 8, x * 16), _conv_bn(x * 16, x * 16))

        # Deep supervision at bottleneck (level4 = smallest spatial)
        self.level4 = nn.Conv1d(x * 16, 1, 1)

        # Decoder + intermediate deep supervision heads
        self.up6 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv6 = nn.Sequential(
            _conv_bn(x * 16 + x * 8, x * 8), _conv_bn(x * 8, x * 8)
        )
        self.level3 = nn.Conv1d(x * 8, 1, 1)

        self.up7 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv7 = nn.Sequential(
            _conv_bn(x * 8 + x * 4, x * 4), _conv_bn(x * 4, x * 4)
        )
        self.level2 = nn.Conv1d(x * 4, 1, 1)

        self.up8 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv8 = nn.Sequential(
            _conv_bn(x * 4 + x * 2, x * 2), _conv_bn(x * 2, x * 2)
        )
        self.level1 = nn.Conv1d(x * 2, 1, 1)

        self.up9 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv9 = nn.Sequential(_conv_bn(x * 2 + x, x), _conv_bn(x, x))
        self.out = nn.Conv1d(x, 1, 1)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        """Return (out, l1, l2, l3, l4) deep-supervision tuple."""
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))

        l4 = self.level4(c5)  # smallest resolution DS output

        # Decoder
        d6 = self.conv6(torch.cat([self.up6(c5), c4], dim=1))
        l3 = self.level3(d6)

        d7 = self.conv7(torch.cat([self.up7(d6), c3], dim=1))
        l2 = self.level2(d7)

        d8 = self.conv8(torch.cat([self.up8(d7), c2], dim=1))
        l1 = self.level1(d8)

        d9 = self.conv9(torch.cat([self.up9(d8), c1], dim=1))
        out = self.out(d9)

        return out, l1, l2, l3, l4


# ---------------------------------------------------------------------------
# MultiResUNet1D building blocks
# ---------------------------------------------------------------------------

class _MultiResBlock(nn.Module):
    """Multi-resolution block following PPG2ABP paper (Section 3.3).

    Computes 3+5+7 parallel convolutions, concatenates, adds shortcut, then
    projects down to ``u`` channels for the subsequent ResPath.
    """

    def __init__(self, u: int, in_channels: int, alpha: float) -> None:
        super().__init__()
        w = int(alpha * u)
        out_3 = int(w * 0.167)
        out_5 = int(w * 0.333)
        out_7 = w - out_3 - out_5  # remainder so concat_ch == w exactly
        concat_ch = out_3 + out_5 + out_7

        self.shortcut = _conv_bn(in_channels, concat_ch, kernel_size=1, activation=False)
        self.conv3 = _conv_bn(in_channels, out_3, kernel_size=3)
        # conv5x5 emulated as two 3x3
        self.conv5 = nn.Sequential(
            _conv_bn(out_3, out_5, kernel_size=3),
            _conv_bn(out_5, out_5, kernel_size=3),
        )
        # conv7x7 emulated as three 3x3
        self.conv7 = nn.Sequential(
            _conv_bn(out_5, out_7, kernel_size=3),
            _conv_bn(out_7, out_7, kernel_size=3),
            _conv_bn(out_7, out_7, kernel_size=3),
        )
        self.bn_concat = nn.BatchNorm1d(concat_ch)
        self.final_bn = nn.BatchNorm1d(concat_ch)
        # Project concatenated features back to u channels for ResPath input
        self.proj = nn.Conv1d(concat_ch, u, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        sc = self.shortcut(x)
        c3 = self.conv3(x)
        c5 = self.conv5(c3)
        c7 = self.conv7(c5)
        cat = torch.cat([c3, c5, c7], dim=1)
        cat = self.bn_concat(cat)
        out = torch.relu(cat + sc)
        out = self.final_bn(out)
        return self.proj(out)


class _ResPath(nn.Module):
    """Residual path to bridge encoder and decoder skip connections."""

    def __init__(self, filters: int, length: int) -> None:
        super().__init__()
        blocks: list[nn.ModuleDict] = []
        for _ in range(length):
            blocks.append(
                nn.ModuleDict(
                    {
                        "shortcut": _conv_bn(filters, filters, kernel_size=1, activation=False),
                        "conv": _conv_bn(filters, filters, kernel_size=3),
                        "bn": nn.BatchNorm1d(filters),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = x
        for blk in self.blocks:
            sc = blk["shortcut"](out)
            out = blk["conv"](out)
            out = torch.relu(out + sc)
            out = blk["bn"](out)
        return out


class _MultiResUNet1DCore(nn.Module):
    """1D MultiResUNet (single-output waveform regressor).

    Channel widths match the original implementation: fixed U values
    [32, 64, 128, 256, 512] regardless of alpha (alpha only controls the
    internal width of multi-resolution blocks).
    """

    def __init__(self, alpha: float = 2.5, in_channels: int = 1) -> None:
        super().__init__()
        a = alpha

        # Encoder
        self.mres1 = _MultiResBlock(32, in_channels, a)
        self.pool1 = nn.MaxPool1d(2)
        self.path1 = _ResPath(32, 4)

        self.mres2 = _MultiResBlock(64, 32, a)
        self.pool2 = nn.MaxPool1d(2)
        self.path2 = _ResPath(64, 3)

        self.mres3 = _MultiResBlock(128, 64, a)
        self.pool3 = nn.MaxPool1d(2)
        self.path3 = _ResPath(128, 2)

        self.mres4 = _MultiResBlock(256, 128, a)
        self.pool4 = nn.MaxPool1d(2)
        self.path4 = _ResPath(256, 1)

        self.mres5 = _MultiResBlock(512, 256, a)

        # Decoder — upsample then cat, then BN, then mres block
        self.up6_bn = nn.BatchNorm1d(512 + 256)
        self.mres6 = _MultiResBlock(256, 512 + 256, a)

        self.up7_bn = nn.BatchNorm1d(256 + 128)
        self.mres7 = _MultiResBlock(128, 256 + 128, a)

        self.up8_bn = nn.BatchNorm1d(128 + 64)
        self.mres8 = _MultiResBlock(64, 128 + 64, a)

        self.up9_bn = nn.BatchNorm1d(64 + 32)
        self.mres9 = _MultiResBlock(32, 64 + 32, a)

        self.out_conv = nn.Conv1d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Encoder
        m1 = self.mres1(x)
        m1_skip = self.path1(m1)

        m2 = self.mres2(self.pool1(m1))
        m2_skip = self.path2(m2)

        m3 = self.mres3(self.pool2(m2))
        m3_skip = self.path3(m3)

        m4 = self.mres4(self.pool3(m3))
        m4_skip = self.path4(m4)

        m5 = self.mres5(self.pool4(m4))

        # Decoder — upsample + cat + BN + mres
        up6 = nn.functional.interpolate(m5, scale_factor=2, mode="nearest")
        m6 = self.mres6(self.up6_bn(torch.cat([up6, m4_skip], dim=1)))

        up7 = nn.functional.interpolate(m6, scale_factor=2, mode="nearest")
        m7 = self.mres7(self.up7_bn(torch.cat([up7, m3_skip], dim=1)))

        up8 = nn.functional.interpolate(m7, scale_factor=2, mode="nearest")
        m8 = self.mres8(self.up8_bn(torch.cat([up8, m2_skip], dim=1)))

        up9 = nn.functional.interpolate(m8, scale_factor=2, mode="nearest")
        m9 = self.mres9(self.up9_bn(torch.cat([up9, m1_skip], dim=1)))

        return self.out_conv(m9)


# ---------------------------------------------------------------------------
# PPG2ABP cascade module — implements BaselineModule contract
# ---------------------------------------------------------------------------

class PPG2ABPModel(BaselineModule):
    """Cascade UNetDS64 (stage 1, deep supervision) -> MultiResUNet1D (stage 2).

    Stage 1 (UNetDS64) receives raw PPG; its full-resolution output feeds stage 2.
    Stage 2 (MultiResUNet1D) refines the stage-1 output to the final ABP waveform.

    ``wave``     = stage-2 final output (B, 1, Lw).
    ``wave_aux`` = the five deep-supervision outputs of stage 1:
                   [out, l1, l2, l3, l4] each (B, 1, Lw_i).
                   l4 is the smallest-resolution DS output; out is full-resolution.
    ``bp``       = None (this cascade outputs waveforms; no BP regression head).

    Attributes:
        work_multiple: 16 — four max-pool(2) levels require divisibility by 16.
        has_bp_head: False — cascade is waveform-only.
    """

    work_multiple: int = 16
    has_bp_head: bool = False

    def __init__(self, base_channels: int = 64, alpha: float = 2.5,
                 in_channels: int = 1) -> None:
        """Initialise both stages.

        Args:
            base_channels: Base channel count for UNetDS64 (default 64, paper value).
            alpha: Weight multiplier for MultiResUNet1D blocks (default 2.5, paper value).
            in_channels: Stacked source channels for stage 1 (2 for ecg_ppg2abp).
                Stage 2 ALWAYS stays 1-channel: it refines stage 1's 1-channel
                output, not the raw source — so the cascade only changes here.
        """
        super().__init__()
        self.stage1 = _UNetDS64Core(base_channels=base_channels, in_channels=in_channels)
        # Stage 2 takes the full-resolution output of stage 1: always 1 channel.
        self.stage2 = _MultiResUNet1DCore(alpha=alpha, in_channels=1)

    def forward(
        self, x: torch.Tensor, want_bp: bool = False
    ) -> Dict[str, object]:
        """Forward pass of the PPG2ABP cascade.

        Args:
            x: Input tensor (B, C, Lw) (C=1, or 2 for ecg_ppg2abp); Lw a multiple of ``work_multiple``.
            want_bp: Ignored (has_bp_head is False). Included for contract compliance.

        Returns:
            dict with keys:
                ``wave``     — (B, 1, Lw) stage-2 ABP waveform (linear output).
                ``wave_aux`` — list of 5 tensors from stage-1 deep supervision:
                               [out, l1, l2, l3, l4]; lengths vary with pooling depth.
                ``bp``       — None.
        """
        # Stage 1 — deep supervision
        s1_out, s1_l1, s1_l2, s1_l3, s1_l4 = self.stage1(x)
        # Stage 2 — refinement from stage-1 full-resolution output
        wave = self.stage2(s1_out)
        return {
            "wave": wave,
            "wave_aux": [s1_out, s1_l1, s1_l2, s1_l3, s1_l4],
            "bp": None,
        }


# ---------------------------------------------------------------------------
# Factory + registry
# ---------------------------------------------------------------------------

@register_model("ppg2abp")
def factory(params: dict, seq_len: int, direction: str) -> PPG2ABPModel:
    """Build a PPG2ABPModel from a params dict.

    Args:
        params: Hyper-parameter overrides. Supported keys:
            ``base_channels`` (int, default 64) — UNetDS64 base width.
            ``alpha`` (float, default 2.5) — MultiResUNet1D block weight multiplier.
        seq_len: Source sequence length. Rounded up to a multiple of
            ``work_multiple`` (16) before being passed to the model.
        direction: Task direction string (e.g. ``"ppg2abp"``). PPG2ABP has no
            BP head regardless of direction (waveform-only cascade).

    Returns:
        PPG2ABPModel ready for training or inference.
    """
    base_channels: int = int(params.get("base_channels", 64))
    alpha: float = float(params.get("alpha", 2.5))
    in_channels: int = int(params.get("in_channels", num_source_channels(direction)))
    return PPG2ABPModel(base_channels=base_channels, alpha=alpha, in_channels=in_channels)
