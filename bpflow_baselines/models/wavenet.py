"""WaveNet baseline — self-contained port from MD-ViSCo.

Faithful port of WaveNetModel (dilated causal convolution stack, gated
activation, residual + skip) stripped of Hydra, SingleStageModel, and
batch_dict routing so it runs purely as an nn.Module.

References:
    - van den Oord et al., "WaveNet: A Generative Model for Raw Audio",
      arXiv:1609.03499, 2016, Sections 2.1–2.4.
    - Original pytorch implementation: github.com/vincentherrmann/pytorch-wavenet
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaselineModule, register_model, next_multiple
from ..norms import num_source_channels

__all__ = ["WaveNet"]


# ---------------------------------------------------------------------------
# Core dilated-conv stack (architecture-faithful to MD-ViSCo WaveNetModel)
# ---------------------------------------------------------------------------

class _WaveNetCore(nn.Module):
    """Dilated causal convolution stack.

    Accepts (B, C_in, L) and returns (B, classes, L) — same spatial length as
    input, achieved via manual left-side causal padding.

    Args:
        layers: Number of dilation layers per block.
        blocks: Number of blocks (dilation pattern repeats).
        dilation_channels: Width of filter/gate convolution outputs.
        residual_channels: Width of the residual stream.
        skip_channels: Width of skip accumulator.
        end_channels: Hidden width of the two post-stack 1×1 convs.
        classes: Output channel width (1 for regression).
        kernel_size: Dilation kernel size (2 → receptive field doubles per layer).
        bias: Whether convolutions carry a bias term.
    """

    def __init__(
        self,
        layers: int,
        blocks: int,
        dilation_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        classes: int,
        kernel_size: int,
        in_channels: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.layers = layers
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.classes = classes

        # ---- build dilation schedule (identical to reference) ----
        self.dilations: List[int] = []
        for _b in range(blocks):
            new_dilation = 1
            for _i in range(layers):
                self.dilations.append(new_dilation)
                new_dilation *= 2

        # ---- channel-expansion 1×1 (input → residual stream) ----
        # in_channels (>=1) is the number of stacked source signals; classes (=1)
        # is the regression OUTPUT width (end_conv_2). They are decoupled so a
        # 2-channel input (ecg_ppg2abp) still yields a 1-channel ABP output.
        self.start_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=residual_channels,
            kernel_size=1,
            bias=bias,
        )

        # ---- per-layer dilated conv modules ----
        n_layers = blocks * layers
        self.filter_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=residual_channels,
                out_channels=dilation_channels,
                kernel_size=kernel_size,
                dilation=self.dilations[i],
                padding=0,
                bias=bias,
            )
            for i in range(n_layers)
        ])
        self.gate_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=residual_channels,
                out_channels=dilation_channels,
                kernel_size=kernel_size,
                dilation=self.dilations[i],
                padding=0,
                bias=bias,
            )
            for i in range(n_layers)
        ])
        self.residual_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=dilation_channels,
                out_channels=residual_channels,
                kernel_size=1,
                bias=bias,
            )
            for _ in range(n_layers)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=dilation_channels,
                out_channels=skip_channels,
                kernel_size=1,
                bias=bias,
            )
            for _ in range(n_layers)
        ])

        # ---- post-stack projections ----
        self.end_conv_1 = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=1,
            bias=True,
        )
        self.end_conv_2 = nn.Conv1d(
            in_channels=end_channels,
            out_channels=classes,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the WaveNet stack.

        Args:
            x: Input tensor of shape (B, in_channels, L).

        Returns:
            Output tensor of shape (B, classes, L) (classes=1), same length as input.
        """
        x = self.start_conv(x)
        skip: Optional[torch.Tensor] = None

        for i in range(self.blocks * self.layers):
            residual = x

            # Causal (left-only) padding so output length == input length
            pad_left = (self.kernel_size - 1) * self.dilations[i]
            x_p = F.pad(x, (pad_left, 0))

            # Gated activation
            f = torch.tanh(self.filter_convs[i](x_p))
            g = torch.sigmoid(self.gate_convs[i](x_p))
            x = f * g

            # Skip accumulation
            s = self.skip_convs[i](x)
            skip = s if skip is None else skip + s

            # Residual connection
            x = self.residual_convs[i](x) + residual

        if skip is None:
            raise RuntimeError("WaveNet forward: skip connection never accumulated")

        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x  # (B, classes, L)


# ---------------------------------------------------------------------------
# BaselineModule wrapper
# ---------------------------------------------------------------------------

class WaveNet(BaselineModule):
    """WaveNet waveform regressor wrapped to the baseline contract.

    - work_multiple = 1  (causal conv; no length divisibility constraint)
    - has_bp_head = False  (single-stage: directly outputs waveform)
    - wave_aux = []  (no deep supervision)

    The source WaveNetModel outputs ``(B, output_length, classes)``; this
    wrapper trims the last ``output_length`` time-steps from the spatial axis
    and permutes to ``(B, 1, output_length)``.

    Args:
        layers: Dilation layers per block (default 10).
        blocks: Number of blocks (default 4).
        dilation_channels: Filter/gate conv width (default 32).
        residual_channels: Residual stream width (default 32).
        skip_channels: Skip accumulator width (default 256).
        end_channels: Post-stack hidden width (default 256).
        kernel_size: Dilation kernel size (default 2).
        output_length: Number of output time-steps trimmed from the right
            (set to seq_len / work_length so output == input length).
        bias: Bias in convolutions (default False).
    """

    work_multiple: int = 1
    has_bp_head: bool = False

    def __init__(
        self,
        layers: int = 10,
        blocks: int = 4,
        dilation_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 256,
        kernel_size: int = 2,
        output_length: int = 1250,
        in_channels: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.output_length = output_length

        # classes=1 for regression (not 256-way softmax); in_channels>1 for
        # multi-source directions (ecg_ppg2abp stacks ECG+PPG into 2 channels).
        self.core = _WaveNetCore(
            layers=layers,
            blocks=blocks,
            dilation_channels=dilation_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            classes=1,
            kernel_size=kernel_size,
            in_channels=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, want_bp: bool = False) -> Dict[str, object]:
        """Forward pass.

        Args:
            x: Input waveform of shape (B, C, Lw) (C=1, or 2 for ecg_ppg2abp).
               Lw must be at least ``output_length``; the trainer reflect-pads to
               work_multiple (=1) before calling, so Lw == output_length typically.
            want_bp: Ignored (has_bp_head is False).

        Returns:
            dict with keys:
                ``wave``      (B, 1, Lw) — trimmed to ``output_length`` from
                              the right, matching causal WaveNet convention.
                ``wave_aux``  [] — no deep supervision.
                ``bp``        None — no BP head.
        """
        # core: (B, 1, Lw) → (B, 1, Lw)
        out = self.core(x)          # (B, 1, Lw)

        # Trim to output_length from the right (causal output; mirrors original
        # x[:, :, -out_len:] then transpose(1,2) which gave (B,T,C) in source)
        Lw = out.shape[-1]
        trim = min(self.output_length, Lw)
        wave = out[:, :, -trim:]    # (B, 1, output_length)

        return {
            "wave": wave,
            "wave_aux": [],
            "bp": None,
        }


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

@register_model("wavenet")
def factory(params: dict, seq_len: int, direction: str) -> WaveNet:
    """Construct a WaveNet baseline.

    Args:
        params: Hyper-parameter overrides (all optional; defaults match the
            original MD-ViSCo regression configuration).
        seq_len: Source sequence length in samples.
        direction: Task direction string, e.g. ``"ecg2abp"`` or ``"ppg2ecg"``.
            WaveNet is single-stage and never builds a BP head regardless of
            direction.

    Returns:
        Configured :class:`WaveNet` instance.
    """
    work_multiple = 1  # class attribute; no padding needed
    work_length = next_multiple(seq_len, work_multiple)  # == seq_len

    return WaveNet(
        layers=int(params.get("layers", 10)),
        blocks=int(params.get("blocks", 4)),
        dilation_channels=int(params.get("dilation_channels", 32)),
        residual_channels=int(params.get("residual_channels", 32)),
        skip_channels=int(params.get("skip_channels", 256)),
        end_channels=int(params.get("end_channels", 256)),
        kernel_size=int(params.get("kernel_size", 2)),
        output_length=work_length,
        in_channels=int(params.get("in_channels", num_source_channels(direction))),
        bias=bool(params.get("bias", False)),
    )
