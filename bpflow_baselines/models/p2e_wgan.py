"""P2E-WGAN baseline — ported from MD-ViSCo, self-contained (no Hydra/MD-ViSCo).

References:
    Paper: "P2E-WGAN: ECG waveform synthesis from PPG with conditional
    Wasserstein generative adversarial networks"
    https://dl.acm.org/doi/10.1145/3412841.3441979

Architecture
------------
Generator  : 1-D U-Net with 4 stride-2 downsampling stages + 3 transposed-conv
             upsampling stages and skip connections.  Final layer is Tanh → [-1,1].
Discriminator: PatchGAN-style, receives cat([cond, waveform], dim=1) (2 channels),
             4 stride-2 downsampling blocks + final conv1d, no sigmoid (WGAN real
             scores).

``work_multiple = 16`` because the generator performs 4 stride-2 downsampling
steps, so the input length must be divisible by 2**4 = 16.

``has_bp_head = False`` — this model was designed for PPG→ECG translation and
does not include a BP regression head.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .base import (
    BaselineModule,
    register_model,
)
from ..norms import num_source_channels

__all__ = ["P2EWGANBaseline", "GeneratorUNet", "Discriminator", "weights_init_normal"]


# ---------------------------------------------------------------------------
# Weight initialisation (faithful to the original)
# ---------------------------------------------------------------------------

def weights_init_normal(m: nn.Module) -> None:
    """Initialise Conv weights N(0, 0.02); BatchNorm1d weights N(1, 0.02).

    Args:
        m: Module to initialise.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class UNetDown(nn.Module):
    """Stride-2 downsampling block: Conv1d → [InstanceNorm1d] → LeakyReLU → [Dropout].

    Args:
        in_size: Input channels.
        out_size: Output channels.
        ksize: Kernel size.
        stride: Convolution stride.
        normalize: Whether to apply InstanceNorm1d.
        dropout: Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        ksize: int = 4,
        stride: int = 2,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = ksize // 2
        layers: List[nn.Module] = [
            nn.Conv1d(
                in_size,
                out_size,
                kernel_size=ksize,
                stride=stride,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )
        ]
        if normalize:
            layers.append(nn.InstanceNorm1d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, L).

        Returns:
            Downsampled tensor (B, out_size, L//stride).
        """
        return self.model(x)


class UNetUp(nn.Module):
    """Stride-2 upsampling block with skip connection.

    Applies ConvTranspose1d → InstanceNorm1d → ReLU → [Dropout], then
    centre-crops the larger of {upsampled, skip} to match the other and
    concatenates along the channel dimension.

    Args:
        in_size: Input channels (to ConvTranspose1d).
        out_size: Output channels before concatenation.
        ksize: Kernel size.
        stride: Transposed-convolution stride.
        dropout: Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        ksize: int = 4,
        stride: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = ksize // 2
        output_padding = stride - 1
        layers: List[nn.Module] = [
            nn.ConvTranspose1d(
                in_size,
                out_size,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.InstanceNorm1d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        """Upsample x and concatenate with skip_input.

        Length mismatches are resolved by centre-cropping the longer tensor.

        Args:
            x: Input tensor (B, in_size, L_in).
            skip_input: Skip-connection tensor from the matching encoder stage.

        Returns:
            Concatenated tensor (B, out_size + skip_channels, L).
        """
        x = self.model(x)
        # Align lengths by centre-cropping the longer tensor
        if x.size(2) != skip_input.size(2):
            if x.size(2) > skip_input.size(2):
                diff = x.size(2) - skip_input.size(2)
                x = x[:, :, diff // 2 : -(diff - diff // 2)]
            else:
                diff = skip_input.size(2) - x.size(2)
                skip_input = skip_input[:, :, diff // 2 : -(diff - diff // 2)]
        return torch.cat((x, skip_input), dim=1)


# ---------------------------------------------------------------------------
# Generator: 1-D U-Net
# ---------------------------------------------------------------------------

class GeneratorUNet(nn.Module):
    """1-D U-Net generator.

    Four encoder stages (each /2) and three decoder stages (+skip), with a
    final Upsample + Conv1d + Tanh that restores the original length.

    Output is in [-1, 1] (Tanh).  The trainer/BaselineModule.forward wrapper
    returns this Tanh output as ``wave``.

    Args:
        in_channels: Source signal channels (default 1).
        out_channels: Target signal channels (default 1).
        init_filters: Feature-map count at the first encoder stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 128,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if init_filters <= 0:
            raise ValueError(f"init_filters must be positive, got {init_filters}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters

        f1 = init_filters          # 1× filters
        f2 = init_filters * 2      # 2× filters
        f3 = init_filters * 4      # 4× filters
        f4 = init_filters * 4      # 4× filters (bottleneck)

        # Encoder (4 down blocks → /16 total)
        self.down1 = UNetDown(in_channels, f1, normalize=False)  # L → L/2
        self.down2 = UNetDown(f1, f2)                            # L/2 → L/4
        self.down3 = UNetDown(f2, f3, dropout=0.5)              # L/4 → L/8
        self.down4 = UNetDown(f3, f4, dropout=0.5, normalize=False)  # L/8 → L/16

        # Decoder (3 up blocks — each doubles; final layer doubles back to L)
        self.up1 = UNetUp(f4, f3, dropout=0.5)                  # L/16 → L/8, +skip d3 → ch f3*2
        self.up2 = UNetUp(f3 * 2, f2)                           # L/8  → L/4, +skip d2 → ch f2*2
        self.up3 = UNetUp(f2 * 2, f1)                           # L/4  → L/2, +skip d1 → ch f1*2

        # Final: Upsample(×2) + Conv + Tanh → (B, out_channels, L)
        final_ksize = 4
        final_padding = final_ksize // 2
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                f1 * 2,
                out_channels,
                kernel_size=final_ksize,
                padding=final_padding,
                padding_mode="replicate",
            ),
            nn.Tanh(),
        )

        self.apply(weights_init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate output waveform from source waveform.

        Args:
            x: Source signal (B, in_channels, L).

        Returns:
            Generated waveform (B, out_channels, L) in [-1, 1].
        """
        original_size = x.size(2)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        out = self.final(u3)

        # Ensure output length matches input length (centre-crop if padding added)
        if out.size(2) != original_size:
            if out.size(2) > original_size:
                diff = out.size(2) - original_size
                out = out[:, :, diff // 2 : diff // 2 + original_size]
            else:
                raise ValueError(
                    f"Generator output length {out.size(2)} < input length "
                    f"{original_size}; check padding."
                )
        return out


# ---------------------------------------------------------------------------
# Discriminator: PatchGAN (no sigmoid, WGAN real scores)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """PatchGAN discriminator for WGAN-GP training.

    Input is cat([cond_source, waveform], dim=1): in_channels (source) +
    out_channels (target) channels.  For the original PPG->ECG that is 1+1=2;
    for ecg_ppg2abp it is 2+1=3.  Output is a 1-D feature map of raw scores (no
    sigmoid), as required by WGAN.

    Four downsampling blocks (kernel 4, stride 2) followed by a final Conv1d
    with stride 1 to collapse to a score patch.

    Args:
        in_channels: Channels of the conditioning source signal(s) (1, or 2 for
            ecg_ppg2abp).
        out_channels: Channels of the target waveform (1).
        init_filters: Feature-map count at the first discriminator stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 128,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if init_filters <= 0:
            raise ValueError(f"init_filters must be positive, got {init_filters}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters

        f1 = init_filters          # 1×
        f2 = init_filters * 2      # 2×
        f3 = init_filters * 4      # 4×
        f4 = init_filters * 8      # 8×

        def _block(
            in_f: int,
            out_f: int,
            ksize: int = 4,
            stride: int = 2,
            normalization: bool = True,
        ) -> List[nn.Module]:
            padding = ksize // 2
            layers: List[nn.Module] = [
                nn.Conv1d(
                    in_f,
                    out_f,
                    ksize,
                    stride=stride,
                    padding=padding,
                    padding_mode="replicate",
                )
            ]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # cond(in_channels) + target(out_channels) channels: 1+1=2, or 2+1=3
            *_block(in_channels + out_channels, f1, normalization=False),  # →(B,f1,L/2)
            *_block(f1, f2),                                     # (B,f1,L/2)→(B,f2,L/4)
            *_block(f2, f3),                                     # (B,f2,L/4)→(B,f3,L/8)
            *_block(f3, f4),                                     # (B,f3,L/8)→(B,f4,L/16)
            nn.Conv1d(f4, 1, kernel_size=4, stride=1, padding=1,
                      padding_mode="replicate"),                  # (B,f4,L/16)→(B,1,L/16)
        )

        self.apply(weights_init_normal)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Score a concatenated (source, waveform) pair.

        Args:
            z: Concatenated tensor (B, in_channels + out_channels, L).  The caller is
               responsible for cat([cond_source, waveform], dim=1).

        Returns:
            Raw score map (B, 1, L') — no sigmoid.
        """
        return self.model(z)


# ---------------------------------------------------------------------------
# BaselineModule wrapper
# ---------------------------------------------------------------------------

class P2EWGANBaseline(BaselineModule):
    """P2E-WGAN wrapped in the BaselineModule contract.

    Exposes ``self.generator`` and ``self.discriminator`` as sub-modules so
    that a GAN trainer can access them directly for adversarial optimisation.

    ``forward(x, want_bp=False)`` delegates to the generator and returns the
    standard baseline dict (used by the shared inference / evaluation path).

    Args:
        in_channels: Source signal channels.
        out_channels: Target signal channels.
        generator_init_filters: Feature-map count at the first generator stage.
        discriminator_init_filters: Feature-map count at the first discriminator
            stage.
    """

    work_multiple: int = 16
    has_bp_head: bool = False

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        generator_init_filters: int = 128,
        discriminator_init_filters: int = 128,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if generator_init_filters <= 0:
            raise ValueError(
                f"generator_init_filters must be positive, got {generator_init_filters}"
            )
        if discriminator_init_filters <= 0:
            raise ValueError(
                f"discriminator_init_filters must be positive, "
                f"got {discriminator_init_filters}"
            )

        self.generator = GeneratorUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=generator_init_filters,
        )
        self.discriminator = Discriminator(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=discriminator_init_filters,
        )

    def forward(
        self, x: torch.Tensor, want_bp: bool = False
    ) -> Dict[str, Any]:
        """Run the generator; discriminator is accessed separately by a GAN trainer.

        Args:
            x: Source waveform (B, C, Lw) (C=1, or 2 for ecg_ppg2abp).  Lw must be divisible by
               ``work_multiple`` (the trainer pads before calling).
            want_bp: Ignored — this model has no BP head.

        Returns:
            Dict with:
                ``wave``      (B, 1, Lw)  — generator output in [-1, 1] (Tanh).
                ``wave_aux``  []           — no deep-supervision auxiliaries.
                ``bp``        None         — no BP head.
        """
        wave = self.generator(x)
        return {
            "wave": wave,
            "wave_aux": [],
            "bp": None,
        }


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

@register_model("p2e_wgan")
def factory(
    params: Dict[str, Any],
    seq_len: int,
    direction: str,
) -> P2EWGANBaseline:
    """Build a P2EWGANBaseline from a params dict.

    All keys are optional; missing keys fall back to the paper / PulseDB defaults.

    Args:
        params: Hyper-parameter overrides.  Supported keys:
            ``in_channels`` (int, default 1),
            ``out_channels`` (int, default 1),
            ``generator_init_filters`` (int, default 128),
            ``discriminator_init_filters`` (int, default 128).
        seq_len: Input sequence length (before padding).  Not used directly
            here because the U-Net is fully-convolutional (any length that is
            a multiple of ``work_multiple`` is accepted).
        direction: E.g. ``"ppg2ecg"`` or ``"ecg_ppg2abp"``.  Used to derive the
            source channel count (``num_source_channels``); ``has_bp_head`` is
            always False for this model.

    Returns:
        Initialised P2EWGANBaseline.
    """
    in_channels: int = int(params.get("in_channels", num_source_channels(direction)))
    out_channels: int = int(params.get("out_channels", 1))
    generator_init_filters: int = int(params.get("generator_init_filters", 128))
    discriminator_init_filters: int = int(
        params.get("discriminator_init_filters", 128)
    )
    return P2EWGANBaseline(
        in_channels=in_channels,
        out_channels=out_channels,
        generator_init_filters=generator_init_filters,
        discriminator_init_filters=discriminator_init_filters,
    )
