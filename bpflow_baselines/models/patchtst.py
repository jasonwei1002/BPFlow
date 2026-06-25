"""PatchTST baseline — self-contained port of MD-ViSCo's patchtst.py.

Architecture:
    Stage 1 (wave reconstruction): PatchTSTForRegression with num_targets=work_length
        input  x (B, C, Lw) -> permute -> (B, Lw, C) -> PatchTST -> (B, Lw) -> (B, 1, Lw)
        (C=1, or 2 for ecg_ppg2abp: two channel-independent input channels)
    Stage 2 (BP head, only when tgt_is_abp):
        a second PatchTSTForRegression with num_targets=2 maps the source waveform x
        directly to (SBP, DBP) scalars.

No Hydra, no MD-ViSCo imports.  All structural hyper-params come from the ``params``
dict passed to the factory.

Default params (matching the original paper / PulseDB settings):
    patch_len=16, stride=8, d_model=128, num_encoder_layers=3, num_heads=16,
    dropout=0.2, fc_dropout=0.1, head_dropout=0.1

References:
    - Nie et al. 2022, "A Time Series is Worth 64 Words: Long-term Forecasting with
      Transformers", https://arxiv.org/abs/2211.14730
    - HuggingFace transformers PatchTSTForRegression (tested on transformers==5.9.0)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import PatchTSTConfig, PatchTSTForRegression

from .base import (
    BaselineModule,
    register_model,
    next_multiple,
)
from ..norms import num_source_channels

__all__ = ["PatchTSTBaseline", "factory"]


def _build_patchtst_config(
    context_length: int,
    num_targets: int,
    patch_len: int,
    stride: int,
    d_model: int,
    num_encoder_layers: int,
    num_heads: int,
    dropout: float,
    fc_dropout: float,
    head_dropout: float,
    num_input_channels: int = 1,
) -> PatchTSTConfig:
    """Build a PatchTSTConfig from individual hyper-parameters.

    Args:
        context_length: Fixed input sequence length seen by the model.
        num_targets: Number of regression outputs.
        num_input_channels: Number of input channels. PatchTST is
            channel-independent: each channel is encoded by the SAME backbone in
            isolation (no cross-channel attention); fusion happens only at the
            flatten+linear regression head (num_input_channels*d_model -> targets).
            For ecg_ppg2abp this is 2 (ECG and PPG as two independent channels),
            which is the model's native multivariate handling.
        patch_len: Length of each temporal patch.
        stride: Stride between consecutive patches.
        d_model: Transformer embedding dimension.
        num_encoder_layers: Number of Transformer encoder layers.
        num_heads: Number of self-attention heads.
        dropout: Attention dropout rate.
        fc_dropout: Feed-forward dropout rate.
        head_dropout: Regression head dropout rate.

    Returns:
        A fully initialised PatchTSTConfig.
    """
    return PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=context_length,
        num_targets=num_targets,
        patch_length=patch_len,
        patch_stride=stride,
        d_model=d_model,
        num_hidden_layers=num_encoder_layers,
        num_attention_heads=num_heads,
        attention_dropout=dropout,
        ff_dropout=fc_dropout,
        head_dropout=head_dropout,
        # Architectural choices kept from the original MD-ViSCo implementation
        share_embedding=True,
        channel_attention=False,
        norm_type="batchnorm",
        activation_function="gelu",
        pre_norm=True,
        positional_encoding_type="sincos",
        use_cls_token=False,
        loss="mse",
    )


class PatchTSTBaseline(BaselineModule):
    """PatchTST baseline compatible with the BPFlow baseline trainer contract.

    Stage 1 reconstructs the target waveform (num_targets = work_length).
    Stage 2 (optional, only when ``tgt_is_abp=True``) predicts SBP/DBP from the
    source waveform with a second PatchTSTForRegression(num_targets=2).

    Attributes:
        work_multiple: Length divisibility requirement (1, i.e. no constraint).
        has_bp_head:   True when a stage-2 BP predictor was built.
    """

    work_multiple: int = 1
    has_bp_head: bool = False

    def __init__(
        self,
        work_length: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 16,
        dropout: float = 0.2,
        fc_dropout: float = 0.1,
        head_dropout: float = 0.1,
        tgt_is_abp: bool = False,
        num_input_channels: int = 1,
    ) -> None:
        """Initialise PatchTSTBaseline.

        Args:
            work_length: Fixed context / output length (= next_multiple(seq_len, 1)).
            patch_len: Temporal patch size.
            stride: Patch stride.
            d_model: Transformer hidden dimension.
            num_encoder_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            dropout: Attention dropout.
            fc_dropout: Feed-forward dropout.
            head_dropout: Head dropout.
            tgt_is_abp: Whether the target modality is ABP; enables the BP head.
            num_input_channels: Stacked source channels (2 for ecg_ppg2abp).
                Channel-independent encoding; both stage-1 and the BP head use it.
        """
        super().__init__()

        self.work_length = work_length
        self.tgt_is_abp = tgt_is_abp

        # ------------------------------------------------------------------
        # Stage 1: waveform reconstruction (num_targets = work_length)
        # ------------------------------------------------------------------
        stage1_cfg = _build_patchtst_config(
            context_length=work_length,
            num_targets=work_length,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            num_input_channels=num_input_channels,
        )
        self.stage1 = PatchTSTForRegression(stage1_cfg)

        # ------------------------------------------------------------------
        # Stage 2: BP head (only for ->ABP directions)
        # ------------------------------------------------------------------
        if tgt_is_abp:
            bp_cfg = _build_patchtst_config(
                context_length=work_length,
                num_targets=2,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                num_encoder_layers=num_encoder_layers,
                num_heads=num_heads,
                dropout=dropout,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                num_input_channels=num_input_channels,
            )
            self.stage2 = PatchTSTForRegression(bp_cfg)
            self.has_bp_head = True

    def forward(
        self, x: torch.Tensor, want_bp: bool = False
    ) -> Dict[str, object]:
        """Forward pass.

        Args:
            x: Source waveform tensor of shape (B, C, Lw), where Lw has already
               been padded to a multiple of ``work_multiple`` (trivially satisfied
               since work_multiple=1). C=1 normally, C=2 for ecg_ppg2abp.
            want_bp: When True and ``has_bp_head`` is set, also run stage 2 and
                     return (SBP, DBP) predictions.

        Returns:
            dict with keys:
                "wave"     : (B, 1, Lw)  — stage-1 waveform (linear output)
                "wave_aux" : []           — no deep-supervision auxiliaries
                "bp"       : (B, 2) (SBP, DBP) in global-min-max [0,1] (->mmHg in reconstruct) or None
        """
        # x: (B, C, Lw) -> PatchTST expects (B, Lw, C) (channel-independent)
        past_values = x.permute(0, 2, 1)  # (B, Lw, C)

        # Stage 1: waveform
        stage1_out = self.stage1(past_values=past_values)
        wave = stage1_out.regression_outputs  # (B, Lw)
        wave = wave.unsqueeze(1)             # (B, 1, Lw)

        # Stage 2: BP scalars (optional)
        bp: Optional[torch.Tensor] = None
        if want_bp and self.has_bp_head:
            stage2_out = self.stage2(past_values=past_values)
            bp = stage2_out.regression_outputs  # (B, 2)

        return {
            "wave": wave,
            "wave_aux": [],
            "bp": bp,
        }


@register_model("patchtst")
def factory(params: dict, seq_len: int, direction: str) -> PatchTSTBaseline:
    """Build a PatchTSTBaseline from a params dict.

    Args:
        params: Hyper-parameter overrides.  Any key absent falls back to the
                default matching the original MD-ViSCo / paper configuration.
        seq_len: Raw input sequence length (e.g. 1250 for PulseDB).
        direction: Task string (e.g. "ecg2abp", "ppg2ecg").  Determines whether
                   the BP head is built.

    Returns:
        Fully constructed and ready-to-train PatchTSTBaseline.
    """
    tgt_is_abp: bool = direction.endswith("2abp")
    work_length: int = next_multiple(seq_len, 1)  # work_multiple == 1

    return PatchTSTBaseline(
        work_length=work_length,
        patch_len=int(params.get("patch_len", 16)),
        stride=int(params.get("stride", 8)),
        d_model=int(params.get("d_model", 128)),
        num_encoder_layers=int(params.get("num_encoder_layers", 3)),
        num_heads=int(params.get("num_heads", 16)),
        dropout=float(params.get("dropout", 0.2)),
        fc_dropout=float(params.get("fc_dropout", 0.1)),
        head_dropout=float(params.get("head_dropout", 0.1)),
        tgt_is_abp=tgt_is_abp,
        num_input_channels=int(params.get("num_input_channels", num_source_channels(direction))),
    )
