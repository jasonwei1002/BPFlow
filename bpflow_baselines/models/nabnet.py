"""NABNet baseline — self-contained port of MD-ViSCo's NABNet.

Architecture: Nested Attention-guided BiConvLSTM UNet with deep supervision.
Source paper: "NABNet: A Nested Attention-guided BiConvLSTM network for robust
prediction of Blood Pressure from reconstructed ABP using PPG and ECG"
https://linkinghub.elsevier.com/retrieve/pii/S1746809422007017

No runtime dependency on MD-ViSCo/src. All structural hyperparameters are
driven through the ``params`` dict in the factory.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    BaselineModule,
    next_multiple,
    register_model,
)
from ..norms import num_source_channels

# ---------------------------------------------------------------------------
# shared primitive blocks (faithful to original)
# ---------------------------------------------------------------------------


def _conv_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
    """Conv1d -> BN -> ReLU (same-padding via kernel//2)."""
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _trans_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose1d(k=2, s=2) -> BN -> ReLU — doubles spatial resolution."""
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# attention blocks (standard and LSTM variants)
# ---------------------------------------------------------------------------


class _AttentionBlock(nn.Module):
    """Standard gated attention on skip connection (original NABNet, A_G=1)."""

    def __init__(self, channels: int, multiplier: int = 2) -> None:
        super().__init__()
        mid = channels * multiplier
        self.conv1 = nn.Conv1d(channels, mid, 1)
        self.bn1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(channels, mid, 1)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv3 = nn.Conv1d(mid, 1, 1)
        self.bn3 = nn.BatchNorm1d(1)

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        """Return attention-weighted skip; spatially aligns gating if needed."""
        if skip.size(2) != gating.size(2):
            gating = F.interpolate(
                gating, size=skip.size(2), mode="linear", align_corners=False
            )
        x = F.relu(self.bn1(self.conv1(skip)) + self.bn2(self.conv2(gating)))
        alpha = torch.sigmoid(self.bn3(self.conv3(x)))  # (B,1,L)
        return skip * alpha


class _AttentionLSTMBlock(nn.Module):
    """BiLSTM-based attention on skip connection (paper's default, attention_type='lstm')."""

    def __init__(self, channels: int, lstm_multiplier: float = 1.0) -> None:
        super().__init__()
        self.lstm_hidden = int(channels * lstm_multiplier)
        self.lstm_skip = nn.LSTM(
            channels, self.lstm_hidden, bidirectional=True, batch_first=True
        )
        self.lstm_up = nn.LSTM(
            channels, self.lstm_hidden, bidirectional=True, batch_first=True
        )
        # Multi-head attention over BiLSTM outputs
        self.mha = nn.MultiheadAttention(self.lstm_hidden * 2, num_heads=1)
        # Project [attn_out || skip_feat] -> channels
        self.out_proj = nn.Sequential(
            nn.Linear(self.lstm_hidden * 4, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Return LSTM-attention-weighted skip tensor (B, C, L)."""
        # (B, C, L) -> (B, L, C)
        skip_t = skip.transpose(1, 2)
        up_t = up.transpose(1, 2)

        skip_feat, _ = self.lstm_skip(skip_t)   # (B, L, 2*H)
        up_feat, _ = self.lstm_up(up_t)          # (B, L, 2*H)

        # MHA expects (L, B, C)
        skip_q = skip_feat.transpose(0, 1)
        up_kv = up_feat.transpose(0, 1)
        attn_out, _ = self.mha(skip_q, up_kv, up_kv)  # (L, B, 2*H)
        attn_out = attn_out.transpose(0, 1)             # (B, L, 2*H)

        combined = torch.cat([attn_out, skip_feat], dim=-1)  # (B, L, 4*H)
        out = self.out_proj(combined)                          # (B, L, C)
        return out.transpose(1, 2)                             # (B, C, L)


# ---------------------------------------------------------------------------
# main NABNet UNet body (stage 1)
# ---------------------------------------------------------------------------


class _NABNetUNet(nn.Module):
    """UNet body of NABNet with optional deep supervision.

    Returns a tuple of (model_depth+1) tensors when d_s=True:
        tuple[0]  — full-resolution output (B,1,Lw)   [primary]
        tuple[1:] — deep-supervision auxiliary outputs  [coarser resolutions]
    When d_s=False returns a single tensor (B,1,Lw).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_depth: int,
        model_width: int,
        kernel_size: int,
        attention_type: str,
        d_s: bool,
        a_g: bool,
    ) -> None:
        super().__init__()
        self.model_depth = model_depth
        self.d_s = d_s
        self.a_g = a_g

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(model_depth):
            out_ch = model_width * (2 ** i)
            self.encoder_blocks.append(
                nn.Sequential(
                    _conv_block(in_ch, out_ch, kernel_size),
                    _conv_block(out_ch, out_ch, kernel_size),
                )
            )
            in_ch = out_ch

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ---- Bridge ----
        bridge_in = model_width * (2 ** (model_depth - 1))
        bridge_out = model_width * (2 ** model_depth)
        self.bridge = nn.Sequential(
            _conv_block(bridge_in, bridge_out, kernel_size),
            _conv_block(bridge_out, bridge_out, kernel_size),
        )

        # ---- Decoder upsampling blocks ----
        self.decoder_ups = nn.ModuleList()
        for i in range(model_depth):
            in_ch = model_width * (2 ** (model_depth - i))
            out_ch = model_width * (2 ** (model_depth - i - 1))
            self.decoder_ups.append(_trans_conv_block(in_ch, out_ch))

        # ---- Attention blocks (decoder depth → 0 order) ----
        if a_g:
            self.attention_blocks: Optional[nn.ModuleList] = nn.ModuleList()
            for i in range(model_depth - 1, -1, -1):
                ch = model_width * (2 ** i)
                if attention_type == "lstm":
                    self.attention_blocks.append(_AttentionLSTMBlock(ch, lstm_multiplier=1.0))
                else:
                    self.attention_blocks.append(_AttentionBlock(ch, multiplier=2))
        else:
            self.attention_blocks = None

        # ---- Decoder conv blocks ----
        self.decoder_convs = nn.ModuleList()
        for i in range(model_depth):
            curr_w = model_width * (2 ** (model_depth - i - 1))
            self.decoder_convs.append(
                nn.Sequential(
                    _conv_block(curr_w * 2, curr_w, kernel_size),
                    _conv_block(curr_w, curr_w, kernel_size),
                )
            )

        # ---- Deep supervision heads (one per decoder level, coarse→fine) ----
        if d_s:
            # Built in reverse index order matching decoder path; reversed later.
            self.ds_convs: Optional[nn.ModuleList] = nn.ModuleList(
                [
                    nn.Conv1d(model_width * (2 ** i), out_channels, 1)
                    for i in range(model_depth)
                ][::-1]
            )
        else:
            self.ds_convs = None

        # ---- Final output ----
        self.final_conv = nn.Conv1d(model_width, out_channels, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run NABNet forward.

        Returns:
            wave:     (B,1,Lw) full-resolution primary output.
            aux_list: list of deep-supervision outputs (coarser), may be [].
        """
        enc_feats: List[torch.Tensor] = []

        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
            enc_feats.append(x)
            x = self.pool(x)

        # Bridge
        x = self.bridge(x)

        # Decoder
        enc_feats_rev = enc_feats[::-1]
        ds_outputs: List[torch.Tensor] = []

        for i in range(self.model_depth):
            x = self.decoder_ups[i](x)
            skip = enc_feats_rev[i]

            if self.a_g and self.attention_blocks is not None:
                skip = self.attention_blocks[i](skip, x)

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_convs[i](x)

            if self.d_s and self.ds_convs is not None:
                ds_outputs.append(self.ds_convs[i](x))

        wave = self.final_conv(x)

        if self.d_s:
            # ds_outputs were collected coarse→fine during decoding.
            # Original code appends full-res last, then reverses: [full, ..., coarsest].
            # Contract: wave=tuple[0] (full-res), wave_aux=list(tuple[1:]).
            ds_outputs.append(wave)
            ordered = ds_outputs[::-1]  # [full, ..., coarsest]
            return ordered[0], ordered[1:]
        else:
            return wave, []


# ---------------------------------------------------------------------------
# BP head (stage 2) — used only for ->ABP directions
# ---------------------------------------------------------------------------


class _BPHead(nn.Module):
    """Lightweight CNN encoder + global-average-pool + MLP -> (SBP, DBP) in [0,1].

    Faithful to the ShallowUNet+MultiMLPRegressor spirit of the original:
    a shallow encoder extracts features from the reconstructed ABP waveform,
    global average pooling collapses the temporal axis, and two independent
    linear heads produce SBP and DBP.  Output is linear (no activation).
    """

    def __init__(self, wave_width: int, bp_hidden: int = 128) -> None:
        """Build the BP head.

        Args:
            wave_width: model_width of the main NABNet (sets encoder channel count).
            bp_hidden: hidden size of the MLP layers (default 128).
        """
        super().__init__()
        # Shallow CNN encoder: 3 double-conv layers with pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(1, wave_width, 3, padding=1),
            nn.BatchNorm1d(wave_width),
            nn.ReLU(inplace=True),
            nn.Conv1d(wave_width, wave_width, 3, padding=1),
            nn.BatchNorm1d(wave_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(wave_width, wave_width * 2, 3, padding=1),
            nn.BatchNorm1d(wave_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(wave_width * 2, wave_width * 2, 3, padding=1),
            nn.BatchNorm1d(wave_width * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # global average pool -> (B, wave_width*2, 1)
        )
        feat_dim = wave_width * 2
        # Two independent MLP heads: SBP and DBP
        self.sbp_head = nn.Sequential(
            nn.Linear(feat_dim, bp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bp_hidden, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(feat_dim, bp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(bp_hidden, 1),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Predict SBP and DBP from a waveform.

        Args:
            wave: (B, 1, Lw) reconstructed ABP or input waveform (normalised).

        Returns:
            (B, 2) tensor of (SBP, DBP) in global-min-max [0,1] (linear output;
            de-normalized to mmHg in reconstruct.py).
        """
        feat = self.encoder(wave).squeeze(-1)      # (B, feat_dim)
        sbp = self.sbp_head(feat)                  # (B, 1)
        dbp = self.dbp_head(feat)                  # (B, 1)
        return torch.cat([sbp, dbp], dim=1)        # (B, 2)


# ---------------------------------------------------------------------------
# BaselineModule wrapper
# ---------------------------------------------------------------------------


class NABNetBaseline(BaselineModule):
    """NABNet wrapped to the BPFlow baseline contract.

    Attributes:
        work_multiple: Minimum divisibility for input length (2**model_depth).
        has_bp_head:   True when direction ends with '2abp'.
    """

    work_multiple: int
    has_bp_head: bool

    def __init__(
        self,
        model_depth: int = 5,
        model_width: int = 128,
        kernel_size: int = 3,
        attention_type: str = "lstm",
        d_s: bool = True,
        a_g: bool = True,
        tgt_is_abp: bool = True,
        bp_hidden: int = 128,
        in_channels: int = 1,
    ) -> None:
        """Initialise NABNetBaseline.

        Args:
            model_depth:    Number of encoder/decoder levels (default 5).
            model_width:    Channel width of shallowest encoder block (default 128).
            kernel_size:    Convolution kernel size (default 3, paper-aligned).
            attention_type: 'lstm' (paper default) or 'standard'.
            d_s:            Enable deep supervision (default True, D_S=1).
            a_g:            Enable guided attention (default True, A_G=1).
            tgt_is_abp:     Build BP head when target is ABP.
            bp_hidden:      Hidden size of the BP-head MLP layers.
            in_channels:    Number of stacked source signals (1, or 2 for
                            ecg_ppg2abp). NABNet's UNet natively accepts this;
                            the BP head reads the ABP wave output, not x, so it
                            is unaffected.
        """
        super().__init__()
        self.work_multiple = 2 ** model_depth
        self.has_bp_head = tgt_is_abp

        self._unet = _NABNetUNet(
            in_channels=in_channels,
            out_channels=1,
            model_depth=model_depth,
            model_width=model_width,
            kernel_size=kernel_size,
            attention_type=attention_type,
            d_s=d_s,
            a_g=a_g,
        )

        self._bp_head: Optional[_BPHead] = None
        if tgt_is_abp:
            self._bp_head = _BPHead(wave_width=model_width, bp_hidden=bp_hidden)

    def forward(
        self, x: torch.Tensor, want_bp: bool = False
    ) -> dict:
        """Run NABNet forward pass.

        Args:
            x:        (B, C, Lw) input waveform (C=1, or 2 for ecg_ppg2abp); Lw a
                      multiple of work_multiple.
            want_bp:  If True and has_bp_head, also return (B,2) BP prediction.

        Returns:
            dict with keys:
                'wave'     : (B, 1, Lw)
                'wave_aux' : List[(B, 1, L_i)] deep-supervision outputs (may be [])
                'bp'       : (B, 2) | None
        """
        wave, aux_list = self._unet(x)

        bp: Optional[torch.Tensor] = None
        if want_bp and self.has_bp_head and self._bp_head is not None:
            bp = self._bp_head(wave)

        return {
            "wave": wave,
            "wave_aux": aux_list,
            "bp": bp,
        }


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


@register_model("nabnet")
def factory(params: dict, seq_len: int, direction: str) -> NABNetBaseline:
    """Build a NABNetBaseline from a params dict.

    Args:
        params:    Hyperparameter overrides (all optional, defaults = paper settings).
                   Recognised keys: model_depth, model_width, kernel_size,
                   attention_type, d_s, a_g, bp_hidden.
        seq_len:   Input sequence length before any padding.
        direction: E.g. 'ecg2abp', 'ppg2abp', 'ppg2ecg'. Determines whether a
                   BP head is constructed.

    Returns:
        NABNetBaseline instance.
    """
    model_depth = int(params.get("model_depth", 5))
    model_width = int(params.get("model_width", 128))
    kernel_size = int(params.get("kernel_size", 3))
    attention_type = str(params.get("attention_type", "lstm"))
    d_s = bool(params.get("d_s", True))
    a_g = bool(params.get("a_g", True))
    bp_hidden = int(params.get("bp_hidden", 128))
    in_channels = int(params.get("in_channels", num_source_channels(direction)))

    tgt_is_abp = direction.endswith("2abp")

    # Verify input length divisibility (informational; trainer pads anyway).
    work_multiple = 2 ** model_depth
    _ = next_multiple(seq_len, work_multiple)  # no-op but validates types

    return NABNetBaseline(
        model_depth=model_depth,
        model_width=model_width,
        kernel_size=kernel_size,
        attention_type=attention_type,
        d_s=d_s,
        a_g=a_g,
        tgt_is_abp=tgt_is_abp,
        bp_hidden=bp_hidden,
        in_channels=in_channels,
    )
