"""Vendored WavFlow primitives (verbatim copies, license headers preserved).

These four modules are copied unchanged from Meta WavFlow's ``model/`` so that
``bpflow`` is self-contained and does not require the ``wavflow`` package to be
importable at runtime. See each file's header for upstream attribution
(MMAudio / facebookresearch DiT / openai glide) and licenses; the DiT-derived
portions are CC BY-NC 4.0 (non-commercial).

Do not edit these files — treat them as a read-only vendored snapshot. BPFlow
code imports them only through ``bpflow.model._wavflow_layers``.
"""

from .embeddings import TimestepEmbedder
from .low_level import ChannelLastConv1d, ConvMLP, MLP
from .rotary_embeddings import compute_rope_rotations
from .transformer_layers import FinalBlock, MMDitSingleBlock, attention

__all__ = [
    "TimestepEmbedder",
    "ChannelLastConv1d",
    "ConvMLP",
    "MLP",
    "compute_rope_rotations",
    "FinalBlock",
    "MMDitSingleBlock",
    "attention",
]
