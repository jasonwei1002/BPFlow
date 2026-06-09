"""Re-export verified WavFlow primitives.

These import cleanly on CPU and forward correctly without torchaudio /
torchdiffeq (verified on torch 2.11). The cuda-typed ``autocast`` contexts
inside ``embeddings`` / ``rotary_embeddings`` are no-ops off-GPU.

Keeping this as a single import surface means the rest of ``bpflow`` never
reaches into ``wavflow`` directly, so the vendored reference stays read-only
and the reuse boundary is explicit.
"""

from wavflow.model.embeddings import TimestepEmbedder
from wavflow.model.low_level import ChannelLastConv1d, ConvMLP, MLP
from wavflow.model.rotary_embeddings import compute_rope_rotations
from wavflow.model.transformer_layers import FinalBlock, MMDitSingleBlock, attention

__all__ = [
    "MMDitSingleBlock",
    "FinalBlock",
    "attention",
    "ChannelLastConv1d",
    "ConvMLP",
    "MLP",
    "TimestepEmbedder",
    "compute_rope_rotations",
]
