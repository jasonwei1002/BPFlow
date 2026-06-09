"""Single import surface for the WavFlow-derived primitives.

The primitives are vendored verbatim under ``bpflow/model/_vendor`` (copied from
Meta WavFlow), so ``bpflow`` is self-contained and never reaches into an external
``wavflow`` package. Keeping this as the one re-export point means the rest of
``bpflow`` imports from here, and the vendored snapshot stays read-only.

They import cleanly on CPU and forward correctly without torchaudio / torchdiffeq.
The cuda-typed ``autocast`` contexts inside ``embeddings`` / ``rotary_embeddings``
are no-ops off-GPU.
"""

from ._vendor import (
    MLP,
    ChannelLastConv1d,
    ConvMLP,
    FinalBlock,
    MMDitSingleBlock,
    TimestepEmbedder,
    attention,
    compute_rope_rotations,
)

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
