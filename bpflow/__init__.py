"""BPFlow: ECG+PPG -> ABP waveform generation via flow matching.

Adapted from Meta WavFlow. WavFlow's flow-matching + DiT primitives are
vendored verbatim under ``bpflow/model/_vendor`` (so bpflow is self-contained
and needs no external ``wavflow`` package), and the audio/video machinery is
replaced with a 3-stream joint-attention DiT over time-aligned 1D
physiological signals (ECG + PPG condition the ABP target).

See ``plan/notes.md`` for the architecture decision record.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
