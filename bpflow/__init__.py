"""BPFlow: ECG+PPG -> ABP waveform generation via flow matching.

Adapted from Meta WavFlow (vendored at ``../wavflow``). Reuses WavFlow's
flow-matching + DiT primitives verbatim, and replaces the audio/video
machinery with a single-stream conditional DiT over time-aligned 1D
physiological signals (ECG, PPG condition the ABP target).

See ``plan/notes.md`` for the architecture decision record.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
