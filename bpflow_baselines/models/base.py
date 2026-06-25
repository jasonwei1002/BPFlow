"""Baseline model interface, length helpers, and registry.

Every baseline is a vendored / Hydra-stripped nn.Module wrapped to a single,
uniform contract so one trainer + one eval path serve all six. See
plan/baselines-repro/design.md.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# length helpers — pad source/target to the model's divisibility requirement,
# crop predictions back to the true length for loss / eval.
# ---------------------------------------------------------------------------

def next_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    return ((length + multiple - 1) // multiple) * multiple


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Reflect-pad the last (time) dim of x up to a multiple of ``multiple``."""
    L = x.shape[-1]
    target = next_multiple(L, multiple)
    if target == L:
        return x
    pad = target - L
    left = pad // 2
    right = pad - left
    # reflect needs pad < L; ABP/ECG segments are long so this always holds.
    return F.pad(x, (left, right), mode="reflect")


def crop_center(x: torch.Tensor, length: int) -> torch.Tensor:
    """Inverse of pad_to_multiple: center-crop the last dim back to ``length``."""
    L = x.shape[-1]
    if L == length:
        return x
    extra = L - length
    left = extra // 2
    return x[..., left:left + length]


# ---------------------------------------------------------------------------
# model contract
# ---------------------------------------------------------------------------
class BaselineModule(nn.Module):
    """Uniform baseline contract.

    Subclasses set ``work_multiple`` (length divisibility) and ``has_bp_head``
    (whether a stage-2 SBP/DBP regressor exists, used only for ->ABP directions),
    and implement ``forward(x, want_bp=False)`` returning a dict with:
        wave      (B, 1, Lw)            primary stage-1 waveform output
        wave_aux  List[(B, 1, Lw_i)]    deep-supervision aux outputs (may be [])
        bp        (B, 2) | None         (SBP, DBP) in global-min-max [0,1] (NOT mmHg;
                                        de-normalized to mmHg in reconstruct.py) if want_bp
    The trainer pads inputs to Lw and crops ``wave`` back for loss/eval.
    """

    work_multiple: int = 1
    has_bp_head: bool = False

    def forward(self, x: torch.Tensor, want_bp: bool = False) -> Dict[str, object]:  # noqa: D401
        raise NotImplementedError


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Callable[..., BaselineModule]] = {}


def register_model(name: str) -> Callable[[Callable[..., BaselineModule]], Callable[..., BaselineModule]]:
    def deco(factory: Callable[..., BaselineModule]) -> Callable[..., BaselineModule]:
        MODEL_REGISTRY[name] = factory
        return factory
    return deco


def build_model(cfg) -> BaselineModule:
    """Construct a baseline from cfg.model.{name,params} and cfg (for seq_len/dir).

    The factory receives (params: dict, seq_len: int, direction: str). The
    direction lets a model know whether the target is ABP (enable bp head) or a
    bridge waveform (stage-1 only).
    """
    name = str(cfg.model.name)
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown baseline model {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    params = dict(cfg.model.params) if cfg.model.params else {}
    seq_len = int(cfg.data.seq_len)
    direction = str(cfg.baseline.direction)
    return MODEL_REGISTRY[name](params=params, seq_len=seq_len, direction=direction)
