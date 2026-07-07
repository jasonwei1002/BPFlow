"""Model registry + factory for BPFlow generators.

One model variant: a 3-stream joint-attention conditional DiT
(``bpflow_jointstream``). The factory/registry pattern is kept so future
variants can register by name and build from a config object.
"""

from typing import Callable, Dict

import torch.nn as nn

from .flow_matching import FlowMatching, log_normal_sample
from .networks import BPConditions, BPFlowModel

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


@register_model("bpflow_jointstream")
def _build_jointstream(cfg) -> nn.Module:
    """3-stream joint-attention DiT (ABP + ECG + PPG), WavFlow JointBlock style."""
    return BPFlowModel(
        seq_len=int(cfg.data.seq_len),
        patch_size=int(cfg.model.patch_size),
        hidden_dim=int(cfg.model.hidden_dim),
        num_heads=int(cfg.model.num_heads),
        depth=int(cfg.model.depth),
        joint_depth=int(cfg.model.joint_depth),
        mlp_ratio=float(cfg.model.mlp_ratio),
        stream_fusion=str(cfg.model.stream_fusion),
    )


def build_model(cfg) -> nn.Module:
    """Build a model from ``cfg.model.name`` (seq_len comes from cfg.data)."""
    name = str(cfg.model.name)
    factory = MODEL_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown model name '{name}'. Registered: {list(MODEL_REGISTRY)}")
    return factory(cfg)


__all__ = [
    "BPFlowModel",
    "BPConditions",
    "FlowMatching",
    "log_normal_sample",
    "MODEL_REGISTRY",
    "register_model",
    "build_model",
]
