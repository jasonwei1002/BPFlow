"""Dataset registry + factory for BPFlow."""

from typing import Callable, Dict

from torch.utils.data import Dataset

from .pulsedb_dataset import PulseDBDataset, build_dataset
from .transforms import (
    build_cond_patches,
    destandardize_abp,
    patchify,
    standardize_abp,
    unpatchify,
)

DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    "pulsedb": PulseDBDataset,
}


def register_dataset(name: str) -> Callable[[Callable[..., Dataset]], Callable[..., Dataset]]:
    def decorator(factory: Callable[..., Dataset]) -> Callable[..., Dataset]:
        DATASET_REGISTRY[name] = factory
        return factory

    return decorator


__all__ = [
    "PulseDBDataset",
    "build_dataset",
    "DATASET_REGISTRY",
    "register_dataset",
    "standardize_abp",
    "destandardize_abp",
    "patchify",
    "unpatchify",
    "build_cond_patches",
]
