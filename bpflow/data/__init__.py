"""Dataset registry + factory for BPFlow."""

from typing import Callable, Dict

from torch.utils.data import Dataset

from .pulsedb_dataset import (
    DEFAULT_MODALITY_DROPOUT_PROBS,
    MODALITY_MASK,
    MODALITY_ORDER,
    PulseDBDataset,
    build_dataset,
    trained_modalities,
)
from .transforms import (
    DEMO_CONT_DIM,
    build_cond_patches,
    destandardize_abp,
    patchify,
    standardize_abp,
    standardize_demo,
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
    "MODALITY_ORDER",
    "MODALITY_MASK",
    "DEFAULT_MODALITY_DROPOUT_PROBS",
    "trained_modalities",
    "DATASET_REGISTRY",
    "register_dataset",
    "standardize_abp",
    "destandardize_abp",
    "patchify",
    "unpatchify",
    "build_cond_patches",
    "standardize_demo",
    "DEMO_CONT_DIM",
]
