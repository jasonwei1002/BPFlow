"""Dataset registry + factory for BPFlow."""

from typing import Callable, Dict

from torch.utils.data import Dataset

from .pulsedb_dataset import (
    ABP_TASKS,
    DEFAULT_TASK_PROBS,
    TARGET_ORDER,
    TASK_ORDER,
    TASK_SPEC,
    PulseDBDataset,
    build_dataset,
    resolve_tasks,
    trained_tasks,
)
from .transforms import (
    DEMO_CONT_DIM,
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
    "TARGET_ORDER",
    "TASK_ORDER",
    "TASK_SPEC",
    "ABP_TASKS",
    "DEFAULT_TASK_PROBS",
    "resolve_tasks",
    "trained_tasks",
    "DATASET_REGISTRY",
    "register_dataset",
    "standardize_abp",
    "destandardize_abp",
    "patchify",
    "unpatchify",
    "standardize_demo",
    "DEMO_CONT_DIM",
]
