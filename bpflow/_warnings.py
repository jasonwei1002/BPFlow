"""Silence non-fatal third-party warnings so train/infer/smoke logs stay readable.

The suppressed categories (Deprecation / Future / PendingDeprecation) only ever
report upstream API churn from torch / swanlab / numpy — none affect a run's
correctness. ``UserWarning`` is deliberately KEPT VISIBLE: it is the channel torch
uses for real correctness signals (DDP unused-params, tensor-shape / num_features
mismatches, dtype issues), and blanket-silencing it would hide genuine problems.
``RuntimeWarning`` is also kept (NaN/inf, divide-by-zero in metrics).

Escape hatch: set ``BPFLOW_SHOW_WARNINGS=1`` to restore Python's default warning
behavior (e.g. when chasing an upstream deprecation).
"""

import logging
import os
import warnings

logger = logging.getLogger(__name__)

# Categories that only signal upstream API churn / library stack noise, never a
# problem with the current run. UserWarning and RuntimeWarning are intentionally
# excluded — they can carry real correctness signals we must not hide.
_NOISE_CATEGORIES: tuple[type[Warning], ...] = (
    DeprecationWarning,
    FutureWarning,
    PendingDeprecationWarning,
)

_TRUTHY = {"1", "true", "yes", "on"}


def configure_warnings() -> None:
    """Install process-wide filters that drop non-fatal third-party warnings.

    Idempotent and safe to call multiple times. No-op when the env var
    ``BPFLOW_SHOW_WARNINGS`` is truthy.
    """
    if os.environ.get("BPFLOW_SHOW_WARNINGS", "").strip().lower() in _TRUTHY:
        logger.debug("BPFLOW_SHOW_WARNINGS set; leaving warning filters untouched")
        return
    for category in _NOISE_CATEGORIES:
        warnings.filterwarnings("ignore", category=category)
