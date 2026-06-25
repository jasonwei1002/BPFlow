"""Baseline model registry. Importing this package registers all six models.

Each model module calls ``@register_model("<name>")`` on its factory. The
factory signature is ``factory(params: dict, seq_len: int, direction: str)``.
"""

from .base import MODEL_REGISTRY, BaselineModule, build_model, register_model

# Import model modules so their @register_model factories run. Wrapped so a
# single broken/in-progress model doesn't block the others during development.
_MODEL_MODULES = ["wavenet", "nabnet", "ppg2abp", "patchtst", "p2e_wgan", "mdvisco"]
for _m in _MODEL_MODULES:
    try:
        __import__(f"{__name__}.{_m}", fromlist=["*"])
    except Exception as _e:  # pragma: no cover - surfaced at build_model time
        import logging
        logging.getLogger(__name__).warning("baseline model %r not loaded: %s", _m, _e)

__all__ = ["MODEL_REGISTRY", "BaselineModule", "build_model", "register_model"]
