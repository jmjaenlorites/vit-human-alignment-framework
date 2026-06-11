from .base import BaseModelAdapter
from .resolver import (
    GOOGLE_MODEL_PREFIX,
    INTERNAL_MODEL_FACTORIES,
    SALIENCY_BASELINE_MODEL_PREFIX,
    TIMM_MODEL_PREFIX,
    list_supported_model_options,
    resolve_model,
    validate_model_name,
)

SUPPORTED_MODELS = set(INTERNAL_MODEL_FACTORIES)


def load_model(model_name: str) -> BaseModelAdapter:
    return resolve_model(model_name)


__all__ = [
    "BaseModelAdapter",
    "GOOGLE_MODEL_PREFIX",
    "SALIENCY_BASELINE_MODEL_PREFIX",
    "SUPPORTED_MODELS",
    "TIMM_MODEL_PREFIX",
    "list_supported_model_options",
    "load_model",
    "resolve_model",
    "validate_model_name",
]
