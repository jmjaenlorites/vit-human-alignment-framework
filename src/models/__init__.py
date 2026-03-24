from .base import BaseModelAdapter
from .resolver import (
    INTERNAL_MODEL_FACTORIES,
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
    "SUPPORTED_MODELS",
    "TIMM_MODEL_PREFIX",
    "list_supported_model_options",
    "load_model",
    "resolve_model",
    "validate_model_name",
]
