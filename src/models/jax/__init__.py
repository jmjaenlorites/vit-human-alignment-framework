"""Soporte de modelos Vision Transformer de Google (Flax/JAX)."""

from .adapter import GOOGLE_MODEL_PREFIX, DynamicGoogleViTAdapter
from .augreg_index import (
    GoogleModelInfo,
    list_available_google_models,
    lookup_google_model,
)

__all__ = [
    "GOOGLE_MODEL_PREFIX",
    "DynamicGoogleViTAdapter",
    "GoogleModelInfo",
    "list_available_google_models",
    "lookup_google_model",
]
