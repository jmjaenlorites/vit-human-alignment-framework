from functools import lru_cache

import timm

from .base import BaseModelAdapter
from .jax.adapter import GOOGLE_MODEL_PREFIX, DynamicGoogleViTAdapter
from .jax.augreg_index import (
    GOOGLE_AUGREG_INDEX_URL,
    list_available_google_models,
)
from .vit_b16 import DynamicTimmModelAdapter, ViT_B_16, ViT_B_32, ViT_H_14, ViT_L_14

TIMM_MODEL_PREFIX = "timm::"
INTERNAL_MODEL_FACTORIES = {
    "vit-b16": ViT_B_16,
    "vit-l14": ViT_L_14,
    "vit-h14": ViT_H_14,
    "vit-b32": ViT_B_32,
}


def list_internal_model_names() -> list[str]:
    return sorted(INTERNAL_MODEL_FACTORIES)


@lru_cache(maxsize=1)
def list_available_timm_models() -> list[str]:
    return sorted(set(timm.list_models()) | set(timm.list_models(pretrained=True)))


def list_supported_model_options() -> list[str]:
    options = list_internal_model_names()
    options.extend(
        f"{TIMM_MODEL_PREFIX}{model_name}"
        for model_name in list_available_timm_models()
    )
    options.extend(
        f"{GOOGLE_MODEL_PREFIX}{model_name}"
        for model_name in list_available_google_models()
    )
    return options


def validate_model_name(model_name: str) -> None:
    if model_name in INTERNAL_MODEL_FACTORIES:
        return

    if model_name.startswith(TIMM_MODEL_PREFIX):
        timm_model_name = model_name[len(TIMM_MODEL_PREFIX) :]
        if timm_model_name in list_available_timm_models():
            return
        raise ValueError(
            f"Model {model_name} not supported. Valid options: "
            f"{', '.join(f'{TIMM_MODEL_PREFIX}{name}' for name in list_available_timm_models())}"
        )

    if model_name.startswith(GOOGLE_MODEL_PREFIX):
        google_model_name = model_name[len(GOOGLE_MODEL_PREFIX) :]
        if google_model_name in set(list_available_google_models()):
            return
        count = len(list_available_google_models())
        raise ValueError(
            f"Model {model_name} not supported. AugReg index has {count} valid "
            f"filenames; see {GOOGLE_AUGREG_INDEX_URL}"
        )

    raise ValueError(
        f"Model {model_name} not supported. Valid options: {', '.join(list_internal_model_names())}"
    )


def resolve_model(model_name: str) -> BaseModelAdapter:
    validate_model_name(model_name)
    if model_name in INTERNAL_MODEL_FACTORIES:
        return INTERNAL_MODEL_FACTORIES[model_name]()
    if model_name.startswith(TIMM_MODEL_PREFIX):
        timm_model_name = model_name[len(TIMM_MODEL_PREFIX) :]
        return DynamicTimmModelAdapter(timm_model_name)
    if model_name.startswith(GOOGLE_MODEL_PREFIX):
        google_model_name = model_name[len(GOOGLE_MODEL_PREFIX) :]
        return DynamicGoogleViTAdapter(google_model_name)
    raise ValueError(f"Model {model_name} not supported")
