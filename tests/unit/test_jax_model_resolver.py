"""Tests del resolver para el namespace ``google::``.

Mismo patrón que `tests/unit/test_model_resolver.py` para `timm::`: se
monkeypatchea `list_available_google_models` y la `__init__` del adapter para
no descargar checkpoints reales.
"""

import pytest

from src.models.jax import augreg_index
from src.models.jax.adapter import DynamicGoogleViTAdapter
from src.models.resolver import (
    GOOGLE_MODEL_PREFIX,
    list_available_google_models,
    list_supported_model_options,
    resolve_model,
    validate_model_name,
)


def _patch_google_index(monkeypatch, names: list[str]) -> None:
    list_available_google_models.cache_clear()
    augreg_index.list_available_google_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_google_models",
        lambda: names,
    )


def test_validate_model_name_accepts_google_filename(monkeypatch) -> None:
    _patch_google_index(monkeypatch, ["Ti_16-i1k-300ep"])
    validate_model_name(f"{GOOGLE_MODEL_PREFIX}Ti_16-i1k-300ep")


def test_validate_model_name_rejects_unknown_google_filename(monkeypatch) -> None:
    _patch_google_index(monkeypatch, ["Ti_16-i1k-300ep"])
    with pytest.raises(ValueError, match="AugReg index"):
        validate_model_name(f"{GOOGLE_MODEL_PREFIX}Bogus_Filename-bogus")


def test_list_supported_model_options_includes_google_namespace(monkeypatch) -> None:
    _patch_google_index(monkeypatch, ["Ti_16-i1k-300ep"])
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224"],
    )
    options = list_supported_model_options()
    assert f"{GOOGLE_MODEL_PREFIX}Ti_16-i1k-300ep" in options
    # Internal y timm siguen presentes (no se rompió nada).
    assert "vit-b16" in options
    assert "timm::vit_base_patch16_224" in options


def test_resolve_model_returns_dynamic_google_adapter(monkeypatch) -> None:
    _patch_google_index(monkeypatch, ["Ti_16-i1k-300ep"])
    captured = {}

    def fake_init(self, filename: str, config=None) -> None:
        captured["filename"] = filename

    monkeypatch.setattr(DynamicGoogleViTAdapter, "__init__", fake_init)

    model = resolve_model(f"{GOOGLE_MODEL_PREFIX}Ti_16-i1k-300ep")

    assert isinstance(model, DynamicGoogleViTAdapter)
    assert captured["filename"] == "Ti_16-i1k-300ep"


def test_validate_model_name_error_for_unknown_google_includes_index_url(
    monkeypatch,
) -> None:
    _patch_google_index(monkeypatch, ["Ti_16-i1k-300ep"])
    with pytest.raises(ValueError) as exc_info:
        validate_model_name(f"{GOOGLE_MODEL_PREFIX}Unknown")
    assert "storage.googleapis.com" in str(exc_info.value)
