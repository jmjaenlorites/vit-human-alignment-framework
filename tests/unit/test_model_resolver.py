from types import SimpleNamespace

import pytest

from src.models.resolver import (
    TIMM_MODEL_PREFIX,
    list_supported_model_options,
    resolve_model,
    validate_model_name,
)


def test_validate_model_name_accepts_internal_alias() -> None:
    validate_model_name("vit-b16")


def test_validate_model_name_accepts_timm_prefixed_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224"],
    )

    validate_model_name(f"{TIMM_MODEL_PREFIX}vit_base_patch16_224")


def test_validate_model_name_rejects_unknown_timm_model_with_options(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224", "eva02_base_patch14_224"],
    )

    with pytest.raises(ValueError, match="timm::vit_base_patch16_224"):
        validate_model_name(f"{TIMM_MODEL_PREFIX}unknown_model")


def test_resolve_model_uses_internal_factory(monkeypatch) -> None:
    fake_model = SimpleNamespace(name="internal")
    monkeypatch.setitem(
        __import__(
            "src.models.resolver", fromlist=["INTERNAL_MODEL_FACTORIES"]
        ).INTERNAL_MODEL_FACTORIES,
        "vit-b16",
        lambda: fake_model,
    )

    assert resolve_model("vit-b16") is fake_model


def test_resolve_model_uses_dynamic_timm_adapter(monkeypatch) -> None:
    captured = {}

    class FakeAdapter:
        def __init__(self, timm_model_name: str):
            captured["timm_model_name"] = timm_model_name

    monkeypatch.setattr(
        "src.models.resolver.validate_model_name",
        lambda model_name: None,
    )
    monkeypatch.setattr("src.models.resolver.DynamicTimmModelAdapter", FakeAdapter)

    resolve_model(f"{TIMM_MODEL_PREFIX}vit_base_patch16_224")

    assert captured == {"timm_model_name": "vit_base_patch16_224"}


def test_supported_model_options_include_internal_and_timm_hint(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224"],
    )

    options = list_supported_model_options()

    assert "vit-b16" in options
    assert f"{TIMM_MODEL_PREFIX}vit_base_patch16_224" in options
