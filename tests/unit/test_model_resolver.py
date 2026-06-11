import pytest

from src.models.resolver import (
    DynamicTimmModelAdapter,
    TIMM_MODEL_PREFIX,
    TIMM_RANDOM_MODEL_PREFIX,
    TIMM_UNTRAINED_MODEL_PREFIX,
    list_available_timm_models,
    list_supported_model_options,
    resolve_model,
    validate_model_name,
)


def test_list_available_timm_models_includes_pretrained_tags(monkeypatch) -> None:
    list_available_timm_models.cache_clear()

    def fake_list_models(*, pretrained: bool = False):
        if pretrained:
            return ["vit_base_patch16_224.mae"]
        return ["vit_base_patch16_224"]

    monkeypatch.setattr("src.models.resolver.timm.list_models", fake_list_models)

    assert list_available_timm_models() == [
        "vit_base_patch16_224",
        "vit_base_patch16_224.mae",
    ]

    list_available_timm_models.cache_clear()


def test_validate_model_name_accepts_timm_pretrained_tag(monkeypatch) -> None:
    list_available_timm_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224", "vit_base_patch16_224.mae"],
    )

    validate_model_name(f"{TIMM_MODEL_PREFIX}vit_base_patch16_224")
    validate_model_name(f"{TIMM_MODEL_PREFIX}vit_base_patch16_224.mae")


def test_validate_model_name_accepts_random_timm_prefixes(monkeypatch) -> None:
    list_available_timm_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224", "vit_base_patch16_224.mae"],
    )

    validate_model_name(f"{TIMM_RANDOM_MODEL_PREFIX}vit_base_patch16_224")
    validate_model_name(f"{TIMM_UNTRAINED_MODEL_PREFIX}vit_base_patch16_224.mae")


def test_validate_model_name_accepts_internal_alias() -> None:
    validate_model_name("vit-b16")


def test_validate_model_name_rejects_unknown_timm_pretrained_tag(monkeypatch) -> None:
    list_available_timm_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224.mae"],
    )

    with pytest.raises(ValueError, match="vit_base_patch16_224.mae"):
        validate_model_name(f"{TIMM_MODEL_PREFIX}vit_base_patch16_224.unknown_tag")


def test_list_supported_model_options_includes_internal_and_timm_models(
    monkeypatch,
) -> None:
    list_available_timm_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224", "vit_base_patch16_224.mae"],
    )
    # Aislamos el namespace google::* para no depender de red.
    monkeypatch.setattr(
        "src.models.resolver.list_available_google_models",
        lambda: [],
    )

    assert list_supported_model_options() == [
        "vit-b16",
        "vit-b32",
        "vit-h14",
        "vit-l14",
        f"{TIMM_MODEL_PREFIX}vit_base_patch16_224",
        f"{TIMM_MODEL_PREFIX}vit_base_patch16_224.mae",
        f"{TIMM_RANDOM_MODEL_PREFIX}vit_base_patch16_224",
        f"{TIMM_RANDOM_MODEL_PREFIX}vit_base_patch16_224.mae",
        f"{TIMM_UNTRAINED_MODEL_PREFIX}vit_base_patch16_224",
        f"{TIMM_UNTRAINED_MODEL_PREFIX}vit_base_patch16_224.mae",
    ]


def test_resolve_model_returns_dynamic_adapter_for_timm_pretrained_tag(
    monkeypatch,
) -> None:
    list_available_timm_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224.mae"],
    )

    captured = {}

    def fake_init(self, timm_model_name: str, config=None):
        captured["timm_model_name"] = timm_model_name

    monkeypatch.setattr(DynamicTimmModelAdapter, "__init__", fake_init)

    model = resolve_model(f"{TIMM_MODEL_PREFIX}vit_base_patch16_224.mae")

    assert isinstance(model, DynamicTimmModelAdapter)
    assert captured["timm_model_name"] == "vit_base_patch16_224.mae"


def test_resolve_model_forces_random_initialization_for_random_timm_prefix(
    monkeypatch,
) -> None:
    list_available_timm_models.cache_clear()
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224"],
    )

    captured = {}

    def fake_init(self, timm_model_name: str, config=None):
        captured["timm_model_name"] = timm_model_name
        captured["config"] = config

    monkeypatch.setattr(DynamicTimmModelAdapter, "__init__", fake_init)

    model = resolve_model(f"{TIMM_RANDOM_MODEL_PREFIX}vit_base_patch16_224")

    assert isinstance(model, DynamicTimmModelAdapter)
    assert captured["timm_model_name"] == "vit_base_patch16_224"
    assert captured["config"] == {
        "force_random_init": True,
        "display_name": "timm-random::vit_base_patch16_224",
    }


def test_dynamic_timm_adapter_ignores_pretrained_weights_when_random_is_forced(
    monkeypatch,
) -> None:
    class FakeTimmModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    captured = {}

    def fake_create_model(model_name: str, pretrained: bool, cache_dir):
        captured["model_name"] = model_name
        captured["pretrained"] = pretrained
        captured["cache_dir"] = cache_dir
        return FakeTimmModel()

    monkeypatch.setattr(
        "src.models.vit_b16.list_pretrained_timm_models",
        lambda: {"vit_base_patch16_224"},
    )
    monkeypatch.setattr("src.models.vit_b16.timm.create_model", fake_create_model)
    monkeypatch.setattr(
        "src.models.vit_b16.timm.data.resolve_data_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "src.models.vit_b16.timm.data.create_transform",
        lambda *args, **kwargs: "fake-transform",
    )

    adapter = DynamicTimmModelAdapter(
        "vit_base_patch16_224",
        config={"force_random_init": True},
    )

    assert adapter.transform == "fake-transform"
    assert captured["model_name"] == "vit_base_patch16_224"
    assert captured["pretrained"] is False
