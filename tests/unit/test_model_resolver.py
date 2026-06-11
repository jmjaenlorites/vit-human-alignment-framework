import pytest

from src.models.resolver import (
    DynamicTimmModelAdapter,
    SALIENCY_BASELINE_MODEL_PREFIX,
    TIMM_MODEL_PREFIX,
    TIMM_RANDOM_MODEL_PREFIX,
    TIMM_UNTRAINED_MODEL_PREFIX,
    list_available_timm_models,
    list_supported_model_options,
    resolve_model,
    validate_model_name,
)
from src.models.saliency_baseline import SaliencyBaselineModelAdapter


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


def test_validate_model_name_accepts_saliency_baselines() -> None:
    validate_model_name(f"{SALIENCY_BASELINE_MODEL_PREFIX}uniform")
    validate_model_name(f"{SALIENCY_BASELINE_MODEL_PREFIX}center-gaussian")


def test_validate_model_name_rejects_unknown_saliency_baseline() -> None:
    with pytest.raises(ValueError, match="uniform.*center-gaussian"):
        validate_model_name(f"{SALIENCY_BASELINE_MODEL_PREFIX}unknown")


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
        f"{SALIENCY_BASELINE_MODEL_PREFIX}uniform",
        f"{SALIENCY_BASELINE_MODEL_PREFIX}center-gaussian",
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


@pytest.mark.parametrize("baseline_name", ["uniform", "center-gaussian"])
def test_resolve_model_returns_saliency_baseline_adapter(baseline_name: str) -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}{baseline_name}")

    assert isinstance(model, SaliencyBaselineModelAdapter)
    assert model.name == f"{SALIENCY_BASELINE_MODEL_PREFIX}{baseline_name}"


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
        "random_seed": 0,
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


def test_dynamic_timm_adapter_uses_deterministic_seed_for_forced_random(
    monkeypatch,
) -> None:
    import torch

    class FakeTimmModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    captured = {}

    def fake_create_model(model_name: str, pretrained: bool, cache_dir):
        captured["initial_seed"] = torch.initial_seed()
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
    torch.manual_seed(999)
    state_before = torch.random.get_rng_state()

    DynamicTimmModelAdapter(
        "vit_base_patch16_224",
        config={"force_random_init": True, "random_seed": 123},
    )

    assert captured["initial_seed"] == 123
    assert torch.equal(torch.random.get_rng_state(), state_before)
