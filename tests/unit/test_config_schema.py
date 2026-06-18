from src.metrics import list_supported_metric_names as list_supported_execution_metrics
from src.runner.config_schema import (
    DATASET_DEFAULT_GROUPS,
    get_metric_spec,
    list_supported_metric_names,
)


def test_metric_spec_exposes_family_and_allowed_keys() -> None:
    metric_spec = get_metric_spec("levels_triplet_accuracy")

    assert metric_spec.family == "levels"
    assert metric_spec.allowed_keys == {
        "split",
        "levels_path",
        "imagenet_path",
        "batch_size",
    }


def test_saliency_metric_spec_allows_json_config_keys() -> None:
    metric_spec = get_metric_spec("saliency_auc_judd")

    assert metric_spec.family == "saliency"
    assert metric_spec.allowed_keys == {"dataset_path", "batch_size"}


def test_dataset_default_groups_match_metric_families() -> None:
    assert DATASET_DEFAULT_GROUPS == {
        "levels",
        "visturing",
        "tid",
        "nights",
        "saliency",
    }


def test_config_schema_and_execution_registry_stay_in_sync() -> None:
    assert list_supported_metric_names() == list_supported_execution_metrics()
