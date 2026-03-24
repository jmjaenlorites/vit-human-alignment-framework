from typing import Any

from ..models.resolver import list_supported_model_options, validate_model_name
from .config_schema import PATH_FIELDS, get_metric_spec, list_supported_metric_names


def _format_options(values) -> str:
    return ", ".join(sorted(values))


def validate_metric_name(metric_name: str) -> None:
    try:
        get_metric_spec(metric_name)
    except ValueError as exc:
        raise ValueError(
            f"Metric {metric_name} not supported. Valid options: {_format_options(list_supported_metric_names())}"
        ) from exc


def validate_json_model_name(model_name: str) -> None:
    try:
        validate_model_name(model_name)
    except ValueError as exc:
        supported_options = list_supported_model_options()
        if model_name.startswith("timm::"):
            raise ValueError(
                f"Model {model_name} not supported. Valid options: {_format_options(supported_options)}"
            ) from exc
        raise ValueError(
            f"Model {model_name} not supported. Valid options: {_format_options(supported_options)}"
        ) from exc


def validate_metric_config(metric_name: str, config: dict[str, Any]) -> None:
    metric_spec = get_metric_spec(metric_name)
    allowed_keys = metric_spec.allowed_keys
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported config keys for metric {metric_name}: {unknown_keys}. "
            f"Allowed keys: {_format_options(allowed_keys)}"
        )

    for key, valid_options in metric_spec.value_options.items():
        if key in config and config[key] not in valid_options:
            raise ValueError(
                f"Invalid value for '{key}' in metric {metric_name}: {config[key]}. "
                f"Valid options: {_format_options(valid_options)}"
            )

    for key, value in config.items():
        if key == "batch_size" and (not isinstance(value, int) or value <= 0):
            raise ValueError(
                "Invalid value for 'batch_size': must be a positive integer"
            )
        if key in PATH_FIELDS and (not isinstance(value, str) or not value):
            raise ValueError(f"Invalid value for '{key}': must be a non-empty string")
