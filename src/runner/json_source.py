import json
from pathlib import Path
from typing import Any

from .config_validation import (
    validate_metric_config,
    validate_metric_name,
    validate_json_model_name,
)
from .config_schema import DATASET_DEFAULT_GROUPS, get_metric_spec
from .types import ResolvedExperiment


class JSONExperimentSource:
    """Carga y expande experimentos declarados en JSON."""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)

    def load(self) -> list[ResolvedExperiment]:
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))

        defaults = self._ensure_dict(payload.get("defaults", {}), field_name="defaults")
        dataset_defaults = self._ensure_dict(
            payload.get("dataset_defaults", {}),
            field_name="dataset_defaults",
        )
        models = payload.get("models", [])
        experiments = payload.get("experiments", [])

        self._validate_dataset_defaults(dataset_defaults)

        if not models:
            raise ValueError("JSON config must include at least one model in 'models'")
        if not experiments:
            raise ValueError(
                "JSON config must include at least one experiment in 'experiments'"
            )

        resolved_experiments: list[ResolvedExperiment] = []
        for model_name in models:
            validate_json_model_name(model_name)

        for experiment in experiments:
            experiment_dict = self._ensure_dict(
                experiment,
                field_name="experiments[]",
            )
            experiment_id = experiment_dict.get("experiment_id")
            metric_name = experiment_dict.get("metric")
            if not experiment_id:
                raise ValueError("Each experiment must define 'experiment_id'")
            if not metric_name:
                raise ValueError(f"Experiment '{experiment_id}' must define 'metric'")
            validate_metric_name(metric_name)

            experiment_config = self._ensure_dict(
                experiment_dict.get("config", {}),
                field_name=f"experiments[{experiment_id}].config",
            )
            dataset_config = dataset_defaults.get(
                get_metric_spec(metric_name).family,
                {},
            )
            effective_config = {**defaults, **dataset_config, **experiment_config}
            validate_metric_config(metric_name, effective_config)

            for model_name in models:
                resolved_experiments.append(
                    ResolvedExperiment.from_parts(
                        experiment_id=experiment_id,
                        metric_name=metric_name,
                        model_name=model_name,
                        config=effective_config,
                    )
                )

        return resolved_experiments

    def _ensure_dict(self, value: Any, field_name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"Field '{field_name}' must be an object")
        return value

    def _validate_dataset_defaults(self, dataset_defaults: dict[str, Any]) -> None:
        unknown_groups = sorted(set(dataset_defaults) - DATASET_DEFAULT_GROUPS)
        if unknown_groups:
            raise ValueError(f"Unsupported dataset_defaults groups: {unknown_groups}")

        for group_name, group_config in dataset_defaults.items():
            self._ensure_dict(group_config, field_name=f"dataset_defaults.{group_name}")
