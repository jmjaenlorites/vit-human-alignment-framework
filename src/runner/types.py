import hashlib
import json
from dataclasses import dataclass
from typing import Any


def normalize_config(config: dict[str, Any]) -> str:
    """Serializa una configuración de forma estable."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def compute_config_hash(config: dict[str, Any]) -> str:
    """Genera un hash estable para una configuración efectiva."""
    return hashlib.sha256(normalize_config(config).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedExperiment:
    experiment_id: str
    metric_name: str
    model_name: str
    config: dict[str, Any]
    run_key: str
    config_hash: str

    @classmethod
    def from_parts(
        cls,
        experiment_id: str,
        metric_name: str,
        model_name: str,
        config: dict[str, Any] | None = None,
    ) -> "ResolvedExperiment":
        effective_config = config or {}
        return cls(
            experiment_id=experiment_id,
            metric_name=metric_name,
            model_name=model_name,
            config=effective_config,
            run_key=f"{model_name}__{experiment_id}",
            config_hash=compute_config_hash(effective_config),
        )
