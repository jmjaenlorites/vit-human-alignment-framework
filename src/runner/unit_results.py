import csv
import re
from pathlib import Path
from typing import Any, Iterable

from .types import ResolvedExperiment


UNIT_RESULT_COLUMNS = [
    "run_key",
    "model_name",
    "experiment_id",
    "metric_name",
    "config_hash",
    "layer_index",
    "unit_index",
    "unit_value",
    "human_value",
    "prediction",
    "is_correct",
]


class UnitResultsWriter:
    """Writes optional per-unit metric outputs for bootstrap analyses."""

    def __init__(self, output_dir: str | None):
        self.output_dir = Path(output_dir) if output_dir else None

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def write(
        self,
        experiment: ResolvedExperiment,
        records: Iterable[dict[str, Any]],
    ) -> Path | None:
        if self.output_dir is None:
            return None

        rows = [
            self._normalize_row(experiment, record)
            for record in records
        ]
        if not rows:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self._filename(experiment)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=UNIT_RESULT_COLUMNS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _normalize_row(
        self,
        experiment: ResolvedExperiment,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        row = {column: "" for column in UNIT_RESULT_COLUMNS}
        row.update(
            {
                "run_key": experiment.run_key,
                "model_name": experiment.model_name,
                "experiment_id": experiment.experiment_id,
                "metric_name": experiment.metric_name,
                "config_hash": experiment.config_hash,
            }
        )
        for column in UNIT_RESULT_COLUMNS:
            if column in record:
                row[column] = record[column]
        return row

    def _filename(self, experiment: ResolvedExperiment) -> str:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment.run_key).strip("_")
        return f"{safe_key}__{experiment.config_hash[:12]}.csv"
