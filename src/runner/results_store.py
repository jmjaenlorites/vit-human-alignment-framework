import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .types import ResolvedExperiment


class _KeepExisting:
    """Sentinel: do not touch the existing column value on an upsert."""


_KEEP = _KeepExisting()


class ResultsStore:
    """Gestiona el estado incremental de ejecuciones JSON en un CSV."""

    COLUMNS = [
        "run_key",
        "model_name",
        "experiment_id",
        "metric_name",
        "status",
        "result",
        "error",
        "config_hash",
        "started_at",
        "duration_seconds",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._perf_starts: dict[str, float] = {}

    def should_run(
        self,
        experiment: ResolvedExperiment,
        rerun_errors: bool = False,
    ) -> bool:
        record = self._get_record(experiment.run_key)
        if record is None:
            return True
        self._validate_config_hash(record, experiment)

        status = record["status"]
        if status == "done":
            return False
        if status == "error":
            return rerun_errors
        if status == "running":
            return True
        return True

    def mark_running(self, experiment: ResolvedExperiment) -> None:
        started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._perf_starts[experiment.run_key] = time.perf_counter()
        self._upsert(
            experiment,
            status="running",
            result=None,
            error=None,
            started_at=started_iso,
            duration_seconds=None,
        )

    def mark_done(self, experiment: ResolvedExperiment, result: Any) -> None:
        duration = self._consume_duration(experiment.run_key)
        self._upsert(
            experiment,
            status="done",
            result=result,
            error=None,
            started_at=_KEEP,
            duration_seconds=duration,
        )

    def mark_error(self, experiment: ResolvedExperiment, error: str) -> None:
        duration = self._consume_duration(experiment.run_key)
        self._upsert(
            experiment,
            status="error",
            result=None,
            error=error,
            started_at=_KEEP,
            duration_seconds=duration,
        )

    def _consume_duration(self, run_key: str) -> float | None:
        started = self._perf_starts.pop(run_key, None)
        if started is None:
            return None
        return round(time.perf_counter() - started, 3)

    def _upsert(
        self,
        experiment: ResolvedExperiment,
        status: str,
        result: Any,
        error: str | None,
        started_at: Any = _KEEP,
        duration_seconds: Any = None,
    ) -> None:
        df = self._normalize_dtypes(self._load_or_create())
        row = {
            "run_key": experiment.run_key,
            "model_name": experiment.model_name,
            "experiment_id": experiment.experiment_id,
            "metric_name": experiment.metric_name,
            "status": status,
            "result": result,
            "error": error,
            "config_hash": experiment.config_hash,
            "started_at": started_at,
            "duration_seconds": duration_seconds,
        }
        mask = df["run_key"] == experiment.run_key
        if mask.any():
            for column, value in row.items():
                if isinstance(value, _KeepExisting):
                    continue
                df.loc[mask, column] = value
        else:
            for column, value in list(row.items()):
                if isinstance(value, _KeepExisting):
                    row[column] = None
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

    def _get_record(self, run_key: str) -> pd.Series | None:
        df = self._load_or_create()
        matches = df.loc[df["run_key"] == run_key]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _validate_config_hash(
        self,
        record: pd.Series,
        experiment: ResolvedExperiment,
    ) -> None:
        if record["config_hash"] != experiment.config_hash:
            raise ValueError(
                f"Existing result for run_key '{experiment.run_key}' has a different config_hash"
            )

    def _load_or_create(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            return self._normalize_dtypes(pd.DataFrame(columns=self.COLUMNS))

        for column in self.COLUMNS:
            if column not in df.columns:
                df[column] = None
        return self._normalize_dtypes(df[self.COLUMNS].copy())

    def _normalize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.COLUMNS:
            df[column] = df[column].astype(object)
        return df
