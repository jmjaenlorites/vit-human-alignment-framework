from typing import Any

import pandas as pd

from .types import ResolvedExperiment


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
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

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
        self._upsert(
            experiment,
            status="running",
            result=None,
            error=None,
        )

    def mark_done(self, experiment: ResolvedExperiment, result: Any) -> None:
        self._upsert(
            experiment,
            status="done",
            result=result,
            error=None,
        )

    def mark_error(self, experiment: ResolvedExperiment, error: str) -> None:
        self._upsert(
            experiment,
            status="error",
            result=None,
            error=error,
        )

    def _upsert(
        self,
        experiment: ResolvedExperiment,
        status: str,
        result: Any,
        error: str | None,
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
        }
        mask = df["run_key"] == experiment.run_key
        if mask.any():
            for column, value in row.items():
                df.loc[mask, column] = value
        else:
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

        missing_columns = [
            column for column in self.COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Results CSV is missing required columns: {missing_columns}"
            )
        return self._normalize_dtypes(df[self.COLUMNS].copy())

    def _normalize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.COLUMNS:
            df[column] = df[column].astype(object)
        return df
