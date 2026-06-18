import csv

import pandas as pd
import pytest

from src.runner.results_store import ResultsStore
from src.runner.types import ResolvedExperiment


def make_experiment(**overrides) -> ResolvedExperiment:
    base = {
        "experiment_id": "levels-within",
        "metric_name": "levels_triplet_accuracy",
        "model_name": "vit-b16",
        "config": {"split": "within_class"},
    }
    base.update(overrides)
    return ResolvedExperiment.from_parts(**base)


def test_results_store_marks_running_done_and_skips_completed(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    assert store.should_run(experiment) is True

    store.mark_running(experiment)
    assert store.should_run(experiment) is True

    store.mark_done(experiment, result="[0.1, 0.2]")
    assert store.should_run(experiment) is False

    frame = pd.read_csv(tmp_path / "results.csv")
    assert frame.loc[0, "status"] == "done"
    assert frame.loc[0, "result"] == "[0.1, 0.2]"


def test_results_store_reruns_stale_running_entries(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    store.mark_running(experiment)

    assert store.should_run(experiment) is True


def test_results_store_skips_errors_by_default(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    store.mark_error(experiment, error="boom")

    assert store.should_run(experiment) is False
    assert store.should_run(experiment, rerun_errors=True) is True


def test_results_store_detects_config_hash_mismatch(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()
    changed = make_experiment(config={"split": "between_class"})

    store.mark_done(experiment, result="[1.0]")

    with pytest.raises(ValueError, match="config_hash"):
        store.should_run(changed)


def test_results_store_writes_expected_columns(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    store.mark_running(experiment)

    with open(tmp_path / "results.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ResultsStore.COLUMNS


def test_results_store_updates_done_record_without_dtype_warning(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    store.mark_running(experiment)
    store.mark_done(experiment, result="[0.1, 0.2]")

    frame = pd.read_csv(tmp_path / "results.csv")
    assert frame.loc[0, "result"] == "[0.1, 0.2]"


def test_results_store_records_started_at_and_duration(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    store.mark_running(experiment)
    store.mark_done(experiment, result="[0.5]")

    frame = pd.read_csv(tmp_path / "results.csv")
    started_at = frame.loc[0, "started_at"]
    duration = frame.loc[0, "duration_seconds"]

    assert isinstance(started_at, str)
    # datetime.fromisoformat parses the ISO 8601 string written by mark_running
    from datetime import datetime

    datetime.fromisoformat(started_at)
    assert float(duration) >= 0.0


def test_results_store_mark_error_records_duration(tmp_path) -> None:
    store = ResultsStore(str(tmp_path / "results.csv"))
    experiment = make_experiment()

    store.mark_running(experiment)
    store.mark_error(experiment, error="boom")

    frame = pd.read_csv(tmp_path / "results.csv")
    assert frame.loc[0, "status"] == "error"
    assert float(frame.loc[0, "duration_seconds"]) >= 0.0


def test_results_store_adds_missing_columns_to_legacy_csv(tmp_path) -> None:
    """A CSV written before the timing columns existed must still be readable."""
    legacy_path = tmp_path / "legacy.csv"
    legacy_df = pd.DataFrame(
        [
            {
                "run_key": "legacy_key",
                "model_name": "vit-b16",
                "experiment_id": "levels-within",
                "metric_name": "levels_triplet_accuracy",
                "status": "done",
                "result": "[0.3]",
                "error": "",
                "config_hash": "deadbeef",
            }
        ]
    )
    legacy_df.to_csv(legacy_path, index=False)

    store = ResultsStore(str(legacy_path))
    experiment = make_experiment()
    store.mark_running(experiment)
    store.mark_done(experiment, result="[0.1]")

    frame = pd.read_csv(legacy_path)
    assert set(ResultsStore.COLUMNS).issubset(frame.columns)
    assert len(frame) == 2
