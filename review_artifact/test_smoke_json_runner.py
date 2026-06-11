import csv

from review_artifact.smoke_json_runner import (
    EXPECTED_RESULTS_PATH,
    STABLE_COLUMNS,
    read_stable_rows,
    run_smoke,
)


def test_smoke_json_runner_persists_expected_rows_and_resumes(tmp_path) -> None:
    summary = run_smoke(
        results_path=tmp_path / "results.csv",
        normalized_path=tmp_path / "results.normalized.csv",
    )

    assert summary.first_pass_model_loads == 1
    assert summary.first_pass_metric_runs == 2
    assert summary.second_pass_model_loads == 0
    assert summary.second_pass_metric_runs == 0
    assert read_stable_rows(summary.normalized_path) == read_stable_rows(
        EXPECTED_RESULTS_PATH
    )

    with summary.results_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames

    assert fieldnames is not None
    assert STABLE_COLUMNS == fieldnames[: len(STABLE_COLUMNS)]
    assert "started_at" in fieldnames
    assert "duration_seconds" in fieldnames
