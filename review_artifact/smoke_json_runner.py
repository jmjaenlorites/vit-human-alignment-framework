"""Reviewer smoke for the JSON runner without model or dataset downloads."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner.base import Runner  # noqa: E402
from src.utils.common_enums import BackendEnum  # noqa: E402


ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ARTIFACT_DIR / "smoke_experiments.json"
EXPECTED_RESULTS_PATH = ARTIFACT_DIR / "expected_results.csv"
STABLE_COLUMNS = [
    "run_key",
    "model_name",
    "experiment_id",
    "metric_name",
    "status",
    "result",
    "error",
    "config_hash",
]
FAKE_RESULTS = {
    "levels-between-fixture": json.dumps([0.25, 0.5, 0.75]),
    "visturing-frequency-low": json.dumps([0.125, 0.25, 0.5]),
}


@dataclass(frozen=True)
class SmokeSummary:
    results_path: Path
    normalized_path: Path
    first_pass_model_loads: int
    first_pass_metric_runs: int
    second_pass_model_loads: int
    second_pass_metric_runs: int


class SmokeRunner(Runner):
    def __init__(
        self,
        *,
        json_path: str,
        results_path: str,
        metric_runs: list[str],
    ):
        super().__init__(json_path=json_path, results_path=results_path)
        self.metric_runs = metric_runs

    def _run_metric_experiment(self, experiment, model):
        if model.name != experiment.model_name:
            raise AssertionError(
                f"Loaded fake model {model.name!r} for {experiment.model_name!r}"
            )
        self.metric_runs.append(experiment.run_key)
        try:
            return FAKE_RESULTS[experiment.experiment_id]
        except KeyError as exc:
            raise AssertionError(
                f"No fake result registered for {experiment.experiment_id!r}"
            ) from exc

    def _release_model(self, model) -> None:
        return None


def run_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    results_path: Path | None = None,
    normalized_path: Path | None = None,
    expected_path: Path = EXPECTED_RESULTS_PATH,
    overwrite: bool = False,
    assert_expected: bool = True,
) -> SmokeSummary:
    if results_path is None:
        output_dir = Path(tempfile.mkdtemp(prefix="vit-review-smoke-"))
        results_path = output_dir / "results.csv"
    else:
        results_path = results_path.resolve()
        if results_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{results_path} already exists. Pass overwrite=True "
                    "or choose a new path."
                )
            results_path.unlink()

    if normalized_path is None:
        normalized_path = results_path.with_name(f"{results_path.stem}.normalized.csv")
    else:
        normalized_path = normalized_path.resolve()

    config_path = config_path.resolve()
    expected_path = expected_path.resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)

    first_model_loads, first_metric_runs = _execute_once(config_path, results_path)
    second_model_loads, second_metric_runs = _execute_once(config_path, results_path)

    write_stable_csv(results_path, normalized_path)
    if assert_expected:
        assert_matches_expected(normalized_path, expected_path)

    if first_model_loads != ["vit-b16"]:
        raise AssertionError(f"Unexpected first-pass model loads: {first_model_loads}")
    if len(first_metric_runs) != len(FAKE_RESULTS):
        raise AssertionError(f"Unexpected first-pass metric runs: {first_metric_runs}")
    if second_model_loads:
        raise AssertionError(f"Resume pass loaded models: {second_model_loads}")
    if second_metric_runs:
        raise AssertionError(f"Resume pass ran metrics: {second_metric_runs}")

    return SmokeSummary(
        results_path=results_path,
        normalized_path=normalized_path,
        first_pass_model_loads=len(first_model_loads),
        first_pass_metric_runs=len(first_metric_runs),
        second_pass_model_loads=len(second_model_loads),
        second_pass_metric_runs=len(second_metric_runs),
    )


def _execute_once(config_path: Path, results_path: Path) -> tuple[list[str], list[str]]:
    model_loads: list[str] = []
    metric_runs: list[str] = []

    def fake_resolve_model(model_name: str):
        model_loads.append(model_name)
        return SimpleNamespace(
            name=model_name,
            backend=BackendEnum.TORCH,
            transform=lambda batch: batch,
            model=None,
        )

    runner = SmokeRunner(
        json_path=str(config_path),
        results_path=str(results_path),
        metric_runs=metric_runs,
    )
    with patch("src.runner.base.resolve_model", fake_resolve_model):
        runner.execute()
    return model_loads, metric_runs


def read_stable_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {column: row.get(column, "") for column in STABLE_COLUMNS}
            for row in reader
        ]


def write_stable_csv(source_path: Path, destination_path: Path) -> None:
    rows = read_stable_rows(source_path)
    with destination_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assert_matches_expected(actual_path: Path, expected_path: Path) -> None:
    actual = read_stable_rows(actual_path)
    expected = read_stable_rows(expected_path)
    if actual != expected:
        raise AssertionError(
            "Smoke results do not match expected stable CSV.\n"
            f"Expected: {json.dumps(expected, indent=2)}\n"
            f"Actual: {json.dumps(actual, indent=2)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the lightweight reviewer JSON-runner smoke artifact."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the smoke experiment JSON.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional results.csv path. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--normalized",
        type=Path,
        default=None,
        help="Optional stable-column CSV output path.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=EXPECTED_RESULTS_PATH,
        help="Expected stable-column CSV to compare against.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --results if it already exists.",
    )
    parser.add_argument(
        "--no-assert-expected",
        action="store_true",
        help="Run the smoke without comparing against expected_results.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(
        config_path=args.config,
        results_path=args.results,
        normalized_path=args.normalized,
        expected_path=args.expected,
        overwrite=args.overwrite,
        assert_expected=not args.no_assert_expected,
    )
    print("Smoke artifact OK")
    print(f"  config: {args.config}")
    print(f"  results: {summary.results_path}")
    print(f"  normalized: {summary.normalized_path}")
    print(
        "  first pass: loaded fake model "
        f"{summary.first_pass_model_loads} time(s), ran "
        f"{summary.first_pass_metric_runs} metric(s)"
    )
    print(
        "  resume pass: loaded fake model "
        f"{summary.second_pass_model_loads} time(s), ran "
        f"{summary.second_pass_metric_runs} metric(s)"
    )
    print(f"  expected CSV: {args.expected}")


if __name__ == "__main__":
    main()
