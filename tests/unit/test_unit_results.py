import csv
import json

import pandas as pd

from src.metrics.nights import PreferenceAccuracy
from src.runner.types import ResolvedExperiment
from src.runner.unit_results import UNIT_RESULT_COLUMNS, UnitResultsWriter
from src.utils.common_enums import BackendEnum


def make_experiment() -> ResolvedExperiment:
    return ResolvedExperiment.from_parts(
        experiment_id="nights-preference",
        metric_name="nights_preference_accuracy",
        model_name="timm-random::vit_base_patch16_224",
        config={"batch_size": 8},
    )


def test_unit_results_writer_uses_stable_schema_and_filename(tmp_path) -> None:
    experiment = make_experiment()
    writer = UnitResultsWriter(str(tmp_path))

    path = writer.write(
        experiment,
        [
            {
                "layer_index": 0,
                "unit_index": 1,
                "unit_value": 1,
                "is_correct": 1,
            }
        ],
    )

    assert path is not None
    assert path.name.startswith("timm-random_vit_base_patch16_224__nights-preference")
    assert path.name.endswith(f"{experiment.config_hash[:12]}.csv")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == UNIT_RESULT_COLUMNS
        rows = list(reader)

    assert rows[0]["run_key"] == experiment.run_key
    assert rows[0]["config_hash"] == experiment.config_hash
    assert rows[0]["unit_value"] == "1"
    assert rows[0]["is_correct"] == "1"


def test_disabled_unit_results_writer_is_noop(tmp_path) -> None:
    writer = UnitResultsWriter(None)

    assert writer.write(make_experiment(), [{"layer_index": 0}]) is None
    assert list(tmp_path.iterdir()) == []


def test_nights_unit_records_match_aggregate_accuracy() -> None:
    metric = PreferenceAccuracy(BackendEnum.TORCH)
    metric.correct_per_layer = [2, 1]
    metric.correct_values_per_layer = [[1, 1, 0], [1, 0, 0]]
    metric.prediction_values_per_layer = [
        ["left", "right", "left"],
        ["left", "right", "left"],
    ]
    metric.human_values_per_layer = [
        ["left", "right", "right"],
        ["left", "left", "right"],
    ]
    metric.total = 3

    aggregate = json.loads(metric.finalize()[metric.name])
    records = pd.DataFrame(metric.unit_records())

    assert aggregate == [2 / 3, 1 / 3]
    assert records.groupby("layer_index")["unit_value"].mean().tolist() == aggregate
    assert set(records["prediction"]) == {"left", "right"}
    assert set(records["human_value"]) == {"left", "right"}
