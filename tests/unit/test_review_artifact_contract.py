from pathlib import Path

from src.runner.json_source import JSONExperimentSource
from src.runner.results_store import ResultsStore


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_framework_baseline_config_expands_selected_metrics() -> None:
    experiments = JSONExperimentSource(
        str(REPO_ROOT / "data" / "experiments.framework-baselines.json")
    ).load()

    assert len(experiments) == 8
    assert {experiment.model_name for experiment in experiments} == {
        "timm-random::vit_base_patch16_224"
    }
    assert {
        experiment.experiment_id for experiment in experiments
    } == {
        "saliency-auc-judd-random-vit",
        "tid-spearman-mos-random-vit",
        "nights-preference-accuracy-random-vit",
        "things-triplet-accuracy-random-vit",
        "levels-between-class-random-vit",
        "levels-class-border-random-vit",
        "levels-within-class-random-vit",
        "psychophysics-weber-law-pearson-random-vit",
    }


def test_result_store_columns_match_technical_spec() -> None:
    assert ResultsStore.COLUMNS == [
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

