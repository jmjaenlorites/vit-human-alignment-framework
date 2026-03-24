import json
from types import SimpleNamespace

import pandas as pd

from src.runner.base import Runner
from src.runner.types import ResolvedExperiment


def test_runner_json_skips_done_runs_and_persists_new_results(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "experiments.json"
    results_path = tmp_path / "results.csv"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "levels-between",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "between_class"},
                    },
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "within_class"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "run_key": "vit-b16__levels-between",
                "model_name": "vit-b16",
                "experiment_id": "levels-between",
                "metric_name": "levels_triplet_accuracy",
                "status": "done",
                "result": "[0.4]",
                "error": None,
                "config_hash": ResolvedExperiment.from_parts(
                    experiment_id="levels-between",
                    metric_name="levels_triplet_accuracy",
                    model_name="vit-b16",
                    config={"split": "between_class"},
                ).config_hash,
            }
        ]
    ).to_csv(results_path, index=False)

    runner = Runner(json_path=str(config_path), results_path=str(results_path))

    model = SimpleNamespace(backend=runner.backend, transform=lambda batch: batch)
    captured_configs = []

    def fake_load_model(model_name: str):
        assert model_name == "vit-b16"
        return model

    def fake_run_metric_experiment(experiment, loaded_model):
        assert loaded_model is model
        captured_configs.append(experiment.config)
        return f"result-for-{experiment.experiment_id}"

    monkeypatch.setattr("src.runner.base.resolve_model", fake_load_model)
    monkeypatch.setattr(runner, "_run_metric_experiment", fake_run_metric_experiment)

    runner.execute()

    assert captured_configs == [{"split": "within_class"}]
    frame = pd.read_csv(results_path)
    assert set(frame["run_key"]) == {
        "vit-b16__levels-between",
        "vit-b16__levels-within",
    }
    assert (
        frame.loc[frame["run_key"] == "vit-b16__levels-within", "status"].item()
        == "done"
    )
    assert (
        frame.loc[frame["run_key"] == "vit-b16__levels-within", "result"].item()
        == "result-for-levels-within"
    )


def test_run_metric_experiment_passes_levels_configuration(monkeypatch) -> None:
    runner = Runner(json_path="experiments.json", results_path="results.csv")
    model = SimpleNamespace(backend=runner.backend, transform=lambda batch: batch)
    experiment = ResolvedExperiment.from_parts(
        experiment_id="levels-custom",
        metric_name="levels_triplet_accuracy",
        model_name="vit-b16",
        config={
            "split": "class_border",
            "levels_path": "/tmp/levels",
            "imagenet_path": "/tmp/imagenet",
        },
    )
    captured_kwargs = {}

    class FakeLevelsCalculator:
        def __init__(self, backend, split, metrics, levels_path, imagenet_path):
            captured_kwargs.update(
                {
                    "backend": backend,
                    "split": split,
                    "metrics": metrics,
                    "levels_path": levels_path,
                    "imagenet_path": imagenet_path,
                }
            )

        def run(self, loaded_model):
            assert loaded_model is model
            return {"levels_triplet_accuracy": "[0.9]"}

    monkeypatch.setattr("src.runner.base.LevelsMetricsCalculator", FakeLevelsCalculator)

    result = runner._run_metric_experiment(experiment, model)

    assert result == "[0.9]"
    assert captured_kwargs["split"] == "class_border"
    assert captured_kwargs["levels_path"] == "/tmp/levels"
    assert captured_kwargs["imagenet_path"] == "/tmp/imagenet"
    assert captured_kwargs["metrics"][0].name == "levels_triplet_accuracy"


def test_run_metric_experiment_passes_visturing_configuration(monkeypatch) -> None:
    runner = Runner(json_path="experiments.json", results_path="results.csv")
    model = SimpleNamespace(backend=runner.backend, transform=lambda batch: batch)
    experiment = ResolvedExperiment.from_parts(
        experiment_id="visturing-csf-rg",
        metric_name="visturing_csf_pearson",
        model_name="vit-b16",
        config={
            "channel": "red_green",
            "data_path": "/tmp/visturing",
            "gt_path": "/tmp/visturing-gt",
            "batch_size": 8,
        },
    )
    captured_kwargs = {}

    class FakeCalculator:
        def run(self, loaded_model):
            assert loaded_model is model
            return {"visturing_csf_pearson": "[0.7]"}

    def fake_factory(backend, channel, data_path, gt_path, batch_size, include_kendall):
        captured_kwargs.update(
            {
                "backend": backend,
                "channel": channel,
                "data_path": data_path,
                "gt_path": gt_path,
                "batch_size": batch_size,
                "include_kendall": include_kendall,
            }
        )
        return FakeCalculator()

    monkeypatch.setattr("src.runner.base.create_prop3_4_calculator", fake_factory)

    result = runner._run_metric_experiment(experiment, model)

    assert result == "[0.7]"
    assert captured_kwargs == {
        "backend": runner.backend,
        "channel": "red_green",
        "data_path": "/tmp/visturing",
        "gt_path": "/tmp/visturing-gt",
        "batch_size": 8,
        "include_kendall": False,
    }


def test_run_metric_experiment_does_not_instantiate_visturing_metric_directly(
    monkeypatch,
) -> None:
    runner = Runner(json_path="experiments.json", results_path="results.csv")
    model = SimpleNamespace(backend=runner.backend, transform=lambda batch: batch)
    experiment = ResolvedExperiment.from_parts(
        experiment_id="visturing-spectral",
        metric_name="visturing_spectral_sensitivity",
        model_name="vit-b16",
        config={
            "data_path": "/tmp/visturing",
            "gt_path": "/tmp/visturing-gt",
            "batch_size": 8,
        },
    )

    def failing_load_metric(metric_name: str):
        raise AssertionError(f"load_metric should not be called for {metric_name}")

    monkeypatch.setattr("src.runner.base.load_metric", failing_load_metric)
    monkeypatch.setattr(
        runner,
        "_run_visturing_metric",
        lambda resolved_experiment, loaded_model: "[0.5]",
    )

    result = runner._run_metric_experiment(experiment, model)

    assert result == "[0.5]"
