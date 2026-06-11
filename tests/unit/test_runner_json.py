import json
from types import SimpleNamespace

import pandas as pd

from src.runner.base import Runner, _is_out_of_memory_error
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


def test_is_out_of_memory_error_detects_jax_and_torch_oom() -> None:
    # JAX-style: la clase suele llamarse JaxRuntimeError o XlaRuntimeError.
    class JaxRuntimeError(RuntimeError):
        pass

    class XlaRuntimeError(RuntimeError):
        pass

    class OutOfMemoryError(RuntimeError):
        pass

    assert _is_out_of_memory_error(
        JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 902.67MiB.")
    )
    assert _is_out_of_memory_error(
        XlaRuntimeError("RESOURCE_EXHAUSTED: ...")
    )
    assert _is_out_of_memory_error(
        OutOfMemoryError("CUDA out of memory. Tried to allocate 5.00 GiB")
    )
    # RuntimeError genérico con mensaje de OOM (torch antiguo) también cuenta.
    assert _is_out_of_memory_error(RuntimeError("CUDA out of memory"))
    # Errores genuinos no-OOM no se confunden.
    assert not _is_out_of_memory_error(ValueError("something else"))
    assert not _is_out_of_memory_error(RuntimeError("totally unrelated"))


def test_runner_continues_after_oom_in_one_metric(tmp_path, monkeypatch) -> None:
    """Si una métrica revienta por OOM, marcamos error y seguimos con la
    siguiente, en vez de tumbar el run completo."""
    config_path = tmp_path / "experiments.json"
    results_path = tmp_path / "results.csv"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "first-ok",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "between_class"},
                    },
                    {
                        "experiment_id": "second-oom",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "within_class"},
                    },
                    {
                        "experiment_id": "third-ok",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "class_border"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = Runner(json_path=str(config_path), results_path=str(results_path))
    model = SimpleNamespace(
        name="vit-b16",
        backend=runner.backend,
        transform=lambda x: x,
        model={"params": object()},
    )
    monkeypatch.setattr("src.runner.base.resolve_model", lambda name: model)
    monkeypatch.setattr(runner, "_release_model", lambda m: None)

    class FakeOOM(RuntimeError):
        pass

    FakeOOM.__name__ = "JaxRuntimeError"

    def fake_run(experiment, m):
        if experiment.experiment_id == "second-oom":
            raise FakeOOM("RESOURCE_EXHAUSTED: Out of memory ...")
        return f"result-{experiment.experiment_id}"

    monkeypatch.setattr(runner, "_run_metric_experiment", fake_run)

    runner.execute()  # NO debe lanzar excepción

    frame = pd.read_csv(results_path)
    statuses = dict(zip(frame["experiment_id"], frame["status"]))
    assert statuses["first-ok"] == "done"
    assert statuses["second-oom"] == "error"
    assert statuses["third-ok"] == "done"
    second = frame.loc[frame["experiment_id"] == "second-oom"].iloc[0]
    assert "Out of memory" in str(second["error"])


def test_runner_reraises_non_oom_errors(tmp_path, monkeypatch) -> None:
    """Errores que NO son OOM siguen tumbando el run (comportamiento previo)."""
    config_path = tmp_path / "experiments.json"
    results_path = tmp_path / "results.csv"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "broken",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "between_class"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = Runner(json_path=str(config_path), results_path=str(results_path))
    model = SimpleNamespace(
        name="vit-b16", backend=runner.backend, transform=lambda x: x, model=None
    )
    monkeypatch.setattr("src.runner.base.resolve_model", lambda name: model)
    monkeypatch.setattr(runner, "_release_model", lambda m: None)
    monkeypatch.setattr(
        runner,
        "_run_metric_experiment",
        lambda *_: (_ for _ in ()).throw(ValueError("not oom")),
    )

    try:
        runner.execute()
    except ValueError as exc:
        assert "not oom" in str(exc)
    else:
        raise AssertionError("Runner should have raised ValueError")


def test_runner_releases_previous_model_when_switching_models(
    tmp_path, monkeypatch
) -> None:
    """Cuando el runner cambia de modelo (model-major iteration), debe liberar
    el anterior antes de cargar el nuevo para que torch y jax no se peleen
    por la VRAM."""
    config_path = tmp_path / "experiments.json"
    results_path = tmp_path / "results.csv"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16", "vit-l14"],
                "experiments": [
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

    loaded: list[str] = []
    released: list[str] = []

    def fake_resolve_model(model_name: str):
        loaded.append(model_name)
        return SimpleNamespace(
            name=model_name,
            backend=Runner().backend,
            transform=lambda x: x,
            model={"params": object()},
        )

    runner = Runner(json_path=str(config_path), results_path=str(results_path))
    monkeypatch.setattr("src.runner.base.resolve_model", fake_resolve_model)
    monkeypatch.setattr(
        runner,
        "_run_metric_experiment",
        lambda experiment, model: "[0.5]",
    )
    monkeypatch.setattr(
        runner, "_release_model", lambda model: released.append(model.name)
    )

    runner.execute()

    # 2 modelos, ambos cargados, ambos liberados (uno al cambiar de modelo y
    # el otro al terminar el run).
    assert loaded == ["vit-b16", "vit-l14"]
    assert released == ["vit-b16", "vit-l14"]


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
        "channel": "all",
        "data_path": "/tmp/visturing",
        "gt_path": "/tmp/visturing-gt",
        "batch_size": 8,
        "include_kendall": False,
    }


def test_run_metric_experiment_passes_saliency_configuration(monkeypatch) -> None:
    runner = Runner(json_path="experiments.json", results_path="results.csv")
    model = SimpleNamespace(backend=runner.backend, transform=lambda batch: batch)
    experiment = ResolvedExperiment.from_parts(
        experiment_id="saliency-auc",
        metric_name="saliency_auc_judd",
        model_name="vit-b16",
        config={
            "dataset_path": "/tmp/mit1003",
            "batch_size": 4,
        },
    )
    captured_kwargs = {}

    class FakeMetric:
        name = "saliency_auc_judd"

    class FakeSaliencyCalculator:
        def __init__(self, backend, metrics, dataset_path, batch_size):
            captured_kwargs.update(
                {
                    "backend": backend,
                    "metrics": metrics,
                    "dataset_path": dataset_path,
                    "batch_size": batch_size,
                }
            )

        def run(self, loaded_model):
            assert loaded_model is model
            return {"saliency_auc_judd": "[0.8]"}

    monkeypatch.setattr(
        "src.runner.base.load_metric", lambda metric_name: lambda backend: FakeMetric()
    )
    monkeypatch.setattr(
        "src.runner.base.SaliencyMetricsCalculator", FakeSaliencyCalculator
    )

    result = runner._run_metric_experiment(experiment, model)

    assert result == "[0.8]"
    assert captured_kwargs["backend"] == runner.backend
    assert captured_kwargs["dataset_path"] == "/tmp/mit1003"
    assert captured_kwargs["batch_size"] == 4
    assert captured_kwargs["metrics"][0].name == "saliency_auc_judd"


def test_run_metric_experiment_passes_things_configuration(monkeypatch) -> None:
    runner = Runner(json_path="experiments.json", results_path="results.csv")
    model = SimpleNamespace(backend=runner.backend, transform=lambda batch: batch)
    experiment = ResolvedExperiment.from_parts(
        experiment_id="things-triplet",
        metric_name="things_triplet_accuracy",
        model_name="vit-b16",
        config={
            "dataset_path": "/tmp/things",
            "batch_size": 7,
        },
    )
    captured_kwargs = {}

    class FakeMetric:
        name = "things_triplet_accuracy"

    class FakeThingsCalculator:
        def __init__(self, backend, metrics, dataset_path, batch_size):
            captured_kwargs.update(
                {
                    "backend": backend,
                    "metrics": metrics,
                    "dataset_path": dataset_path,
                    "batch_size": batch_size,
                }
            )

        def run(self, loaded_model):
            assert loaded_model is model
            return {"things_triplet_accuracy": "[0.5]"}

    monkeypatch.setattr(
        "src.runner.base.load_metric", lambda metric_name: lambda backend: FakeMetric()
    )
    monkeypatch.setattr("src.runner.base.ThingsMetricsCalculator", FakeThingsCalculator)

    result = runner._run_metric_experiment(experiment, model)

    assert result == "[0.5]"
    assert captured_kwargs["backend"] == runner.backend
    assert captured_kwargs["dataset_path"] == "/tmp/things"
    assert captured_kwargs["batch_size"] == 7
    assert captured_kwargs["metrics"][0].name == "things_triplet_accuracy"


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
