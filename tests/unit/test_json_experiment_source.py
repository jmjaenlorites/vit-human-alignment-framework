import json

import pytest

from src.runner.json_source import JSONExperimentSource


def test_json_source_expands_models_and_merges_defaults(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "levels_path": "/tmp/levels",
                    "batch_size": 16,
                },
                "models": ["vit-b16", "vit-l14"],
                "experiments": [
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                        "config": {
                            "split": "within_class",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    experiments = JSONExperimentSource(str(config_path)).load()

    assert [experiment.model_name for experiment in experiments] == [
        "vit-b16",
        "vit-l14",
    ]
    assert all(
        experiment.metric_name == "levels_triplet_accuracy"
        for experiment in experiments
    )
    assert all(
        experiment.experiment_id == "levels-within" for experiment in experiments
    )
    assert all(
        experiment.config["split"] == "within_class" for experiment in experiments
    )
    assert all(
        experiment.config["levels_path"] == "/tmp/levels" for experiment in experiments
    )
    assert all(experiment.config["batch_size"] == 16 for experiment in experiments)
    assert experiments[0].run_key == "vit-b16__levels-within"
    assert experiments[1].run_key == "vit-l14__levels-within"


def test_json_source_merges_dataset_defaults_by_metric_family(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "batch_size": 16,
                },
                "dataset_defaults": {
                    "levels": {
                        "levels_path": "/tmp/levels",
                        "imagenet_path": "/tmp/imagenet",
                    },
                    "visturing": {
                        "data_path": "/tmp/visturing",
                        "gt_path": "/tmp/visturing-gt",
                    },
                },
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "within_class"},
                    },
                    {
                        "experiment_id": "visturing-csf",
                        "metric": "visturing_csf_pearson",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    experiments = JSONExperimentSource(str(config_path)).load()

    levels_experiment = next(
        experiment
        for experiment in experiments
        if experiment.experiment_id == "levels-within"
    )
    visturing_experiment = next(
        experiment
        for experiment in experiments
        if experiment.experiment_id == "visturing-csf"
    )

    assert levels_experiment.config == {
        "batch_size": 16,
        "levels_path": "/tmp/levels",
        "imagenet_path": "/tmp/imagenet",
        "split": "within_class",
    }
    assert visturing_experiment.config == {
        "batch_size": 16,
        "data_path": "/tmp/visturing",
        "gt_path": "/tmp/visturing-gt",
    }


def test_json_source_merges_saliency_dataset_defaults(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_defaults": {
                    "saliency": {
                        "dataset_path": "/tmp/mit1003",
                        "batch_size": 4,
                    }
                },
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "saliency-auc",
                        "metric": "saliency_auc_judd",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    experiments = JSONExperimentSource(str(config_path)).load()

    assert experiments[0].config == {
        "dataset_path": "/tmp/mit1003",
        "batch_size": 4,
    }


def test_json_source_rejects_unknown_dataset_defaults_group(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_defaults": {
                    "foo": {"path": "/tmp/foo"},
                },
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                        "config": {
                            "split": "within_class",
                            "levels_path": "/tmp/levels",
                            "imagenet_path": "/tmp/imagenet",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset_defaults"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_requires_experiment_id_and_metric(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [{"config": {"split": "within_class"}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment_id"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_generates_stable_config_hash_for_equivalent_dict_order(
    tmp_path,
) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16", "vit-b32"],
                "experiments": [
                    {
                        "experiment_id": "visturing-csf",
                        "metric": "visturing_csf_pearson",
                        "config": {
                            "batch_size": 4,
                            "data_path": "/tmp/visturing",
                        },
                    },
                    {
                        "experiment_id": "visturing-csf-2",
                        "metric": "visturing_csf_pearson",
                        "config": {
                            "data_path": "/tmp/visturing",
                            "batch_size": 4,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    experiments = JSONExperimentSource(str(config_path)).load()

    assert experiments[0].config_hash == experiments[2].config_hash


def test_json_source_rejects_unknown_model(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-does-not-exist"],
                "experiments": [
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Valid options:.*vit-b16"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_accepts_timm_prefixed_model(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["timm::vit_base_patch16_224"],
                "experiments": [
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                        "config": {
                            "split": "within_class",
                            "levels_path": "/tmp/levels",
                            "imagenet_path": "/tmp/imagenet",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224"],
    )

    experiments = JSONExperimentSource(str(config_path)).load()

    assert experiments[0].model_name == "timm::vit_base_patch16_224"


def test_json_source_accepts_timm_prefixed_pretrained_tag_model(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["timm::vit_base_patch16_224.mae"],
                "experiments": [
                    {
                        "experiment_id": "levels-within",
                        "metric": "levels_triplet_accuracy",
                        "config": {
                            "split": "within_class",
                            "levels_path": "/tmp/levels",
                            "imagenet_path": "/tmp/imagenet",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.models.resolver.list_available_timm_models",
        lambda: ["vit_base_patch16_224.mae"],
    )

    experiments = JSONExperimentSource(str(config_path)).load()

    assert experiments[0].model_name == "timm::vit_base_patch16_224.mae"


def test_json_source_rejects_unknown_metric(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "unknown-metric",
                        "metric": "unknown_metric",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Valid options:.*levels_triplet_accuracy"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_rejects_invalid_levels_split(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "levels-invalid-split",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "bad_split"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Valid options:.*between_class"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_rejects_unknown_levels_config_key(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "levels-invalid-key",
                        "metric": "levels_triplet_accuracy",
                        "config": {"channel": "red_green"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Allowed keys:.*split"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_rejects_channel_for_csf_metric(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "visturing-invalid-channel",
                        "metric": "visturing_csf_pearson",
                        "config": {"channel": "red_green"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Allowed keys:.*batch_size"):
        JSONExperimentSource(str(config_path)).load()


def test_json_source_rejects_unknown_visturing_config_key(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "visturing-invalid-key",
                        "metric": "visturing_spectral_sensitivity",
                        "config": {"split": "between_class"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Allowed keys:.*data_path"):
        JSONExperimentSource(str(config_path)).load()
