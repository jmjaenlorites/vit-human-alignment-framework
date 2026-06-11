import json

import pytest

from src.runner.json_source import JSONExperimentSource


@pytest.mark.parametrize(
    "model_name",
    [
        "saliency-baseline::uniform",
        "saliency-baseline::center-gaussian",
    ],
)
def test_json_source_accepts_saliency_baseline_models(
    tmp_path,
    model_name: str,
) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": [model_name],
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

    assert experiments[0].model_name == model_name


def test_json_source_rejects_unknown_saliency_baseline(tmp_path) -> None:
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps(
            {
                "models": ["saliency-baseline::unknown"],
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

    with pytest.raises(ValueError, match="saliency-baseline::uniform"):
        JSONExperimentSource(str(config_path)).load()
