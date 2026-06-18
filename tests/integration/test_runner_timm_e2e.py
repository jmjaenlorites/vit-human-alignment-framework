import json
from pathlib import Path

import pandas as pd
import pytest

from src.runner.base import Runner


@pytest.mark.real_model
@pytest.mark.slow
def test_runner_json_executes_end_to_end_with_external_timm_vit(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "experiments.json"
    results_path = tmp_path / "results.csv"
    config_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "batch_size": 2,
                },
                "dataset_defaults": {
                    "levels": {
                        "levels_path": str(repo_root / "tests" / "data" / "levels"),
                        "imagenet_path": str(
                            repo_root / "tests" / "data" / "levels" / "images"
                        ),
                    }
                },
                "models": ["timm::vit_small_patch16_224"],
                "experiments": [
                    {
                        "experiment_id": "levels-between-external-timm-vit",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "between_class"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = Runner(json_path=str(config_path), results_path=str(results_path))
    runner.execute()

    frame = pd.read_csv(results_path)
    assert frame.loc[0, "status"] == "done"
    assert frame.loc[0, "model_name"] == "timm::vit_small_patch16_224"
    assert frame.loc[0, "experiment_id"] == "levels-between-external-timm-vit"
    assert frame.loc[0, "result"].startswith("[")

    first_results = frame.copy()
    runner.execute()
    rerun_frame = pd.read_csv(results_path)
    pd.testing.assert_frame_equal(first_results, rerun_frame)
