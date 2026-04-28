"""End-to-end con un modelo Google ViT (JAX/Flax) real.

Marcado `@pytest.mark.real_model @pytest.mark.slow`: descarga el checkpoint
Ti/16 más pequeño desde el bucket público de AugReg y corre `levels_triplet_accuracy`
sobre el dataset de tests para validar el flujo completo.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.runner.base import Runner

GOOGLE_TI16_FILENAME = (
    "Ti_16-i1k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.1-sd_0.1"
)


@pytest.mark.real_model
@pytest.mark.slow
def test_runner_json_executes_end_to_end_with_google_jax_vit(tmp_path) -> None:
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
                "models": [f"google::{GOOGLE_TI16_FILENAME}"],
                "experiments": [
                    {
                        "experiment_id": "levels-between-google-ti16",
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
    assert frame.loc[0, "model_name"] == f"google::{GOOGLE_TI16_FILENAME}"
    assert frame.loc[0, "experiment_id"] == "levels-between-google-ti16"
    assert frame.loc[0, "result"].startswith("[")

    first_results = frame.copy()
    runner.execute()
    rerun_frame = pd.read_csv(results_path)
    pd.testing.assert_frame_equal(first_results, rerun_frame)
