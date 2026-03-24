import json
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as F

from src.runner.base import Runner
from src.utils.common_enums import BackendEnum


def identity_transform(batch):
    return F.resize(batch, [224, 224])


class FakeModel:
    def __init__(self):
        self.backend = BackendEnum.TORCH
        self.transform = identity_transform

    def forward(
        self,
        batch: torch.Tensor,
        return_features: bool = False,
        return_saliency: bool = False,
    ) -> dict[str, object]:
        batch_size = batch.shape[0]
        features = None
        if return_features:
            features = [
                torch.ones(batch_size, 8, 4, dtype=torch.float32) * (layer_idx + 1)
                for layer_idx in range(12)
            ]
        saliency = None
        if return_saliency:
            saliency = [
                torch.ones(batch_size, 14, 14, dtype=torch.float32) * (layer_idx + 1)
                for layer_idx in range(12)
            ]
        return {"features": features, "saliency": saliency}


def test_runner_json_executes_end_to_end_with_results_resume(
    tmp_path, monkeypatch
) -> None:
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
                    },
                    "visturing": {
                        "data_path": str(repo_root / "tests" / "data" / "visturing"),
                        "gt_path": str(repo_root / "tests" / "data" / "visturing"),
                    },
                },
                "models": ["vit-b16"],
                "experiments": [
                    {
                        "experiment_id": "levels-between",
                        "metric": "levels_triplet_accuracy",
                        "config": {"split": "between_class"},
                    },
                    {
                        "experiment_id": "visturing-spectral",
                        "metric": "visturing_spectral_sensitivity",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.runner.base.resolve_model", lambda _: FakeModel())

    runner = Runner(json_path=str(config_path), results_path=str(results_path))
    runner.execute()

    frame = pd.read_csv(results_path)
    assert set(frame["status"]) == {"done"}
    assert set(frame["experiment_id"]) == {"levels-between", "visturing-spectral"}
    assert frame["result"].notna().all()

    first_results = frame.copy()
    runner.execute()
    rerun_frame = pd.read_csv(results_path)
    pd.testing.assert_frame_equal(first_results, rerun_frame)
