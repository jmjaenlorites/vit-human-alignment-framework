import torch

from src.metrics.representation import representation_vector


def test_representation_vector_pooling_modes() -> None:
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    assert representation_vector(features, "flatten").tolist() == list(range(12))
    assert representation_vector(features, "cls").tolist() == [0.0, 1.0, 2.0, 3.0]
    assert representation_vector(features, "mean_patch").tolist() == [
        6.0,
        7.0,
        8.0,
        9.0,
    ]
