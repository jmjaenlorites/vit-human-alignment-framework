import pytest
import torch

from src.models.resolver import SALIENCY_BASELINE_MODEL_PREFIX, resolve_model
from src.models.saliency_baseline import DEFAULT_NUM_LAYERS


@pytest.mark.parametrize(
    "baseline_name",
    [
        "uniform",
        "center-gaussian",
    ],
)
def test_saliency_baselines_return_layerwise_maps(baseline_name: str) -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}{baseline_name}")
    batch = torch.zeros(2, 3, 224, 224)

    outputs = model.forward(
        batch,
        return_features=False,
        return_saliency=True,
    )

    saliency_maps = outputs["saliency"]
    assert outputs["features"] is None
    assert len(saliency_maps) == DEFAULT_NUM_LAYERS
    assert all(saliency_map.shape == (2, 14, 14) for saliency_map in saliency_maps)
    assert all(
        torch.allclose(
            saliency_map.amax(dim=(1, 2)),
            torch.ones(2, device=saliency_map.device),
        )
        for saliency_map in saliency_maps
    )


def test_uniform_baseline_returns_ones() -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}uniform")
    batch = torch.zeros(2, 3, 224, 224)

    saliency_map = model.forward(
        batch,
        return_features=False,
        return_saliency=True,
    )["saliency"][0]

    assert torch.equal(
        saliency_map,
        torch.ones(2, 14, 14, device=saliency_map.device),
    )


def test_center_gaussian_baseline_peaks_at_center() -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}center-gaussian")
    batch = torch.zeros(1, 3, 224, 224)

    saliency_map = model.forward(
        batch,
        return_features=False,
        return_saliency=True,
    )["saliency"][0][0]

    assert saliency_map[6, 6] == pytest.approx(1.0)
    assert saliency_map[0, 0] < saliency_map[6, 6]
    assert torch.equal(
        saliency_map,
        model.forward(batch, return_features=False, return_saliency=True)["saliency"][
            -1
        ][0],
    )


def test_saliency_baselines_honor_requested_layers() -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}uniform")
    batch = torch.zeros(1, 3, 224, 224)

    outputs = model.forward(
        batch,
        layers=[0, 3, 11],
        return_features=False,
        return_saliency=True,
    )

    assert len(outputs["saliency"]) == 3


def test_saliency_baselines_reject_feature_extraction() -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}uniform")
    batch = torch.zeros(1, 3, 224, 224)

    with pytest.raises(NotImplementedError, match="saliency-only"):
        model.forward(batch, return_features=True, return_saliency=False)

    with pytest.raises(NotImplementedError, match="saliency-only"):
        model.forward_features(batch)


def test_saliency_baselines_use_input_derived_patch_grid() -> None:
    model = resolve_model(f"{SALIENCY_BASELINE_MODEL_PREFIX}uniform")
    batch = torch.zeros(1, 3, 256, 128)

    outputs = model.forward(
        batch,
        return_features=False,
        return_saliency=True,
    )

    assert outputs["saliency"][0].shape == (1, 16, 8)
