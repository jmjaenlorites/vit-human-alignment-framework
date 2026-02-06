"""Unit tests for visturing metric components."""

import numpy as np
import torch

from src.metrics.visturing.distance_functions import (
    calculate_correlations,
    cosine_distance,
    euclidean_distance,
)
from src.metrics.visturing.prop1 import SpectralSensitivityPearson
from src.metrics.visturing.prop2 import WeberLawPearson
from src.metrics.visturing.prop3_4 import CSFPearson
from src.utils.common_enums import BackendEnum


class TestVisturingDistanceFunctions:
    """Tests for distance and correlation helpers."""

    def test_euclidean_distance_shape_and_range(self):
        features_test = torch.randn(4, 768).numpy()
        features_ref = torch.randn(1, 768).numpy()

        diffs = np.asarray(euclidean_distance(features_test, features_ref))

        assert diffs.shape == (4,)
        assert np.all(diffs >= 0)

    def test_cosine_distance_shape_and_range(self):
        features_test = torch.randn(4, 768).numpy()
        features_ref = torch.randn(1, 768).numpy()

        diffs = np.asarray(cosine_distance(features_test, features_ref))

        assert diffs.shape == (4,)
        assert np.all(diffs >= 0)
        assert np.all(diffs <= 2)

    def test_calculate_correlations_perfect_linear_relation(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

        correlations = calculate_correlations(x, y)

        assert set(correlations.keys()) == {"pearson", "kendall", "spearman"}
        assert correlations["pearson"] > 0.99
        assert correlations["kendall"] > 0.99
        assert correlations["spearman"] > 0.99


class TestVisturingMetricInitialization:
    """Tests for metric class initialization contract."""

    def test_prop1_metric_initializes_without_gt_files(self, tmp_path):
        gt_path = str(tmp_path / "visturing-empty")
        metric = SpectralSensitivityPearson(
            backend=BackendEnum.TORCH,
            gt_path=gt_path,
            lambdas=np.linspace(380, 720, 10),
        )

        assert metric.name == "visturing_spectral_sensitivity"
        assert metric.type == "visturing"

    def test_prop2_metric_initializes_with_current_signature(self, tmp_path):
        gt_path = str(tmp_path / "visturing-empty")
        metric = WeberLawPearson(
            backend=BackendEnum.TORCH,
            gt_path=gt_path,
            x_values_map={
                "achrom": np.linspace(0, 1, 5),
                "red_green": np.linspace(0, 1, 5),
                "yellow_blue": np.linspace(0, 1, 5),
            },
            levels_map={"achrom": 5, "red_green": 5, "yellow_blue": 5},
        )

        assert metric.name == "visturing_weber_law_pearson"
        assert metric.type == "visturing"

    def test_prop3_4_metric_initializes_with_current_signature(self, tmp_path):
        gt_path = str(tmp_path / "visturing-empty")
        metric = CSFPearson(
            backend=BackendEnum.TORCH,
            gt_path=gt_path,
            freqs=np.array([1.5, 3, 6, 12, 24], dtype=np.float32),
        )

        assert metric.name == "visturing_csf_pearson"
        assert metric.type == "visturing"
