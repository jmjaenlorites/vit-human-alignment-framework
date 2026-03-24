"""Unit tests for saliency metrics."""

import pytest
import torch

from src.metrics.saliency import (
    AUC_Judd,
    PearsonCorrelationCoefficient,
    SaliencyMetricsCalculator,
)
from src.utils.common_enums import BackendEnum


class TestAUCJudd:
    """Tests for AUC_Judd metric."""

    def test_perfect_prediction(self):
        """Test AUC_Judd with perfect prediction (identical to GT fixation map)."""
        metric = AUC_Judd(BackendEnum.TORCH)

        # Create a 14x14 prediction map
        pred_map = torch.zeros(14, 14)
        pred_map[5:9, 5:9] = 1.0  # High attention in center

        # Ground truth saliency (not used in AUC_Judd, but required)
        saliency_gt = pred_map.clone()

        # Fixation map: binary, same high region
        fixation_gt = torch.zeros(14, 14)
        fixation_gt[5:9, 5:9] = 1.0

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Perfect prediction should give AUC close to 1.0
        assert result.item() > 0.95, f"Expected AUC > 0.95, got {result.item()}"

    def test_inverse_prediction(self):
        """Test AUC_Judd with inverse prediction (low where fixations are)."""
        metric = AUC_Judd(BackendEnum.TORCH)

        # Prediction: high outside center
        pred_map = torch.ones(14, 14)
        pred_map[5:9, 5:9] = 0.0  # Low attention in center

        saliency_gt = torch.zeros(14, 14)

        # Fixation map: high in center
        fixation_gt = torch.zeros(14, 14)
        fixation_gt[5:9, 5:9] = 1.0

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Inverse prediction should give AUC close to 0.0
        assert result.item() < 0.05, f"Expected AUC < 0.05, got {result.item()}"

    def test_random_prediction(self):
        """Test AUC_Judd with uniform random prediction."""
        metric = AUC_Judd(BackendEnum.TORCH)

        # Uniform prediction
        pred_map = torch.ones(14, 14) * 0.5
        saliency_gt = torch.zeros(14, 14)

        # Fixation map: some fixations
        fixation_gt = torch.zeros(14, 14)
        fixation_gt[2, 3] = 1.0
        fixation_gt[7, 8] = 1.0
        fixation_gt[11, 12] = 1.0

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Random/uniform should give AUC around 0.5
        assert 0.45 < result.item() < 0.55, f"Expected AUC ≈ 0.5, got {result.item()}"

    def test_no_fixations(self):
        """Test AUC_Judd with no fixations returns NaN."""
        metric = AUC_Judd(BackendEnum.TORCH)

        pred_map = torch.rand(14, 14)
        saliency_gt = torch.zeros(14, 14)
        fixation_gt = torch.zeros(14, 14)  # No fixations

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        assert torch.isnan(result), "Expected NaN for no fixations"

    def test_accumulation(self):
        """Test that AUC_Judd correctly accumulates values across batches."""
        metric = AUC_Judd(BackendEnum.TORCH)

        # Create a simple batch
        batch_size = 4
        saliency_maps = [torch.rand(batch_size, 14, 14) for _ in range(3)]  # 3 layers

        # Create ground truth
        saliency_gt = torch.rand(batch_size, 1, 14, 14)
        fixation_gt = torch.zeros(batch_size, 1, 14, 14)
        fixation_gt[:, 0, 5:9, 5:9] = 1.0  # Center fixations

        stimulus = torch.rand(batch_size, 3, 224, 224)
        batch = (stimulus, saliency_gt, fixation_gt)

        batch_results = {"saliency": saliency_maps, "features": None}

        # Calculate
        metric.calculate(batch, batch_results)

        # Check accumulation
        assert len(metric.accumulated_values) == 3, "Should have 3 layers"
        assert len(metric.accumulated_values[0]) == batch_size, (
            f"Should have {batch_size} values in layer 0"
        )

        # Finalize
        results = metric.finalize()

        assert "saliency_auc_judd" in results
        import json

        values = json.loads(results["saliency_auc_judd"])
        assert len(values) == 3, "Should have 3 layer values"
        assert all(0 <= v <= 1 for v in values), "All AUC values should be in [0, 1]"


class TestPearsonCorrelationCoefficient:
    """Tests for Pearson Correlation Coefficient metric."""

    def test_perfect_correlation(self):
        """Test Pearson with perfect positive correlation."""
        metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)

        # Identical maps
        pred_map = torch.rand(14, 14)
        saliency_gt = pred_map.clone()
        fixation_gt = torch.zeros(14, 14)  # Not used in Pearson

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Perfect correlation should be 1.0
        assert abs(result.item() - 1.0) < 1e-5, (
            f"Expected correlation = 1.0, got {result.item()}"
        )

    def test_inverse_correlation(self):
        """Test Pearson with perfect negative correlation."""
        metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)

        # Create map with known range
        pred_map = torch.linspace(0, 1, 196).reshape(14, 14)
        saliency_gt = 1.0 - pred_map  # Perfect inverse
        fixation_gt = torch.zeros(14, 14)

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Perfect negative correlation should be -1.0
        assert abs(result.item() + 1.0) < 1e-5, (
            f"Expected correlation = -1.0, got {result.item()}"
        )

    def test_no_correlation(self):
        """Test Pearson with uncorrelated maps."""
        metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)

        # Create orthogonal patterns
        pred_map = torch.zeros(14, 14)
        pred_map[:, :7] = 1.0  # Left half high

        saliency_gt = torch.zeros(14, 14)
        saliency_gt[:7, :] = 1.0  # Top half high

        fixation_gt = torch.zeros(14, 14)

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Should be close to 0 (some small correlation due to overlap)
        assert abs(result.item()) < 0.5, (
            f"Expected correlation ≈ 0, got {result.item()}"
        )

    def test_constant_maps(self):
        """Test Pearson with constant maps returns NaN."""
        metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)

        # Both constant (no variance)
        pred_map = torch.ones(14, 14) * 0.5
        saliency_gt = torch.ones(14, 14) * 0.7
        fixation_gt = torch.zeros(14, 14)

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        assert torch.isnan(result), "Expected NaN for constant maps"

    def test_accumulation(self):
        """Test that Pearson correctly accumulates values across batches."""
        metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)

        # Create a batch
        batch_size = 4
        saliency_maps = [torch.rand(batch_size, 14, 14) for _ in range(3)]  # 3 layers

        # Create correlated ground truth
        saliency_gt = torch.rand(batch_size, 1, 14, 14)
        fixation_gt = torch.zeros(batch_size, 1, 14, 14)

        stimulus = torch.rand(batch_size, 3, 224, 224)
        batch = (stimulus, saliency_gt, fixation_gt)

        batch_results = {"saliency": saliency_maps, "features": None}

        # Calculate
        metric.calculate(batch, batch_results)

        # Check accumulation
        assert len(metric.accumulated_values) == 3, "Should have 3 layers"
        assert len(metric.accumulated_values[0]) == batch_size, (
            f"Should have {batch_size} values"
        )

        # Finalize
        results = metric.finalize()

        assert "saliency_pearson_correlation_coefficient" in results
        import json

        values = json.loads(results["saliency_pearson_correlation_coefficient"])
        assert len(values) == 3, "Should have 3 layer values"
        assert all(-1 <= v <= 1 for v in values), (
            "All correlation values should be in [-1, 1]"
        )

    def test_multi_dimensional_squeeze(self):
        """Test that multi-dimensional inputs are correctly squeezed."""
        metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)

        # Add extra dimensions
        pred_map = torch.rand(1, 1, 14, 14).squeeze()  # Should work
        saliency_gt = torch.rand(1, 14, 14)
        fixation_gt = torch.zeros(14, 14)

        result = metric._calculate_torch(pred_map, saliency_gt, fixation_gt)

        # Should work without errors
        assert not torch.isnan(result) or result.item() != 0, (
            "Should calculate without dimension errors"
        )


def test_saliency_calculator_uses_configurable_loader_arguments(monkeypatch) -> None:
    captured_kwargs = {}

    class FakeLoader:
        def __init__(
            self,
            batch_size,
            shuffle,
            num_workers,
            transform=None,
            dataset_path=None,
        ):
            captured_kwargs.update(
                {
                    "batch_size": batch_size,
                    "shuffle": shuffle,
                    "num_workers": num_workers,
                    "transform": transform,
                    "dataset_path": dataset_path,
                }
            )

    monkeypatch.setattr(
        "src.metrics.saliency.SaliencyMIT1003TorchDatasetLoader", FakeLoader
    )

    calculator = SaliencyMetricsCalculator(
        BackendEnum.TORCH,
        dataset_path="/tmp/mit1003",
        batch_size=6,
        num_workers=1,
    )
    transform = lambda batch: batch

    loader = calculator.get_dataset_loader(transform=transform)

    assert isinstance(loader, FakeLoader)
    assert captured_kwargs == {
        "batch_size": 6,
        "shuffle": False,
        "num_workers": 1,
        "transform": transform,
        "dataset_path": "/tmp/mit1003",
    }
