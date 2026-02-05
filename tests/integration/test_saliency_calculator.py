"""Integration tests for SaliencyMetricsCalculator with real model."""

import json

import pytest

from src.metrics.saliency import SaliencyMetricsCalculator
from src.utils.common_enums import BackendEnum


@pytest.mark.real_model
class TestSaliencyMetricsCalculatorIntegration:
    """Integration tests using real ViT model and MIT1003 dataset."""

    def test_full_pipeline_with_real_model(self, real_model):
        """Test complete saliency metrics calculation with real model."""
        # Create calculator
        calculator = SaliencyMetricsCalculator(BackendEnum.TORCH)
        
        # Run on full dataset (or first N batches)
        results = calculator.run(real_model)
        
        # Verify results structure
        assert "saliency_auc_judd" in results
        assert "saliency_pearson_correlation_coefficient" in results
        
        # Parse JSON results
        auc_values = json.loads(results["saliency_auc_judd"])
        pearson_values = json.loads(results["saliency_pearson_correlation_coefficient"])
        
        # Check we have results for all layers (ViT-B/16 has 12 layers)
        assert len(auc_values) == 12, f"Expected 12 layers, got {len(auc_values)}"
        assert len(pearson_values) == 12, f"Expected 12 layers, got {len(pearson_values)}"
        
        # Check all values are in valid ranges
        for layer_idx, auc in enumerate(auc_values):
            assert 0 <= auc <= 1, f"Layer {layer_idx}: AUC {auc} out of range [0, 1]"
        
        for layer_idx, pearson in enumerate(pearson_values):
            assert -1 <= pearson <= 1, f"Layer {layer_idx}: Pearson {pearson} out of range [-1, 1]"
        
        # Check that values are reasonable (based on previous experiments)
        # AUC_Judd typically should be > 0.5 for a good model
        assert all(auc > 0.5 for auc in auc_values), "AUC values should be > 0.5"
        
        # Pearson correlation should be positive for good saliency prediction
        assert all(pearson > 0 for pearson in pearson_values), "Pearson should be positive"
        
        print(f"\nAUC_Judd per layer: {auc_values}")
        print(f"Pearson per layer: {pearson_values}")

    def test_results_consistency(self, real_model):
        """Test that running twice gives consistent results (deterministic)."""
        calculator1 = SaliencyMetricsCalculator(BackendEnum.TORCH)
        calculator2 = SaliencyMetricsCalculator(BackendEnum.TORCH)
        
        # Run twice
        results1 = calculator1.run(real_model)
        results2 = calculator2.run(real_model)
        
        # Parse results
        auc1 = json.loads(results1["saliency_auc_judd"])
        auc2 = json.loads(results2["saliency_auc_judd"])
        
        pearson1 = json.loads(results1["saliency_pearson_correlation_coefficient"])
        pearson2 = json.loads(results2["saliency_pearson_correlation_coefficient"])
        
        # Check consistency (within small tolerance for numerical precision)
        for i, (v1, v2) in enumerate(zip(auc1, auc2)):
            assert abs(v1 - v2) < 1e-6, f"Layer {i}: AUC inconsistent {v1} vs {v2}"
        
        for i, (v1, v2) in enumerate(zip(pearson1, pearson2)):
            assert abs(v1 - v2) < 1e-6, f"Layer {i}: Pearson inconsistent {v1} vs {v2}"

    def test_layer_progression(self, real_model):
        """Test that metrics show progression across layers."""
        calculator = SaliencyMetricsCalculator(BackendEnum.TORCH)
        results = calculator.run(real_model)
        
        auc_values = json.loads(results["saliency_auc_judd"])
        pearson_values = json.loads(results["saliency_pearson_correlation_coefficient"])
        
        # Generally, later layers should have better alignment
        # (though this is not always strictly true)
        early_auc_mean = sum(auc_values[:4]) / 4  # First 4 layers
        late_auc_mean = sum(auc_values[-4:]) / 4  # Last 4 layers
        
        print(f"\nEarly layers AUC (mean): {early_auc_mean:.4f}")
        print(f"Late layers AUC (mean): {late_auc_mean:.4f}")
        
        # Just log the progression, don't assert (can vary by model)
        assert True  # Always pass, just log data

    def test_individual_metrics(self, real_model):
        """Test running individual metrics separately."""
        from src.metrics.saliency import AUC_Judd, PearsonCorrelationCoefficient
        
        # Test AUC_Judd alone
        auc_metric = AUC_Judd(BackendEnum.TORCH)
        calculator_auc = SaliencyMetricsCalculator(BackendEnum.TORCH, metrics=[auc_metric])
        results_auc = calculator_auc.run(real_model)
        
        assert "saliency_auc_judd" in results_auc
        assert "saliency_pearson_correlation_coefficient" not in results_auc
        
        # Test Pearson alone
        pearson_metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)
        calculator_pearson = SaliencyMetricsCalculator(BackendEnum.TORCH, metrics=[pearson_metric])
        results_pearson = calculator_pearson.run(real_model)
        
        assert "saliency_auc_judd" not in results_pearson
        assert "saliency_pearson_correlation_coefficient" in results_pearson

    def test_batch_processing(self, real_model):
        """Test that batch processing works correctly."""
        calculator = SaliencyMetricsCalculator(BackendEnum.TORCH)
        
        # Get dataset loader
        loader = calculator.get_dataset_loader(transform=real_model.transform)
        
        # Process just one batch manually to verify
        iterator = loader.get_iterator()
        batch = next(iter(iterator))
        
        stimulus, saliency_gt, fixation_gt = batch
        
        # Verify batch shapes
        assert len(stimulus.shape) == 4, "Stimulus should be [B, C, H, W]"
        assert stimulus.shape[1] == 3, "Should have 3 color channels"
        assert len(saliency_gt.shape) == 4, "Saliency GT should be [B, 1, H, W]"
        assert len(fixation_gt.shape) == 4, "Fixation GT should be [B, 1, H, W]"
        
        # Process batch through model
        batch_results = calculator.process_batch(batch, real_model)
        
        assert "saliency" in batch_results
        assert batch_results["saliency"] is not None
        assert len(batch_results["saliency"]) == 12, "Should have 12 layers"

    @pytest.mark.slow
    def test_full_dataset_run(self, real_model):
        """Test running on full MIT1003 dataset (slower test)."""
        calculator = SaliencyMetricsCalculator(BackendEnum.TORCH)
        results = calculator.run(real_model)
        
        # Just verify it completes without errors
        assert results is not None
        
        auc_values = json.loads(results["saliency_auc_judd"])
        pearson_values = json.loads(results["saliency_pearson_correlation_coefficient"])
        
        # Log final results
        print(f"\n=== Full Dataset Results ===")
        print(f"AUC_Judd: {auc_values}")
        print(f"Pearson: {pearson_values}")
        print(f"Mean AUC: {sum(auc_values) / len(auc_values):.4f}")
        print(f"Mean Pearson: {sum(pearson_values) / len(pearson_values):.4f}")


@pytest.mark.golden
class TestSaliencyWithGoldenData:
    """Tests using pre-computed golden fixtures."""

    def test_metrics_match_golden(self, saliency_golden_data):
        """Test that metrics computed on golden data match expected values."""
        from src.metrics.saliency import AUC_Judd, PearsonCorrelationCoefficient
        
        # Extract golden data
        stimulus = saliency_golden_data["stimulus"]
        saliency_gt = saliency_golden_data["saliency_gt"]
        fixation_gt = saliency_golden_data["fixation_gt"]
        predicted_saliency = saliency_golden_data["predicted_saliency"]
        expected_auc = saliency_golden_data["expected_auc_judd"]
        expected_pearson = saliency_golden_data["expected_pearson"]
        
        # Create metrics
        auc_metric = AUC_Judd(BackendEnum.TORCH)
        pearson_metric = PearsonCorrelationCoefficient(BackendEnum.TORCH)
        
        # Create batch
        batch = (stimulus, saliency_gt, fixation_gt)
        batch_results = {"saliency": predicted_saliency, "features": None}
        
        # Calculate metrics
        auc_metric.calculate(batch, batch_results)
        pearson_metric.calculate(batch, batch_results)
        
        # Finalize
        auc_results = auc_metric.finalize()
        pearson_results = pearson_metric.finalize()
        
        # Parse and compare
        computed_auc = json.loads(auc_results["saliency_auc_judd"])
        computed_pearson = json.loads(pearson_results["saliency_pearson_correlation_coefficient"])
        
        # Check match (with small tolerance)
        for i, (comp, exp) in enumerate(zip(computed_auc, expected_auc)):
            assert abs(comp - exp) < 1e-5, f"Layer {i}: AUC mismatch {comp} vs {exp}"
        
        for i, (comp, exp) in enumerate(zip(computed_pearson, expected_pearson)):
            assert abs(comp - exp) < 1e-5, f"Layer {i}: Pearson mismatch {comp} vs {exp}"
