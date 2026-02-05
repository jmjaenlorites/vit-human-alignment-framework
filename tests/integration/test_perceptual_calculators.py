"""Integration tests for perceptual metrics calculators with real model."""

import json

import pytest

from src.metrics.levels import LevelsMetricsCalculator
from src.metrics.nights import NightsMetricsCalculator
from src.metrics.tid import TIDMetricsCalculator
from src.utils.common_enums import BackendEnum


@pytest.mark.real_model
class TestTIDMetricsCalculatorIntegration:
    """Integration tests for TID calculator with real model and test data."""

    def test_full_pipeline_with_real_model(self, real_model, tid_test_data_path):
        """Test complete TID metrics calculation with real model and test data."""
        # Create calculator with test data path
        calculator = TIDMetricsCalculator(
            BackendEnum.TORCH, dataset_path=tid_test_data_path
        )
        
        # Run on test dataset
        results = calculator.run(real_model)
        
        # Verify results structure
        assert "tid_spearman_mos" in results
        
        # Parse JSON results
        spearman_values = json.loads(results["tid_spearman_mos"])
        
        # Check we have results for all layers (ViT-B/16 has 12 layers)
        assert len(spearman_values) == 12, f"Expected 12 layers, got {len(spearman_values)}"
        
        # Check all values are in valid range
        for layer_idx, spearman in enumerate(spearman_values):
            assert -1 <= spearman <= 1, f"Layer {layer_idx}: Spearman {spearman} out of range [-1, 1]"
        
        print(f"\nTID Spearman per layer: {spearman_values}")
        print(f"Mean Spearman: {sum(spearman_values) / len(spearman_values):.4f}")

    def test_results_consistency(self, real_model, tid_test_data_path):
        """Test that running twice gives consistent results."""
        calculator1 = TIDMetricsCalculator(
            BackendEnum.TORCH, dataset_path=tid_test_data_path
        )
        calculator2 = TIDMetricsCalculator(
            BackendEnum.TORCH, dataset_path=tid_test_data_path
        )
        
        # Run twice
        results1 = calculator1.run(real_model)
        results2 = calculator2.run(real_model)
        
        # Parse results
        spearman1 = json.loads(results1["tid_spearman_mos"])
        spearman2 = json.loads(results2["tid_spearman_mos"])
        
        # Check consistency
        for i, (v1, v2) in enumerate(zip(spearman1, spearman2)):
            assert abs(v1 - v2) < 1e-6, f"Layer {i}: Spearman inconsistent {v1} vs {v2}"

    def test_batch_processing(self, real_model, tid_test_data_path):
        """Test that batch processing works correctly."""
        calculator = TIDMetricsCalculator(
            BackendEnum.TORCH, dataset_path=tid_test_data_path
        )
        
        # Get dataset loader
        loader = calculator.get_dataset_loader(transform=real_model.transform)
        
        # Process just one batch
        iterator = loader.get_iterator()
        batch = next(iter(iterator))
        
        reference_images, distorted_images, mos_scores = batch
        
        # Verify batch shapes
        assert len(reference_images.shape) == 4, "Reference should be [B, C, H, W]"
        assert len(distorted_images.shape) == 4, "Distorted should be [B, C, H, W]"
        assert reference_images.shape == distorted_images.shape
        assert len(mos_scores) == reference_images.shape[0]
        
        # Process batch through calculator
        batch_results = calculator.process_batch(batch, real_model)
        
        assert "features_ref" in batch_results
        assert "features_dist" in batch_results
        assert len(batch_results["features_ref"]) == 12, "Should have 12 layers"
        assert len(batch_results["features_dist"]) == 12, "Should have 12 layers"


@pytest.mark.real_model
class TestLevelsMetricsCalculatorIntegration:
    """Integration tests for Levels calculator with real model and test data."""

    def test_full_pipeline_with_real_model(
        self, real_model, levels_test_data_path, levels_images_path
    ):
        """Test complete Levels metrics calculation with real model and test data."""
        # Create calculator with test data paths
        calculator = LevelsMetricsCalculator(
            BackendEnum.TORCH,
            split="between_class",
            levels_path=levels_test_data_path,
            imagenet_path=levels_images_path,
        )
        
        # Run on test dataset
        results = calculator.run(real_model)
        
        # Verify results structure
        assert "levels_triplet_accuracy" in results
        
        # Parse JSON results
        accuracy_values = json.loads(results["levels_triplet_accuracy"])
        
        # Check we have results for all layers
        assert len(accuracy_values) == 12, f"Expected 12 layers, got {len(accuracy_values)}"
        
        # Check all values are in valid range [0, 1]
        for layer_idx, accuracy in enumerate(accuracy_values):
            assert 0 <= accuracy <= 1, f"Layer {layer_idx}: Accuracy {accuracy} out of range [0, 1]"
        
        print(f"\nLevels Triplet Accuracy per layer: {accuracy_values}")
        print(f"Mean Accuracy: {sum(accuracy_values) / len(accuracy_values):.4f}")

    def test_results_consistency(
        self, real_model, levels_test_data_path, levels_images_path
    ):
        """Test that running twice gives consistent results."""
        calculator1 = LevelsMetricsCalculator(
            BackendEnum.TORCH,
            split="between_class",
            levels_path=levels_test_data_path,
            imagenet_path=levels_images_path,
        )
        calculator2 = LevelsMetricsCalculator(
            BackendEnum.TORCH,
            split="between_class",
            levels_path=levels_test_data_path,
            imagenet_path=levels_images_path,
        )
        
        # Run twice
        results1 = calculator1.run(real_model)
        results2 = calculator2.run(real_model)
        
        # Parse results
        acc1 = json.loads(results1["levels_triplet_accuracy"])
        acc2 = json.loads(results2["levels_triplet_accuracy"])
        
        # Check consistency
        for i, (v1, v2) in enumerate(zip(acc1, acc2)):
            assert abs(v1 - v2) < 1e-6, f"Layer {i}: Accuracy inconsistent {v1} vs {v2}"

    def test_batch_processing(
        self, real_model, levels_test_data_path, levels_images_path
    ):
        """Test that batch processing works correctly."""
        calculator = LevelsMetricsCalculator(
            BackendEnum.TORCH,
            split="between_class",
            levels_path=levels_test_data_path,
            imagenet_path=levels_images_path,
        )
        
        # Get dataset loader
        loader = calculator.get_dataset_loader(transform=real_model.transform)
        
        # Process just one batch
        iterator = loader.get_iterator()
        batch = next(iter(iterator))
        
        img1, img2, img3, selected, img1_names, img2_names, img3_names = batch
        
        # Verify batch shapes
        assert len(img1.shape) == 4, "img1 should be [B, C, H, W]"
        assert len(img2.shape) == 4, "img2 should be [B, C, H, W]"
        assert len(img3.shape) == 4, "img3 should be [B, C, H, W]"
        assert img1.shape == img2.shape == img3.shape
        
        # Process batch through calculator
        batch_results = calculator.process_batch(batch, real_model)
        
        assert "features_img1" in batch_results
        assert "features_img2" in batch_results
        assert "features_img3" in batch_results
        assert len(batch_results["features_img1"]) == 12, "Should have 12 layers"


@pytest.mark.real_model
class TestNightsMetricsCalculatorIntegration:
    """Integration tests for Nights calculator with real model and test data."""

    def test_full_pipeline_with_real_model(self, real_model, nights_test_data_path):
        """Test complete Nights metrics calculation with real model and test data."""
        # Create calculator with test data path
        calculator = NightsMetricsCalculator(
            BackendEnum.TORCH, dataset_path=nights_test_data_path
        )
        
        # Run on test dataset
        results = calculator.run(real_model)
        
        # Verify results structure
        assert "nights_preference_accuracy" in results
        
        # Parse JSON results
        accuracy_values = json.loads(results["nights_preference_accuracy"])
        
        # Check we have results for all layers
        assert len(accuracy_values) == 12, f"Expected 12 layers, got {len(accuracy_values)}"
        
        # Check all values are in valid range [0, 1]
        for layer_idx, accuracy in enumerate(accuracy_values):
            assert 0 <= accuracy <= 1, f"Layer {layer_idx}: Accuracy {accuracy} out of range [0, 1]"
        
        print(f"\nNights Preference Accuracy per layer: {accuracy_values}")
        print(f"Mean Accuracy: {sum(accuracy_values) / len(accuracy_values):.4f}")

    def test_results_consistency(self, real_model, nights_test_data_path):
        """Test that running twice gives consistent results."""
        calculator1 = NightsMetricsCalculator(
            BackendEnum.TORCH, dataset_path=nights_test_data_path
        )
        calculator2 = NightsMetricsCalculator(
            BackendEnum.TORCH, dataset_path=nights_test_data_path
        )
        
        # Run twice
        results1 = calculator1.run(real_model)
        results2 = calculator2.run(real_model)
        
        # Parse results
        acc1 = json.loads(results1["nights_preference_accuracy"])
        acc2 = json.loads(results2["nights_preference_accuracy"])
        
        # Check consistency
        for i, (v1, v2) in enumerate(zip(acc1, acc2)):
            assert abs(v1 - v2) < 1e-6, f"Layer {i}: Accuracy inconsistent {v1} vs {v2}"

    def test_batch_processing(self, real_model, nights_test_data_path):
        """Test that batch processing works correctly."""
        calculator = NightsMetricsCalculator(
            BackendEnum.TORCH, dataset_path=nights_test_data_path
        )
        
        # Get dataset loader
        loader = calculator.get_dataset_loader(transform=real_model.transform)
        
        # Process just one batch
        iterator = loader.get_iterator()
        batch = next(iter(iterator))
        
        reference, left, right, left_votes, right_votes = batch
        
        # Verify batch shapes
        assert len(reference.shape) == 4, "Reference should be [B, C, H, W]"
        assert len(left.shape) == 4, "Left should be [B, C, H, W]"
        assert len(right.shape) == 4, "Right should be [B, C, H, W]"
        assert reference.shape == left.shape == right.shape
        
        # Process batch through calculator
        batch_results = calculator.process_batch(batch, real_model)
        
        assert "features_ref" in batch_results
        assert "features_left" in batch_results
        assert "features_right" in batch_results
        assert len(batch_results["features_ref"]) == 12, "Should have 12 layers"


@pytest.mark.golden
class TestPerceptualWithGoldenData:
    """Tests using pre-computed golden fixtures for perceptual metrics."""

    def test_tid_matches_golden(self, tid_golden_data):
        """Test that TID metrics computed on golden data match expected values."""
        from src.metrics.tid import SpearmanCorrelationMOS, TIDMetricsCalculator
        
        # Extract golden data
        reference_images = tid_golden_data["reference_images"]
        distorted_images = tid_golden_data["distorted_images"]
        mos_scores = tid_golden_data["mos_scores"]
        features_ref = tid_golden_data["features_ref"]
        features_dist = tid_golden_data["features_dist"]
        expected_spearman = tid_golden_data["expected_spearman"]
        
        # Create metric
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        # Create batch
        batch = (reference_images, distorted_images, mos_scores)
        batch_results = {"features_ref": features_ref, "features_dist": features_dist}
        
        # Calculate metric
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        # Parse and compare
        computed_spearman = json.loads(results["tid_spearman_mos"])
        
        for i, (comp, exp) in enumerate(zip(computed_spearman, expected_spearman)):
            assert abs(comp - exp) < 1e-5, f"Layer {i}: Spearman mismatch {comp} vs {exp}"

    def test_levels_matches_golden(self, levels_golden_data):
        """Test that Levels metrics computed on golden data match expected values."""
        from src.metrics.levels import TripletAccuracy
        
        # Extract golden data
        img1 = levels_golden_data["img1"]
        img2 = levels_golden_data["img2"]
        img3 = levels_golden_data["img3"]
        selected = levels_golden_data["selected"]
        img1_names = levels_golden_data["img1_names"]
        img2_names = levels_golden_data["img2_names"]
        img3_names = levels_golden_data["img3_names"]
        features_img1 = levels_golden_data["features_img1"]
        features_img2 = levels_golden_data["features_img2"]
        features_img3 = levels_golden_data["features_img3"]
        expected_accuracy = levels_golden_data["expected_accuracy"]
        
        # Create metric
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        # Create batch
        batch = (img1, img2, img3, selected, img1_names, img2_names, img3_names)
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }
        
        # Calculate metric
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        # Parse and compare
        computed_accuracy = json.loads(results["levels_triplet_accuracy"])
        
        for i, (comp, exp) in enumerate(zip(computed_accuracy, expected_accuracy)):
            assert abs(comp - exp) < 1e-5, f"Layer {i}: Accuracy mismatch {comp} vs {exp}"

    def test_nights_matches_golden(self, nights_golden_data):
        """Test that Nights metrics computed on golden data match expected values."""
        from src.metrics.nights import PreferenceAccuracy
        
        # Extract golden data
        reference = nights_golden_data["reference"]
        left = nights_golden_data["left"]
        right = nights_golden_data["right"]
        left_votes = nights_golden_data["left_votes"]
        right_votes = nights_golden_data["right_votes"]
        features_ref = nights_golden_data["features_ref"]
        features_left = nights_golden_data["features_left"]
        features_right = nights_golden_data["features_right"]
        expected_accuracy = nights_golden_data["expected_accuracy"]
        
        # Create metric
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        # Create batch
        batch = (reference, left, right, left_votes, right_votes)
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        # Calculate metric
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        # Parse and compare
        computed_accuracy = json.loads(results["nights_preference_accuracy"])
        
        for i, (comp, exp) in enumerate(zip(computed_accuracy, expected_accuracy)):
            assert abs(comp - exp) < 1e-5, f"Layer {i}: Accuracy mismatch {comp} vs {exp}"
