"""Unit tests for Levels metrics."""

import json

import pytest
import torch

from src.metrics.levels import TripletAccuracy
from src.utils.common_enums import BackendEnum


class TestTripletAccuracy:
    """Tests for Triplet Accuracy metric."""

    def test_perfect_prediction(self):
        """Test TripletAccuracy with perfect outlier prediction."""
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 3
        hidden_dim = 768
        
        features_img1 = []
        features_img2 = []
        features_img3 = []
        selected_batch = []
        img1_names = []
        img2_names = []
        img3_names = []
        
        for layer_idx in range(num_layers):
            layer_feat1 = []
            layer_feat2 = []
            layer_feat3 = []
            
            for i in range(num_samples):
                # Create triplets where img1 and img2 are similar, img3 is outlier
                base_vector = torch.randn(hidden_dim)
                
                feat1 = base_vector + torch.randn(hidden_dim) * 0.01
                feat2 = base_vector + torch.randn(hidden_dim) * 0.01
                feat3 = torch.randn(hidden_dim)  # Different (outlier)
                
                layer_feat1.append(feat1)
                layer_feat2.append(feat2)
                layer_feat3.append(feat3)
            
            features_img1.append(torch.stack(layer_feat1))
            features_img2.append(torch.stack(layer_feat2))
            features_img3.append(torch.stack(layer_feat3))
        
        # Selected should be img3 (the outlier) for all samples
        selected_batch = [f"img3_{i}.JPEG" for i in range(num_samples)]
        img1_names = [f"img1_{i}.JPEG" for i in range(num_samples)]
        img2_names = [f"img2_{i}.JPEG" for i in range(num_samples)]
        img3_names = [f"img3_{i}.JPEG" for i in range(num_samples)]
        
        # Create batch
        batch = (
            torch.zeros(num_samples, 3, 224, 224),  # img1
            torch.zeros(num_samples, 3, 224, 224),  # img2
            torch.zeros(num_samples, 3, 224, 224),  # img3
            selected_batch,
            img1_names,
            img2_names,
            img3_names,
        )
        
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }
        
        # Calculate
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        # Parse results
        accuracies = json.loads(results["levels_triplet_accuracy"])
        
        # Should have perfect accuracy (1.0)
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected accuracy = 1.0, got {acc}"

    def test_random_prediction(self):
        """Test TripletAccuracy with random features (should be ~33% accuracy)."""
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        num_samples = 30  # Need more samples for stable random estimate
        num_layers = 2
        hidden_dim = 768
        
        features_img1 = []
        features_img2 = []
        features_img3 = []
        
        for layer_idx in range(num_layers):
            # Completely random features (no clear outlier)
            features_img1.append(torch.randn(num_samples, hidden_dim))
            features_img2.append(torch.randn(num_samples, hidden_dim))
            features_img3.append(torch.randn(num_samples, hidden_dim))
        
        # Randomly selected images (uniform distribution)
        import random
        selected_batch = [random.choice([f"img{j}_{i}.JPEG" for j in [1, 2, 3]]) for i in range(num_samples)]
        img1_names = [f"img1_{i}.JPEG" for i in range(num_samples)]
        img2_names = [f"img2_{i}.JPEG" for i in range(num_samples)]
        img3_names = [f"img3_{i}.JPEG" for i in range(num_samples)]
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            selected_batch,
            img1_names,
            img2_names,
            img3_names,
        )
        
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["levels_triplet_accuracy"])
        
        # Should be around 1/3 (random chance)
        for layer_idx, acc in enumerate(accuracies):
            assert 0.1 < acc < 0.6, f"Layer {layer_idx}: Expected accuracy ≈ 0.33, got {acc}"

    def test_img1_outlier(self):
        """Test when img1 is consistently the outlier."""
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_img1 = []
        features_img2 = []
        features_img3 = []
        
        for layer_idx in range(num_layers):
            layer_feat1 = []
            layer_feat2 = []
            layer_feat3 = []
            
            for i in range(num_samples):
                # img2 and img3 are similar, img1 is outlier
                base_vector = torch.randn(hidden_dim)
                
                feat1 = torch.randn(hidden_dim)  # Outlier
                feat2 = base_vector + torch.randn(hidden_dim) * 0.01
                feat3 = base_vector + torch.randn(hidden_dim) * 0.01
                
                layer_feat1.append(feat1)
                layer_feat2.append(feat2)
                layer_feat3.append(feat3)
            
            features_img1.append(torch.stack(layer_feat1))
            features_img2.append(torch.stack(layer_feat2))
            features_img3.append(torch.stack(layer_feat3))
        
        # Selected should be img1
        selected_batch = [f"img1_{i}.JPEG" for i in range(num_samples)]
        img1_names = [f"img1_{i}.JPEG" for i in range(num_samples)]
        img2_names = [f"img2_{i}.JPEG" for i in range(num_samples)]
        img3_names = [f"img3_{i}.JPEG" for i in range(num_samples)]
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            selected_batch,
            img1_names,
            img2_names,
            img3_names,
        )
        
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["levels_triplet_accuracy"])
        
        # Should have perfect accuracy
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected accuracy = 1.0, got {acc}"

    def test_img2_outlier(self):
        """Test when img2 is consistently the outlier."""
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_img1 = []
        features_img2 = []
        features_img3 = []
        
        for layer_idx in range(num_layers):
            layer_feat1 = []
            layer_feat2 = []
            layer_feat3 = []
            
            for i in range(num_samples):
                # img1 and img3 are similar, img2 is outlier
                base_vector = torch.randn(hidden_dim)
                
                feat1 = base_vector + torch.randn(hidden_dim) * 0.01
                feat2 = torch.randn(hidden_dim)  # Outlier
                feat3 = base_vector + torch.randn(hidden_dim) * 0.01
                
                layer_feat1.append(feat1)
                layer_feat2.append(feat2)
                layer_feat3.append(feat3)
            
            features_img1.append(torch.stack(layer_feat1))
            features_img2.append(torch.stack(layer_feat2))
            features_img3.append(torch.stack(layer_feat3))
        
        selected_batch = [f"img2_{i}.JPEG" for i in range(num_samples)]
        img1_names = [f"img1_{i}.JPEG" for i in range(num_samples)]
        img2_names = [f"img2_{i}.JPEG" for i in range(num_samples)]
        img3_names = [f"img3_{i}.JPEG" for i in range(num_samples)]
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            selected_batch,
            img1_names,
            img2_names,
            img3_names,
        )
        
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["levels_triplet_accuracy"])
        
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected accuracy = 1.0, got {acc}"

    def test_accumulation_across_batches(self):
        """Test that metric accumulates correctly across batches."""
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        num_layers = 3
        hidden_dim = 768
        
        # First batch
        batch_size_1 = 5
        features_img1_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        features_img2_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        features_img3_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        
        batch_1 = (
            torch.zeros(batch_size_1, 3, 224, 224),
            torch.zeros(batch_size_1, 3, 224, 224),
            torch.zeros(batch_size_1, 3, 224, 224),
            [f"img1_{i}.JPEG" for i in range(batch_size_1)],
            [f"img1_{i}.JPEG" for i in range(batch_size_1)],
            [f"img2_{i}.JPEG" for i in range(batch_size_1)],
            [f"img3_{i}.JPEG" for i in range(batch_size_1)],
        )
        
        batch_results_1 = {
            "features_img1": features_img1_1,
            "features_img2": features_img2_1,
            "features_img3": features_img3_1,
        }
        
        metric.calculate(batch_1, batch_results_1)
        
        # Second batch
        batch_size_2 = 7
        features_img1_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        features_img2_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        features_img3_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        
        batch_2 = (
            torch.zeros(batch_size_2, 3, 224, 224),
            torch.zeros(batch_size_2, 3, 224, 224),
            torch.zeros(batch_size_2, 3, 224, 224),
            [f"img2_{i}.JPEG" for i in range(batch_size_2)],
            [f"img1_{i}.JPEG" for i in range(batch_size_2)],
            [f"img2_{i}.JPEG" for i in range(batch_size_2)],
            [f"img3_{i}.JPEG" for i in range(batch_size_2)],
        )
        
        batch_results_2 = {
            "features_img1": features_img1_2,
            "features_img2": features_img2_2,
            "features_img3": features_img3_2,
        }
        
        metric.calculate(batch_2, batch_results_2)
        
        # Check accumulation
        assert len(metric.correct_per_layer) == num_layers, f"Should have {num_layers} layers"
        assert metric.total == batch_size_1 + batch_size_2, "Should accumulate all samples"
        
        # Finalize
        results = metric.finalize()
        accuracies = json.loads(results["levels_triplet_accuracy"])
        
        assert len(accuracies) == num_layers, f"Should have {num_layers} accuracy values"

    def test_reset(self):
        """Test that reset clears accumulated state."""
        metric = TripletAccuracy(BackendEnum.TORCH)
        
        # Add some data
        features = [torch.randn(5, 768) for _ in range(2)]
        
        batch = (
            torch.zeros(5, 3, 224, 224),
            torch.zeros(5, 3, 224, 224),
            torch.zeros(5, 3, 224, 224),
            ["img1_0.JPEG"] * 5,
            ["img1_0.JPEG"] * 5,
            ["img2_0.JPEG"] * 5,
            ["img3_0.JPEG"] * 5,
        )
        
        batch_results = {
            "features_img1": features,
            "features_img2": features,
            "features_img3": features,
        }
        
        metric.calculate(batch, batch_results)
        
        assert len(metric.correct_per_layer) > 0, "Should have accumulated data"
        assert metric.total > 0, "Should have total count"
        
        # Reset
        metric.reset()
        
        assert len(metric.correct_per_layer) == 0, "Correct counts should be cleared"
        assert metric.total == 0, "Total should be reset"
