"""Unit tests for Nights metrics."""

import json

import pytest
import torch

from src.metrics.nights import PreferenceAccuracy
from src.utils.common_enums import BackendEnum


class TestPreferenceAccuracy:
    """Tests for Preference Accuracy metric."""

    def test_perfect_prediction_left(self):
        """Test PreferenceAccuracy when left is always preferred and more similar."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 3
        hidden_dim = 768
        
        features_ref = []
        features_left = []
        features_right = []
        
        for layer_idx in range(num_layers):
            layer_feat_ref = []
            layer_feat_left = []
            layer_feat_right = []
            
            for i in range(num_samples):
                # Reference vector
                feat_ref = torch.randn(hidden_dim)
                
                # Left is very similar to reference
                feat_left = feat_ref + torch.randn(hidden_dim) * 0.01
                
                # Right is different from reference
                feat_right = torch.randn(hidden_dim)
                
                layer_feat_ref.append(feat_ref)
                layer_feat_left.append(feat_left)
                layer_feat_right.append(feat_right)
            
            features_ref.append(torch.stack(layer_feat_ref))
            features_left.append(torch.stack(layer_feat_left))
            features_right.append(torch.stack(layer_feat_right))
        
        # All votes for left (left=1, right=0)
        left_votes = [1] * num_samples
        right_votes = [0] * num_samples
        
        # Create batch
        batch = (
            torch.zeros(num_samples, 3, 224, 224),  # reference
            torch.zeros(num_samples, 3, 224, 224),  # left
            torch.zeros(num_samples, 3, 224, 224),  # right
            left_votes,
            right_votes,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        # Calculate
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        # Parse results
        accuracies = json.loads(results["nights_preference_accuracy"])
        
        # Should have perfect accuracy (1.0)
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected accuracy = 1.0, got {acc}"

    def test_perfect_prediction_right(self):
        """Test PreferenceAccuracy when right is always preferred and more similar."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_left = []
        features_right = []
        
        for layer_idx in range(num_layers):
            layer_feat_ref = []
            layer_feat_left = []
            layer_feat_right = []
            
            for i in range(num_samples):
                feat_ref = torch.randn(hidden_dim)
                
                # Left is different
                feat_left = torch.randn(hidden_dim)
                
                # Right is very similar to reference
                feat_right = feat_ref + torch.randn(hidden_dim) * 0.01
                
                layer_feat_ref.append(feat_ref)
                layer_feat_left.append(feat_left)
                layer_feat_right.append(feat_right)
            
            features_ref.append(torch.stack(layer_feat_ref))
            features_left.append(torch.stack(layer_feat_left))
            features_right.append(torch.stack(layer_feat_right))
        
        # All votes for right (left=0, right=1)
        left_votes = [0] * num_samples
        right_votes = [1] * num_samples
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            left_votes,
            right_votes,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["nights_preference_accuracy"])
        
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected accuracy = 1.0, got {acc}"

    def test_random_prediction(self):
        """Test PreferenceAccuracy with random features (should be ~50% accuracy)."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_samples = 30  # Need more samples for stable random estimate
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_left = []
        features_right = []
        
        for layer_idx in range(num_layers):
            # Completely random features
            features_ref.append(torch.randn(num_samples, hidden_dim))
            features_left.append(torch.randn(num_samples, hidden_dim))
            features_right.append(torch.randn(num_samples, hidden_dim))
        
        # Random votes
        import random
        left_votes = [random.choice([0, 1]) for _ in range(num_samples)]
        right_votes = [1 - v for v in left_votes]  # Complementary
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            left_votes,
            right_votes,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["nights_preference_accuracy"])
        
        # Should be around 0.5 (random chance)
        for layer_idx, acc in enumerate(accuracies):
            assert 0.2 < acc < 0.8, f"Layer {layer_idx}: Expected accuracy ≈ 0.5, got {acc}"

    def test_mixed_preferences(self):
        """Test with mixed preferences (some left, some right)."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_left = []
        features_right = []
        
        for layer_idx in range(num_layers):
            layer_feat_ref = []
            layer_feat_left = []
            layer_feat_right = []
            
            for i in range(num_samples):
                feat_ref = torch.randn(hidden_dim)
                
                if i < 5:
                    # First half: left more similar
                    feat_left = feat_ref + torch.randn(hidden_dim) * 0.01
                    feat_right = torch.randn(hidden_dim)
                else:
                    # Second half: right more similar
                    feat_left = torch.randn(hidden_dim)
                    feat_right = feat_ref + torch.randn(hidden_dim) * 0.01
                
                layer_feat_ref.append(feat_ref)
                layer_feat_left.append(feat_left)
                layer_feat_right.append(feat_right)
            
            features_ref.append(torch.stack(layer_feat_ref))
            features_left.append(torch.stack(layer_feat_left))
            features_right.append(torch.stack(layer_feat_right))
        
        # Votes match the similarity pattern
        left_votes = [1 if i < 5 else 0 for i in range(num_samples)]
        right_votes = [0 if i < 5 else 1 for i in range(num_samples)]
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            left_votes,
            right_votes,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["nights_preference_accuracy"])
        
        # Should have perfect accuracy since votes match similarities
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected accuracy = 1.0, got {acc}"

    def test_inverse_prediction(self):
        """Test when preferences are opposite of similarities."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_left = []
        features_right = []
        
        for layer_idx in range(num_layers):
            layer_feat_ref = []
            layer_feat_left = []
            layer_feat_right = []
            
            for i in range(num_samples):
                feat_ref = torch.randn(hidden_dim)
                
                # Left is more similar to reference
                feat_left = feat_ref + torch.randn(hidden_dim) * 0.01
                feat_right = torch.randn(hidden_dim)
                
                layer_feat_ref.append(feat_ref)
                layer_feat_left.append(feat_left)
                layer_feat_right.append(feat_right)
            
            features_ref.append(torch.stack(layer_feat_ref))
            features_left.append(torch.stack(layer_feat_left))
            features_right.append(torch.stack(layer_feat_right))
        
        # But votes are for right (opposite of similarity)
        left_votes = [0] * num_samples
        right_votes = [1] * num_samples
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            left_votes,
            right_votes,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["nights_preference_accuracy"])
        
        # Should have 0% accuracy
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 0.0, f"Layer {layer_idx}: Expected accuracy = 0.0, got {acc}"

    def test_accumulation_across_batches(self):
        """Test that metric accumulates correctly across batches."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_layers = 3
        hidden_dim = 768
        
        # First batch
        batch_size_1 = 5
        features_ref_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        features_left_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        features_right_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        
        batch_1 = (
            torch.zeros(batch_size_1, 3, 224, 224),
            torch.zeros(batch_size_1, 3, 224, 224),
            torch.zeros(batch_size_1, 3, 224, 224),
            [1] * batch_size_1,
            [0] * batch_size_1,
        )
        
        batch_results_1 = {
            "features_ref": features_ref_1,
            "features_left": features_left_1,
            "features_right": features_right_1,
        }
        
        metric.calculate(batch_1, batch_results_1)
        
        # Second batch
        batch_size_2 = 7
        features_ref_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        features_left_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        features_right_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        
        batch_2 = (
            torch.zeros(batch_size_2, 3, 224, 224),
            torch.zeros(batch_size_2, 3, 224, 224),
            torch.zeros(batch_size_2, 3, 224, 224),
            [0] * batch_size_2,
            [1] * batch_size_2,
        )
        
        batch_results_2 = {
            "features_ref": features_ref_2,
            "features_left": features_left_2,
            "features_right": features_right_2,
        }
        
        metric.calculate(batch_2, batch_results_2)
        
        # Check accumulation
        assert len(metric.correct_per_layer) == num_layers, f"Should have {num_layers} layers"
        assert metric.total == batch_size_1 + batch_size_2, "Should accumulate all samples"
        
        # Finalize
        results = metric.finalize()
        accuracies = json.loads(results["nights_preference_accuracy"])
        
        assert len(accuracies) == num_layers, f"Should have {num_layers} accuracy values"

    def test_reset(self):
        """Test that reset clears accumulated state."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        # Add some data
        features = [torch.randn(5, 768) for _ in range(2)]
        
        batch = (
            torch.zeros(5, 3, 224, 224),
            torch.zeros(5, 3, 224, 224),
            torch.zeros(5, 3, 224, 224),
            [1] * 5,
            [0] * 5,
        )
        
        batch_results = {
            "features_ref": features,
            "features_left": features,
            "features_right": features,
        }
        
        metric.calculate(batch, batch_results)
        
        assert len(metric.correct_per_layer) > 0, "Should have accumulated data"
        assert metric.total > 0, "Should have total count"
        
        # Reset
        metric.reset()
        
        assert len(metric.correct_per_layer) == 0, "Correct counts should be cleared"
        assert metric.total == 0, "Total should be reset"

    def test_tensor_votes(self):
        """Test with votes as tensors (common from DataLoader)."""
        metric = PreferenceAccuracy(BackendEnum.TORCH)
        
        num_samples = 5
        num_layers = 2
        hidden_dim = 768
        
        features_ref = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]
        features_left = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]
        features_right = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]
        
        # Votes as tensors
        left_votes = torch.tensor([1, 0, 1, 0, 1])
        right_votes = torch.tensor([0, 1, 0, 1, 0])
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            left_votes,
            right_votes,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_left": features_left,
            "features_right": features_right,
        }
        
        # Should work without errors
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        accuracies = json.loads(results["nights_preference_accuracy"])
        assert len(accuracies) == num_layers, "Should handle tensor votes"
