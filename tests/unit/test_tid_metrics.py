"""Unit tests for TID metrics."""

import json

import pytest
import torch

from src.metrics.tid import SpearmanCorrelationMOS
from src.utils.common_enums import BackendEnum


class TestSpearmanCorrelationMOS:
    """Tests for Spearman Correlation with MOS scores."""

    def test_perfect_positive_correlation(self):
        """Test Spearman with perfect positive correlation between similarity and MOS."""
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        # Create mock features where similarity increases with index
        # This simulates better quality (higher MOS) having higher similarity
        num_samples = 10
        num_layers = 3
        hidden_dim = 768
        
        features_ref = []
        features_dist = []
        mos_scores = []
        
        for layer_idx in range(num_layers):
            layer_features_ref = []
            layer_features_dist = []
            
            for i in range(num_samples):
                # Reference: constant vector
                feat_ref = torch.ones(hidden_dim)
                
                # Distorted: increasingly similar to reference
                # Higher index = higher similarity = should correspond to higher MOS
                similarity_factor = i / num_samples
                feat_dist = feat_ref * similarity_factor + torch.randn(hidden_dim) * 0.01
                
                layer_features_ref.append(feat_ref)
                layer_features_dist.append(feat_dist)
            
            features_ref.append(torch.stack(layer_features_ref))
            features_dist.append(torch.stack(layer_features_dist))
        
        # MOS scores increase with index (perfect rank correlation)
        mos_scores = list(range(num_samples))
        
        # Create batch
        batch = (
            torch.stack([torch.zeros(3, 224, 224) for _ in range(num_samples)]),
            torch.stack([torch.zeros(3, 224, 224) for _ in range(num_samples)]),
            mos_scores,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_dist": features_dist,
        }
        
        # Calculate
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        # Parse results
        correlations = json.loads(results["tid_spearman_mos"])
        
        # Should have positive correlation (close to 1.0)
        for layer_idx, corr in enumerate(correlations):
            assert corr > 0.7, f"Layer {layer_idx}: Expected high positive correlation, got {corr}"

    def test_perfect_negative_correlation(self):
        """Test Spearman with perfect negative correlation."""
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_dist = []
        
        for layer_idx in range(num_layers):
            layer_features_ref = []
            layer_features_dist = []
            
            for i in range(num_samples):
                feat_ref = torch.ones(hidden_dim)
                
                # Higher index = lower similarity (inverse of MOS)
                similarity_factor = (num_samples - i) / num_samples
                feat_dist = feat_ref * similarity_factor + torch.randn(hidden_dim) * 0.01
                
                layer_features_ref.append(feat_ref)
                layer_features_dist.append(feat_dist)
            
            features_ref.append(torch.stack(layer_features_ref))
            features_dist.append(torch.stack(layer_features_dist))
        
        # MOS increases, but similarity decreases (negative correlation)
        mos_scores = list(range(num_samples))
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            mos_scores,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_dist": features_dist,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        correlations = json.loads(results["tid_spearman_mos"])
        
        # Should have negative correlation
        for layer_idx, corr in enumerate(correlations):
            assert corr < -0.7, f"Layer {layer_idx}: Expected high negative correlation, got {corr}"

    def test_no_correlation(self):
        """Test Spearman with no correlation (random)."""
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        num_samples = 20
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_dist = []
        
        for layer_idx in range(num_layers):
            # Completely random features (no relationship with MOS)
            layer_features_ref = torch.randn(num_samples, hidden_dim)
            layer_features_dist = torch.randn(num_samples, hidden_dim)
            
            features_ref.append(layer_features_ref)
            features_dist.append(layer_features_dist)
        
        # Random MOS scores
        mos_scores = torch.randperm(num_samples).tolist()
        
        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            mos_scores,
        )
        
        batch_results = {
            "features_ref": features_ref,
            "features_dist": features_dist,
        }
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        correlations = json.loads(results["tid_spearman_mos"])
        
        # Should have low correlation (close to 0)
        for layer_idx, corr in enumerate(correlations):
            assert -0.5 < corr < 0.5, f"Layer {layer_idx}: Expected low correlation, got {corr}"

    def test_accumulation_across_batches(self):
        """Test that metric correctly accumulates across multiple batches."""
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        num_layers = 3
        hidden_dim = 768
        
        # First batch
        batch_size_1 = 5
        features_ref_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        features_dist_1 = [torch.randn(batch_size_1, hidden_dim) for _ in range(num_layers)]
        mos_1 = list(range(batch_size_1))
        
        batch_1 = (torch.zeros(batch_size_1, 3, 224, 224), torch.zeros(batch_size_1, 3, 224, 224), mos_1)
        batch_results_1 = {"features_ref": features_ref_1, "features_dist": features_dist_1}
        
        metric.calculate(batch_1, batch_results_1)
        
        # Second batch
        batch_size_2 = 7
        features_ref_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        features_dist_2 = [torch.randn(batch_size_2, hidden_dim) for _ in range(num_layers)]
        mos_2 = list(range(batch_size_2))
        
        batch_2 = (torch.zeros(batch_size_2, 3, 224, 224), torch.zeros(batch_size_2, 3, 224, 224), mos_2)
        batch_results_2 = {"features_ref": features_ref_2, "features_dist": features_dist_2}
        
        metric.calculate(batch_2, batch_results_2)
        
        # Check accumulation
        assert len(metric.similarities) == num_layers, f"Should have {num_layers} layers"
        assert len(metric.similarities[0]) == batch_size_1 + batch_size_2, "Should accumulate all samples"
        assert len(metric.mos_scores) == batch_size_1 + batch_size_2, "Should accumulate all MOS scores"
        
        # Finalize
        results = metric.finalize()
        correlations = json.loads(results["tid_spearman_mos"])
        
        assert len(correlations) == num_layers, f"Should have {num_layers} correlation values"

    def test_reset(self):
        """Test that reset clears accumulated state."""
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        # Add some data
        features_ref = [torch.randn(5, 768) for _ in range(2)]
        features_dist = [torch.randn(5, 768) for _ in range(2)]
        mos = [1, 2, 3, 4, 5]
        
        batch = (torch.zeros(5, 3, 224, 224), torch.zeros(5, 3, 224, 224), mos)
        batch_results = {"features_ref": features_ref, "features_dist": features_dist}
        
        metric.calculate(batch, batch_results)
        
        assert len(metric.similarities) > 0, "Should have accumulated data"
        assert len(metric.mos_scores) > 0, "Should have accumulated MOS scores"
        
        # Reset
        metric.reset()
        
        assert len(metric.similarities) == 0, "Similarities should be cleared"
        assert len(metric.mos_scores) == 0, "MOS scores should be cleared"

    def test_identical_features(self):
        """Test with identical reference and distorted features (perfect similarity)."""
        metric = SpearmanCorrelationMOS(BackendEnum.TORCH)
        
        num_samples = 10
        num_layers = 2
        hidden_dim = 768
        
        features_ref = []
        features_dist = []
        
        for layer_idx in range(num_layers):
            # Identical features (similarity = 1.0 for all)
            layer_features = torch.randn(num_samples, hidden_dim)
            features_ref.append(layer_features.clone())
            features_dist.append(layer_features.clone())
        
        # Varying MOS scores
        mos_scores = list(range(num_samples))
        
        batch = (torch.zeros(num_samples, 3, 224, 224), torch.zeros(num_samples, 3, 224, 224), mos_scores)
        batch_results = {"features_ref": features_ref, "features_dist": features_dist}
        
        metric.calculate(batch, batch_results)
        results = metric.finalize()
        
        correlations = json.loads(results["tid_spearman_mos"])
        
        # All similarities are 1.0 (no variance), so correlation should be NaN or undefined
        # Spearman correlation with constant values is NaN
        for corr in correlations:
            assert torch.isnan(torch.tensor(corr)), "Correlation should be NaN for constant similarities"
