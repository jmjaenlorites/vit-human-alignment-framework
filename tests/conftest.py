"""Shared fixtures for tests."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.models.vit_b16 import ViT_B_16
from src.utils.common_enums import BackendEnum


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tid_test_data_path(test_data_dir: Path) -> str:
    """Return path to TID test data."""
    return str(test_data_dir / "tid")


@pytest.fixture(scope="session")
def levels_test_data_path(test_data_dir: Path) -> str:
    """Return path to Levels test CSVs."""
    return str(test_data_dir / "levels")


@pytest.fixture(scope="session")
def levels_images_path(test_data_dir: Path) -> str:
    """Return path to Levels test images (flat structure for tests)."""
    return str(test_data_dir / "levels" / "images")


@pytest.fixture(scope="session")
def nights_test_data_path(test_data_dir: Path) -> str:
    """Return path to Nights test data."""
    return str(test_data_dir / "nights")


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model that returns deterministic features and saliency maps.
    
    Returns:
        Mock model with:
        - backend = BackendEnum.TORCH
        - transform = identity function
        - forward() returns dict with features and saliency
        - Features: [num_layers, batch_size, num_tokens, hidden_dim]
        - Saliency: [num_layers, batch_size, grid_size, grid_size]
    """
    mock = MagicMock()
    mock.backend = BackendEnum.TORCH
    mock.transform = lambda x: x
    mock.device = torch.device("cpu")
    
    def mock_forward(
        batch: torch.Tensor,
        return_features: bool = False,
        return_saliency: bool = False,
    ) -> dict[str, Any]:
        """Mock forward pass that returns deterministic outputs."""
        batch_size = batch.shape[0]
        num_layers = 12  # Standard for ViT-B/16
        num_tokens = 197  # CLS + 14x14 patches for 224x224 image
        hidden_dim = 768  # ViT-B/16 hidden dimension
        grid_size = 14  # sqrt(196) patches
        
        result = {}
        
        if return_features:
            # Create deterministic features per layer
            features = []
            for layer_idx in range(num_layers):
                # Use layer index to create different but deterministic features
                layer_features = torch.ones(batch_size, num_tokens, hidden_dim) * (layer_idx + 1) / num_layers
                features.append(layer_features)
            result["features"] = features
        else:
            result["features"] = None
            
        if return_saliency:
            # Create deterministic saliency maps per layer
            saliency = []
            for layer_idx in range(num_layers):
                # Create a gradient pattern that varies by layer
                layer_saliency = torch.zeros(batch_size, grid_size, grid_size)
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Value depends on position and layer
                        layer_saliency[:, i, j] = (i + j + layer_idx) / (2 * grid_size + num_layers)
                saliency.append(layer_saliency)
            result["saliency"] = saliency
        else:
            result["saliency"] = None
            
        return result
    
    mock.forward = mock_forward
    
    return mock


@pytest.fixture(scope="session")
def real_model() -> ViT_B_16:
    """Load real ViT_B_16 model (cached for session).
    
    This fixture loads the actual model once per test session to avoid
    repeated downloads and initialization overhead.
    
    Returns:
        ViT_B_16 model instance
    """
    model = ViT_B_16()
    return model


@pytest.fixture
def saliency_golden_data(fixtures_dir: Path) -> dict[str, Any]:
    """Load golden data for saliency metrics.
    
    Returns:
        Dict with:
        - stimulus: Tensor of input images
        - saliency_gt: Ground truth saliency maps
        - fixation_gt: Ground truth fixation maps
        - predicted_saliency: List of predicted saliency per layer
        - expected_auc_judd: Expected AUC_Judd values per layer
        - expected_pearson: Expected Pearson correlation per layer
    """
    golden_path = fixtures_dir / "saliency_golden.pt"
    
    if not golden_path.exists():
        pytest.skip(f"Golden data not found at {golden_path}. Run generate_golden_fixtures.py first.")
    
    return torch.load(golden_path)


@pytest.fixture
def tid_golden_data(fixtures_dir: Path) -> dict[str, Any]:
    """Load golden data for TID metrics.
    
    Returns:
        Dict with:
        - reference_images: Reference image tensors
        - distorted_images: Distorted image tensors
        - mos_scores: MOS scores
        - features_ref: Reference features per layer
        - features_dist: Distorted features per layer
        - expected_spearman: Expected Spearman correlation per layer
    """
    golden_path = fixtures_dir / "tid_golden.pt"
    
    if not golden_path.exists():
        pytest.skip(f"Golden data not found at {golden_path}. Run generate_golden_fixtures.py first.")
    
    return torch.load(golden_path)


@pytest.fixture
def levels_golden_data(fixtures_dir: Path) -> dict[str, Any]:
    """Load golden data for Levels metrics.
    
    Returns:
        Dict with:
        - img1: First image tensors
        - img2: Second image tensors
        - img3: Third image tensors
        - selected: Selected image names
        - features_img1: Features for img1 per layer
        - features_img2: Features for img2 per layer
        - features_img3: Features for img3 per layer
        - expected_accuracy: Expected accuracy per layer
    """
    golden_path = fixtures_dir / "levels_golden.pt"
    
    if not golden_path.exists():
        pytest.skip(f"Golden data not found at {golden_path}. Run generate_golden_fixtures.py first.")
    
    return torch.load(golden_path)


@pytest.fixture
def nights_golden_data(fixtures_dir: Path) -> dict[str, Any]:
    """Load golden data for Nights metrics.
    
    Returns:
        Dict with:
        - reference: Reference image tensors
        - left: Left distortion tensors
        - right: Right distortion tensors
        - left_votes: Left preference votes
        - right_votes: Right preference votes
        - features_ref: Reference features per layer
        - features_left: Left features per layer
        - features_right: Right features per layer
        - expected_accuracy: Expected accuracy per layer
    """
    golden_path = fixtures_dir / "nights_golden.pt"
    
    if not golden_path.exists():
        pytest.skip(f"Golden data not found at {golden_path}. Run generate_golden_fixtures.py first.")
    
    return torch.load(golden_path)
