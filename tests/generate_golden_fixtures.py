"""Generate golden fixtures for regression testing.

This script loads real model and datasets, processes small batches,
and saves the results as golden data for future regression tests.

Usage:
    python tests/generate_golden_fixtures.py
"""

import json
import logging
from pathlib import Path

import torch

from src.dataset_loaders.levels import LevelsTorchDatasetLoader
from src.dataset_loaders.nights import NightsTorchDatasetLoader
from src.dataset_loaders.saliency import SaliencyMIT1003TorchDatasetLoader
from src.dataset_loaders.tid import TID2013TorchDatasetLoader
from src.metrics.levels import LevelsMetricsCalculator
from src.metrics.nights import NightsMetricsCalculator
from src.metrics.saliency import SaliencyMetricsCalculator
from src.metrics.tid import TIDMetricsCalculator
from src.models.vit_b16 import ViT_B_16
from src.utils.common_enums import BackendEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_saliency_golden(
    model: ViT_B_16, test_data_path: str, fixtures_dir: Path
) -> None:
    """Generate golden data for saliency metrics."""
    logger.info("Generating saliency golden fixtures...")
    
    # Load a small batch from MIT1003
    loader = SaliencyMIT1003TorchDatasetLoader(
        batch_size=4, shuffle=False, num_workers=0, transform=model.transform
    )
    
    # Get first batch
    iterator = loader.get_iterator()
    batch = next(iter(iterator))
    stimulus, saliency_gt, fixation_gt = batch
    
    # Run model forward
    stimulus_device = stimulus.to(model.device)
    model_output = model.forward(stimulus_device, return_features=True, return_saliency=True)
    
    # Move to CPU for saving
    saliency_maps = [s.cpu() for s in model_output["saliency"]]
    features = [f.cpu() for f in model_output["features"]]
    
    # Calculate metrics
    calculator = SaliencyMetricsCalculator(BackendEnum.TORCH)
    results = calculator.run(model)
    
    # Parse expected values
    expected_auc_judd = json.loads(results["saliency_auc_judd"])
    expected_pearson = json.loads(results["saliency_pearson_correlation_coefficient"])
    
    # Save golden data
    golden_data = {
        "stimulus": stimulus.cpu(),
        "saliency_gt": saliency_gt.cpu(),
        "fixation_gt": fixation_gt.cpu(),
        "predicted_saliency": saliency_maps,
        "features": features,
        "expected_auc_judd": expected_auc_judd,
        "expected_pearson": expected_pearson,
    }
    
    output_path = fixtures_dir / "saliency_golden.pt"
    torch.save(golden_data, output_path)
    logger.info(f"Saved saliency golden data to {output_path}")
    logger.info(f"  AUC_Judd: {expected_auc_judd[:3]}...")
    logger.info(f"  Pearson: {expected_pearson[:3]}...")


def generate_tid_golden(
    model: ViT_B_16, test_data_path: str, fixtures_dir: Path
) -> None:
    """Generate golden data for TID metrics."""
    logger.info("Generating TID golden fixtures...")
    
    # Load from test data
    loader = TID2013TorchDatasetLoader(
        batch_size=8,
        shuffle=False,
        num_workers=0,
        transform=model.transform,
        dataset_path=test_data_path,
    )
    
    # Get first batch
    iterator = loader.get_iterator()
    batch = next(iter(iterator))
    reference_images, distorted_images, mos_scores = batch
    
    # Run model forward
    ref_device = reference_images.to(model.device)
    dist_device = distorted_images.to(model.device)
    
    model_output_ref = model.forward(ref_device, return_features=True, return_saliency=False)
    model_output_dist = model.forward(dist_device, return_features=True, return_saliency=False)
    
    # Move to CPU
    features_ref = [f.cpu() for f in model_output_ref["features"]]
    features_dist = [f.cpu() for f in model_output_dist["features"]]
    
    # Calculate metrics
    calculator = TIDMetricsCalculator(BackendEnum.TORCH, dataset_path=test_data_path)
    results = calculator.run(model)
    
    expected_spearman = json.loads(results["tid_spearman_mos"])
    
    # Save golden data
    golden_data = {
        "reference_images": reference_images.cpu(),
        "distorted_images": distorted_images.cpu(),
        "mos_scores": mos_scores,
        "features_ref": features_ref,
        "features_dist": features_dist,
        "expected_spearman": expected_spearman,
    }
    
    output_path = fixtures_dir / "tid_golden.pt"
    torch.save(golden_data, output_path)
    logger.info(f"Saved TID golden data to {output_path}")
    logger.info(f"  Spearman: {expected_spearman[:3]}...")


def generate_levels_golden(
    model: ViT_B_16, test_data_path: str, images_path: str, fixtures_dir: Path
) -> None:
    """Generate golden data for Levels metrics."""
    logger.info("Generating Levels golden fixtures...")
    
    # Load from test data
    loader = LevelsTorchDatasetLoader(
        batch_size=6,
        shuffle=False,
        num_workers=0,
        split="between_class",
        transform=model.transform,
        levels_path=test_data_path,
        imagenet_path=images_path,
    )
    
    # Get first batch
    iterator = loader.get_iterator()
    batch = next(iter(iterator))
    img1, img2, img3, selected, img1_names, img2_names, img3_names = batch
    
    # Run model forward
    img1_device = img1.to(model.device)
    img2_device = img2.to(model.device)
    img3_device = img3.to(model.device)
    
    model_output_1 = model.forward(img1_device, return_features=True, return_saliency=False)
    model_output_2 = model.forward(img2_device, return_features=True, return_saliency=False)
    model_output_3 = model.forward(img3_device, return_features=True, return_saliency=False)
    
    # Move to CPU
    features_img1 = [f.cpu() for f in model_output_1["features"]]
    features_img2 = [f.cpu() for f in model_output_2["features"]]
    features_img3 = [f.cpu() for f in model_output_3["features"]]
    
    # Calculate metrics
    calculator = LevelsMetricsCalculator(
        BackendEnum.TORCH,
        split="between_class",
        levels_path=test_data_path,
        imagenet_path=images_path,
    )
    results = calculator.run(model)
    
    expected_accuracy = json.loads(results["levels_triplet_accuracy"])
    
    # Save golden data
    golden_data = {
        "img1": img1.cpu(),
        "img2": img2.cpu(),
        "img3": img3.cpu(),
        "selected": list(selected),
        "img1_names": list(img1_names),
        "img2_names": list(img2_names),
        "img3_names": list(img3_names),
        "features_img1": features_img1,
        "features_img2": features_img2,
        "features_img3": features_img3,
        "expected_accuracy": expected_accuracy,
    }
    
    output_path = fixtures_dir / "levels_golden.pt"
    torch.save(golden_data, output_path)
    logger.info(f"Saved Levels golden data to {output_path}")
    logger.info(f"  Accuracy: {expected_accuracy[:3]}...")


def generate_nights_golden(
    model: ViT_B_16, test_data_path: str, fixtures_dir: Path
) -> None:
    """Generate golden data for Nights metrics."""
    logger.info("Generating Nights golden fixtures...")
    
    # Load from test data
    loader = NightsTorchDatasetLoader(
        batch_size=6,
        shuffle=False,
        num_workers=0,
        transform=model.transform,
        dataset_path=test_data_path,
    )
    
    # Get first batch
    iterator = loader.get_iterator()
    batch = next(iter(iterator))
    reference, left, right, left_votes, right_votes = batch
    
    # Run model forward
    ref_device = reference.to(model.device)
    left_device = left.to(model.device)
    right_device = right.to(model.device)
    
    model_output_ref = model.forward(ref_device, return_features=True, return_saliency=False)
    model_output_left = model.forward(left_device, return_features=True, return_saliency=False)
    model_output_right = model.forward(right_device, return_features=True, return_saliency=False)
    
    # Move to CPU
    features_ref = [f.cpu() for f in model_output_ref["features"]]
    features_left = [f.cpu() for f in model_output_left["features"]]
    features_right = [f.cpu() for f in model_output_right["features"]]
    
    # Calculate metrics
    calculator = NightsMetricsCalculator(
        BackendEnum.TORCH, dataset_path=test_data_path
    )
    results = calculator.run(model)
    
    expected_accuracy = json.loads(results["nights_preference_accuracy"])
    
    # Save golden data
    golden_data = {
        "reference": reference.cpu(),
        "left": left.cpu(),
        "right": right.cpu(),
        "left_votes": left_votes,
        "right_votes": right_votes,
        "features_ref": features_ref,
        "features_left": features_left,
        "features_right": features_right,
        "expected_accuracy": expected_accuracy,
    }
    
    output_path = fixtures_dir / "nights_golden.pt"
    torch.save(golden_data, output_path)
    logger.info(f"Saved Nights golden data to {output_path}")
    logger.info(f"  Accuracy: {expected_accuracy[:3]}...")


def main():
    """Generate all golden fixtures."""
    # Setup paths
    test_dir = Path(__file__).parent
    fixtures_dir = test_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    data_dir = test_dir / "data"
    tid_path = str(data_dir / "tid")
    levels_csv_path = str(data_dir / "levels")
    levels_images_path = str(data_dir / "levels" / "images")
    nights_path = str(data_dir / "nights")
    
    # Load model once
    logger.info("Loading ViT_B_16 model...")
    model = ViT_B_16()
    logger.info("Model loaded successfully")
    
    # Generate fixtures for each dataset
    try:
        generate_saliency_golden(model, None, fixtures_dir)
    except Exception as e:
        logger.error(f"Failed to generate saliency golden: {e}")
    
    try:
        generate_tid_golden(model, tid_path, fixtures_dir)
    except Exception as e:
        logger.error(f"Failed to generate TID golden: {e}")
    
    try:
        generate_levels_golden(model, levels_csv_path, levels_images_path, fixtures_dir)
    except Exception as e:
        logger.error(f"Failed to generate Levels golden: {e}")
    
    try:
        generate_nights_golden(model, nights_path, fixtures_dir)
    except Exception as e:
        logger.error(f"Failed to generate Nights golden: {e}")
    
    logger.info("Golden fixtures generation complete!")


if __name__ == "__main__":
    main()
