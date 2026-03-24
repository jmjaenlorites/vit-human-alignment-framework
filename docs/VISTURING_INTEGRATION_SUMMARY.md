# Visturing Integration Summary

## Overview

Successfully integrated all 10 psychophysical properties from the visturing project into the vit-human-alignment-framework. The integration allows evaluation of Vision Transformers using metrics that assess alignment with human visual perception.

## What Was Implemented

### 1. Base Infrastructure

**Files Created:**
- `src/utils/download.py` - Zenodo dataset download utilities
- `src/dataset_loaders/visturing/base.py` - Base dataset loader classes
- `src/metrics/visturing/base.py` - Base metric and calculator classes
- `src/metrics/visturing/ground_truth.py` - Ground truth file loading
- `src/metrics/visturing/distance_functions.py` - Feature distance and correlation utilities

### 2. Dataset Loaders (10 properties)

**Files Created:**
- `src/dataset_loaders/visturing/prop1.py` - Spectral Sensitivities (Experiment_1)
- `src/dataset_loaders/visturing/prop2.py` - Weber Law (Experiment_2)
- `src/dataset_loaders/visturing/prop3_4.py` - CSF (Experiment_3_4)
- `src/dataset_loaders/visturing/prop5.py` - Campbell-Blakemore (Experiment_5)
- `src/dataset_loaders/visturing/prop6_7.py` - Contrast Curves (Experiment_6_7)
- `src/dataset_loaders/visturing/prop8.py` - Contrast Masking (Experiment_8)
- `src/dataset_loaders/visturing/prop9.py` - Frequency Masking (Experiment_9)
- `src/dataset_loaders/visturing/prop10.py` - Orientation Masking (Experiment_10)

Each loader:
- Automatically downloads datasets from Zenodo
- Provides PyTorch DataLoader compatibility
- Handles multiple channels (achromatic, red-green, yellow-blue)
- Manages various experimental parameters (frequencies, contrasts, orientations)

### 3. Metrics (10+ metric classes)

**Files Created:**
- `src/metrics/visturing/prop1.py` - SpectralSensitivityPearson
- `src/metrics/visturing/prop2.py` - WeberLawPearson, WeberLawKendall
- `src/metrics/visturing/prop3_4.py` - CSFPearson, CSFKendall
- `src/metrics/visturing/prop5.py` - CampbellBlakemorePearson, CampbellBlakemoreKendall
- `src/metrics/visturing/prop6_7.py` - ContrastCurvesPearson, ContrastCurvesKendall
- `src/metrics/visturing/prop8.py` - ContrastMaskingKendall
- `src/metrics/visturing/prop9.py` - FrequencyMaskingKendall
- `src/metrics/visturing/prop10.py` - OrientationMaskingKendall

Each metric:
- Calculates correlations with human psychophysical ground truth
- Supports per-layer evaluation (all ViT layers)
- Returns JSON-serialized results
- Includes factory functions for easy instantiation

### 4. Registry and Runner Updates

**Files Modified:**
- `src/metrics/__init__.py` - Added all visturing metrics to registry
- `src/metrics/visturing/__init__.py` - Exported all metrics and factories
- `src/runner/base.py` - Added detection and warning for visturing metrics

### 5. Tests

**Files Created:**
- `tests/test_visturing.py` - Unit tests for distance functions, metrics, calculators, and loaders

Tests cover:
- Distance calculation (Euclidean, Cosine)
- Correlation functions (Pearson, Kendall, Spearman)
- Metric initialization
- Calculator creation
- Dataset loading (when datasets available)

### 6. Documentation

**Files Created:**
- `docs/VISTURING_METRICS.md` - Complete usage guide
- `examples/visturing_example.py` - Practical usage examples
- `VISTURING_INTEGRATION_SUMMARY.md` - This summary

## Dependencies Installed

Added via `uv add`:
- `wget` - Dataset downloading
- `natsort` - Natural file sorting
- `scipy` - Already present, used for .mat files and correlations

## Key Features

### 1. Automatic Dataset Management
- Datasets automatically download from Zenodo on first use
- Ground truth files cached locally
- No manual setup required

### 2. Multi-Backend Support
- PyTorch (fully implemented)
- JAX (structure in place, ready for implementation)

### 3. Per-Layer Evaluation
- All metrics compute correlations for each model layer
- Enables analysis of alignment throughout the network
- Results serialized as JSON arrays

### 4. Factory Pattern
- Each property has a `create_propN_calculator()` function
- Handles all configuration automatically
- Clean, simple API

### 5. Framework Integration
- Follows existing metric protocol
- Compatible with BaseMetric interface
- Registered in metric loader system

## Usage Example

```python
from src.utils.common_enums import BackendEnum
from src.models import load_model
from src.metrics.visturing import create_prop1_calculator

# Load model
model = load_model("vit_b16_imagenet1k")

# Create calculator
calculator = create_prop1_calculator(
    backend=BackendEnum.TORCH,
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
)

# Run evaluation
results = calculator.run(model)

# Parse results
import json
correlations = json.loads(results["visturing_spectral_sensitivity"])
print(f"Layer 0: {correlations[0]:.3f}")
```

## Metric Naming Convention

All visturing metrics follow the pattern:
```
visturing_{property}_{correlation_type}_{optional_channel}
```

Examples:
- `visturing_spectral_sensitivity`
- `visturing_weber_law_pearson_achrom`
- `visturing_csf_kendall_rg`

## Integration Status

### ✅ Completed
- [x] All 10 properties implemented
- [x] Dataset loaders for all experiments
- [x] Metric classes with correlation calculations
- [x] Factory functions for easy usage
- [x] Registry integration
- [x] Runner awareness
- [x] Unit tests
- [x] Documentation
- [x] Examples

### ⚠️ Notes
- **CSV Support**: Visturing metrics can be used in CSV with default configurations
  - Similar to Levels (uses default split "between_class")
  - Prop1 (spectral sensitivity) used as default dataset
  - For custom configurations (channels, frequencies, masks), use factory functions
- **Future JSON Config**: Migration to JSON format planned to support metric-specific parameters
- JAX backend structure in place but needs implementation
- Ground truth files auto-download from Zenodo on first use

## File Structure

```
vit-human-alignment-framework/
├── src/
│   ├── dataset_loaders/
│   │   └── visturing/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── prop1.py through prop10.py
│   ├── metrics/
│   │   ├── __init__.py (updated)
│   │   └── visturing/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── distance_functions.py
│   │       ├── ground_truth.py
│   │       ├── prop1.py through prop10.py
│   ├── runner/
│   │   └── base.py (updated)
│   └── utils/
│       └── download.py
├── tests/
│   └── test_visturing.py
├── docs/
│   └── VISTURING_METRICS.md
├── examples/
│   └── visturing_example.py
└── VISTURING_INTEGRATION_SUMMARY.md
```

## Credits

Original visturing project: https://github.com/pablombg/visturing
Integration by: Current implementation team
Dataset: Zenodo record 17700252

## Next Steps

To use the integrated metrics:

1. **Basic Usage**: See `examples/visturing_example.py`
2. **Full Documentation**: See `docs/VISTURING_METRICS.md`
3. **API Reference**: Each metric has docstrings with parameter details
4. **Testing**: Run `pytest tests/test_visturing.py`

To evaluate your model:
```bash
cd vit-human-alignment-framework
python examples/visturing_example.py
```

The datasets will download automatically on first run.
