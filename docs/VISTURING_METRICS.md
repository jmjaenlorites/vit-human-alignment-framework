# Visturing Metrics - Usage Guide

This document describes how to use the visturing psychophysical metrics integrated into the framework.

## Overview

The visturing metrics evaluate 10 psychophysical and physiological properties of the human visual system (Retina-V1). Each property has been integrated as an independent metric in the framework.

## Available Metrics

| Property | Metric Name | Description |
|----------|-------------|-------------|
| prop1 | `visturing_spectral_sensitivity` | Spectral sensitivity (achromatic) |
| prop2 | `visturing_weber_law_pearson` | Weber law (luminance/chromatic responses) |
| prop2 | `visturing_weber_law_kendall` | Weber law (ordering) |
| prop3_4 | `visturing_csf_pearson` | Contrast Sensitivity Function |
| prop3_4 | `visturing_csf_kendall` | CSF (ordering) |
| prop5 | `visturing_campbell_blakemore_pearson` | Campbell-Blakemore frequency masking |
| prop5 | `visturing_campbell_blakemore_kendall` | Campbell-Blakemore (ordering) |
| prop6_7 | `visturing_contrast_curves_pearson` | Contrast curves without mask |
| prop6_7 | `visturing_contrast_curves_kendall` | Contrast curves (ordering) |
| prop8 | `visturing_contrast_masking_kendall` | Contrast masking |
| prop9 | `visturing_frequency_masking_kendall` | Frequency masking |
| prop10 | `visturing_orientation_masking_kendall` | Orientation masking |

## Usage

### Option 1: CSV-Driven (Default Configurations)

The simplest way to use visturing metrics is through the CSV-driven runner with default configurations:

**Example CSV** (`data/vit-visturing-test.csv`):
```csv
model_name,metric_visturing_spectral_sensitivity,metric_visturing_weber_law_pearson
vit-b16,,
```

**Running:**
```python
from src.runner.base import Runner
from src.utils.common_enums import BackendEnum

runner = Runner(
    csv_path="data/vit-visturing-test.csv",
    backend=BackendEnum.TORCH,
)
runner.execute()
```

**Default Configurations Used:**
- Each metric runs its corresponding experiment dataset (Prop1..Prop10)
- Defaults use `all` where applicable (channels, frequencies, masks), matching visturing's full sweep
- Similar to how Levels uses default split `"between_class"`

**Note:** This approach is suitable for quick evaluations. For specific channels, frequencies, or mask parameters, use factory functions (see Option 2).

**Future Migration:** The framework will migrate from CSV to JSON configuration, allowing:
```json
{
  "model_name": "vit_b16",
  "metrics": [
    {"name": "visturing_weber_law_pearson", "config": {"channel": "red_green"}},
    {"name": "visturing_csf_pearson", "config": {"channel": "achrom", "freq": "6"}},
    {"name": "levels_triplet_accuracy", "config": {"split": "class_border"}}
  ]
}
```

### Option 2: Factory Functions (Custom Configurations)

Each property has a factory function to create a configured calculator:

```python
from src.utils.common_enums import BackendEnum
from src.metrics.visturing import create_prop1_calculator
from src.models import load_model

# Load model
model = load_model("vit_b16_imagenet1k")

# Create calculator for Prop1 (spectral sensitivity)
calculator = create_prop1_calculator(
    backend=BackendEnum.TORCH,
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
)

# Run evaluation
results = calculator.run(model)
print(results)
```

### Example: Prop2 (Weber Law)

```python
from src.metrics.visturing import create_prop2_calculator

# Evaluate achromatic channel with both Pearson and Kendall
calculator = create_prop2_calculator(
    backend=BackendEnum.TORCH,
    channel="achrom",  # Options: 'achrom', 'red_green', 'yellow_blue'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
    include_kendall=True,
)

results = calculator.run(model)
# Results contain correlations per layer as JSON arrays
```

### Example: Prop3_4 (CSF)

```python
from src.metrics.visturing import create_prop3_4_calculator

# Evaluate achromatic channel at low frequency
calculator = create_prop3_4_calculator(
    backend=BackendEnum.TORCH,
    channel="achrom",  # Options: 'achrom', 'rg', 'yb'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
    include_kendall=True,
)

results = calculator.run(model)
```

### Example: Prop5 (Campbell-Blakemore)

```python
from src.metrics.visturing import create_prop5_calculator

# Evaluate frequency masking at 3 cpd
calculator = create_prop5_calculator(
    backend=BackendEnum.TORCH,
    mask_freq="3",  # Options: '3', '6', '12'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
    include_kendall=True,
)

results = calculator.run(model)
```

### Example: Prop6_7 (Contrast Curves)

```python
from src.metrics.visturing import create_prop6_7_calculator

# Evaluate contrast curves for achromatic channel at 1.5 cpd
calculator = create_prop6_7_calculator(
    backend=BackendEnum.TORCH,
    channel="a",  # Options: 'a', 'rg', 'yb'
    freq="1p5",  # Options: '1p5', '3', '6', '12', '24'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
    include_kendall=True,
)

results = calculator.run(model)
```

### Example: Prop8 (Contrast Masking)

```python
from src.metrics.visturing import create_prop8_calculator

# Evaluate contrast masking at low frequency (3 cpd) with mask contrast 0.075
calculator = create_prop8_calculator(
    backend=BackendEnum.TORCH,
    freq="low",  # Options: 'low' (3cpd), 'high' (12cpd)
    mask_contrast="0075",  # Options: '0075', '0150', '0225', '0300'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
)

results = calculator.run(model)
```

### Example: Prop9 (Frequency Masking)

```python
from src.metrics.visturing import create_prop9_calculator

# Evaluate frequency masking at low frequency with 1.5 cpd mask
calculator = create_prop9_calculator(
    backend=BackendEnum.TORCH,
    freq="low",  # Options: 'low', 'high'
    mask_freq="1p5",  # Options: '1p5', '3', '6', '12', '24'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
)

results = calculator.run(model)
```

### Example: Prop10 (Orientation Masking)

```python
from src.metrics.visturing import create_prop10_calculator

# Evaluate orientation masking at low frequency with 0 degree mask
calculator = create_prop10_calculator(
    backend=BackendEnum.TORCH,
    freq="low",  # Options: 'low', 'high'
    mask_orientation="0",  # Options: '0', '22p5', '45', '67p5', '90', '112p5', '135'
    data_path="./data/visturing",
    gt_path="./data/visturing",
    batch_size=32,
)

results = calculator.run(model)
```

## Data Download

All datasets and ground truth files are automatically downloaded from Zenodo when first needed. The downloads are cached in `data_path`.

## Results Format

Results are returned as dictionaries with metric names as keys and JSON-serialized arrays as values. Each array contains one correlation value per model layer.

Example:
```python
{
    "visturing_spectral_sensitivity": "[0.85, 0.87, 0.89, ...]",
    "visturing_spectral_sensitivity_pvalues": "[0.001, 0.0005, 0.0003, ...]"
}
```

To parse:
```python
import json

correlations = json.loads(results["visturing_spectral_sensitivity"])
print(f"Correlation at layer 0: {correlations[0]}")
```

## Integration with Automated Runner

### CSV Support (Default Configurations)

Visturing metrics **are supported** in the CSV-driven automated runner with default configurations:

```csv
model_name,metric_visturing_spectral_sensitivity,metric_visturing_csf_pearson
vit-b16,,
```

This uses default parameters (per-property dataset and full sweep defaults) similar to how Levels uses default split "between_class".

### Custom Configurations

For custom configurations (specific channels, frequencies, masks), use factory functions directly:

```python
from src.metrics.visturing import create_prop2_calculator

# Custom: Red-Green channel instead of default achromatic
calculator = create_prop2_calculator(
    backend=backend,
    channel="red_green",  # Non-default
    include_kendall=True,
)
results = calculator.run(model)
```

### Future: JSON Configuration

The framework will migrate to JSON format to support metric-specific configurations in batch runs:

```json
{
  "model_name": "vit_b16",
  "metrics": [
    {"name": "visturing_weber_law_pearson", "config": {"channel": "red_green"}},
    {"name": "levels_triplet_accuracy", "config": {"split": "class_border"}}
  ]
}
```

## Architecture

Each visturing property consists of:

1. **Dataset Loader** (`src/dataset_loaders/visturing/propN.py`): Loads experimental images and metadata
2. **Metric** (`src/metrics/visturing/propN.py`): Calculates correlations with ground truth
3. **Factory Function** (`create_propN_calculator`): Configures and returns a `VisTuringCalculator`

All metrics inherit from `BaseVisTuringMetric` and follow the framework's metric protocol.

## References

- Visturing original repository: https://github.com/pablombg/visturing
- Zenodo dataset: https://zenodo.org/records/17700252
