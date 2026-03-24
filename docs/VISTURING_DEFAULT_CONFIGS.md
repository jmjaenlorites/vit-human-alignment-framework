# Visturing Metrics - Default Configurations

This document lists the default configurations used when visturing metrics are run from CSV (without explicit parameters).

## Default Configurations Table

| Metric Name | Default Dataset | Default Channel | Default Frequency | Default Mask | Notes |
|-------------|-----------------|-----------------|-------------------|--------------|-------|
| `visturing_spectral_sensitivity` | Experiment_1 | - | - | - | Prop1: Spectral sensitivities |
| `visturing_weber_law_pearson` | Experiment_2 | `all` | - | - | Prop2: achrom + chromatic combined |
| `visturing_weber_law_kendall` | Experiment_2 | `all` | - | - | Prop2: Order correlation |
| `visturing_csf_pearson` | Experiment_3_4 | `all` | - | - | Prop3_4: Contrast sensitivity |
| `visturing_csf_kendall` | Experiment_3_4 | `all` | - | - | Prop3_4: Order correlation |
| `visturing_campbell_blakemore_pearson` | Experiment_5 | - | - | `all` | Prop5: Frequency masking |
| `visturing_campbell_blakemore_kendall` | Experiment_5 | - | - | `all` | Prop5: Order correlation |
| `visturing_contrast_curves_pearson` | Experiment_6_7 | `all` | `all` | - | Prop6_7: Contrast curves |
| `visturing_contrast_curves_kendall` | Experiment_6_7 | `all` | `all` | - | Prop6_7: Order correlation |
| `visturing_contrast_masking_kendall` | Experiment_8 | - | `all` | `all` | Prop8: Contrast masking |
| `visturing_frequency_masking_kendall` | Experiment_9 | - | `all` | `all` | Prop9: Frequency masking |
| `visturing_orientation_masking_kendall` | Experiment_10 | - | `all` | `all` | Prop10: Orientation masking |

## Channel Options

When using factory functions, you can specify:

- **All channels**: `"all"` - Runs all channels and combines as in visturing
- **Achromatic**: `"achrom"` or `"a"` - Luminance channel
- **Red-Green**: `"red_green"` or `"rg"` - Red-green opponent channel
- **Yellow-Blue**: `"yellow_blue"` or `"yb"` - Yellow-blue opponent channel

## Frequency Options

- **All frequencies**: `"all"` - Runs full sweep
- **Low**: `"low"` = 3 cpd (cycles per degree)
- **High**: `"high"` = 12 cpd
- **Specific**: `"1p5"`, `"3"`, `"6"`, `"12"`, `"24"` (1.5, 3, 6, 12, 24 cpd)

## Mask Options

### Contrast Masking (Prop8)
- `"all"` = all mask contrasts
- `"0075"` = 0.075
- `"0150"` = 0.150
- `"0225"` = 0.225
- `"0300"` = 0.300

### Frequency Masking (Prop9)
- `"all"` = all mask frequencies
- `"1p5"`, `"3"`, `"6"`, `"12"`, `"24"` cpd

### Orientation Masking (Prop10)
- `"all"` = all orientations
- `"0"` = 0°
- `"22p5"` = 22.5°
- `"45"` = 45°
- `"67p5"` = 67.5°
- `"90"` = 90°
- `"112p5"` = 112.5°
- `"135"` = 135°

## Custom Configurations Example

To use custom configurations, use factory functions:

```python
from src.metrics.visturing import create_prop2_calculator

# Evaluate red-green channel (instead of default achromatic)
calculator = create_prop2_calculator(
    backend=BackendEnum.TORCH,
    channel="red_green",  # Custom
    data_path="./data/visturing",
    batch_size=32,
)

results = calculator.run(model)
```

## Why Default Configurations?

The CSV format has limitations:
- Only supports metric names as column headers
- No way to specify metric-specific parameters
- Same issue affects Levels (can't specify split in CSV)

**Solution: JSON Migration (Planned)**

Future versions will use JSON configuration files that support metric-specific parameters while maintaining backward compatibility with CSV for simple cases.
