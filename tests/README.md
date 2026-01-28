# Tests for ViT Human Alignment Framework

This directory contains regression tests for the metrics calculation framework.

## Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── data/                          # Minimal test datasets
│   ├── tid/                      # TID2013 test data
│   ├── levels/                   # Levels test data
│   └── nights/                   # Nights test data
├── fixtures/                      # Golden data (generated)
│   ├── saliency_golden.pt
│   ├── tid_golden.pt
│   ├── levels_golden.pt
│   └── nights_golden.pt
├── unit/                          # Unit tests
│   ├── test_saliency_metrics.py
│   ├── test_tid_metrics.py
│   ├── test_levels_metrics.py
│   └── test_nights_metrics.py
└── integration/                   # Integration tests
    ├── test_saliency_calculator.py
    └── test_perceptual_calculators.py
```

## Setup

Install test dependencies:

```bash
cd vit-human-alignment-framework
uv pip install -e ".[dev]"
```

## Generating Golden Fixtures

Before running golden data tests, generate the fixtures:

```bash
python tests/generate_golden_fixtures.py
```

This will:
- Load the real ViT_B_16 model
- Process small batches from each test dataset
- Save model outputs and expected metrics to `tests/fixtures/`

## Running Tests

### All tests

```bash
uv run pytest tests/
```

### Unit tests only (fast)

```bash
uv run pytest tests/unit/
```

### Integration tests with real model (slower)

```bash
uv run pytest tests/integration/ -m real_model
```

### Golden data tests only

```bash
uv run pytest -m golden
```

### Skip slow tests

```bash
uv run pytest -m "not slow"
```

### With coverage

```bash
uv run pytest tests/ --cov=src --cov-report=html
```

## Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.real_model` - Tests using actual ViT model (slower)
- `@pytest.mark.golden` - Tests using pre-computed golden fixtures
- `@pytest.mark.slow` - Tests that take significant time

## What the Tests Cover

### Unit Tests

- **Saliency Metrics** (`test_saliency_metrics.py`)
  - `AUC_Judd._calculate_torch()` - Perfect/inverse/random predictions
  - `PearsonCorrelationCoefficient._calculate_torch()` - Correlation calculations
  - Accumulation and finalization logic

- **TID Metrics** (`test_tid_metrics.py`)
  - `SpearmanCorrelationMOS` - Correlation with MOS scores
  - Perfect positive/negative/no correlation cases
  - Multi-batch accumulation

- **Levels Metrics** (`test_levels_metrics.py`)
  - `TripletAccuracy` - Outlier detection logic
  - All three outlier positions (img1, img2, img3)
  - Random baseline (~33% accuracy)

- **Nights Metrics** (`test_nights_metrics.py`)
  - `PreferenceAccuracy` - Preference prediction
  - Left/right preference scenarios
  - Inverse predictions

### Integration Tests

- **Saliency Calculator** (`test_saliency_calculator.py`)
  - Full pipeline with real model
  - Result consistency across runs
  - Layer progression
  - Batch processing
  - Golden data matching

- **Perceptual Calculators** (`test_perceptual_calculators.py`)
  - TID, Levels, and Nights full pipelines
  - Real model integration
  - Test data loading
  - Result consistency
  - Golden data matching

## Expected Results

After optimization changes, tests should:

1. **Pass all unit tests** - Core calculation logic unchanged
2. **Match golden data** - Same results for same inputs
3. **Show consistent results** - Deterministic across runs
4. **Maintain valid ranges** - Metrics within expected bounds

If tests fail after optimization:
- Check if calculation logic was accidentally changed
- Verify numerical precision issues
- Re-generate golden fixtures if model/data changed intentionally

## Updating Golden Fixtures

If you intentionally change:
- Model architecture
- Metric calculations
- Dataset preprocessing

Re-generate golden fixtures:

```bash
python tests/generate_golden_fixtures.py
```

Then verify the new results are correct before committing.

## CI/CD

Recommended CI pipeline:

```yaml
# Run fast tests on every commit
- pytest tests/unit/

# Run integration tests on PR
- pytest tests/integration/ -m "real_model and not slow"

# Run full suite weekly
- pytest tests/
```
