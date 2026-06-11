## Summary
This PR integrates VisTuring into the `vit-human-alignment-framework` as a first-class perceptual benchmark:
- Adds full VisTuring metric support (Prop1-Prop10) in the framework pipeline.
- Enables CSV-driven execution of VisTuring metrics by metric column name.
- Aligns metric computation logic with VisTuring reference behavior (keeping framework-specific backend/layer output conventions).
- Adds test coverage and test-structure alignment for VisTuring components.

## What Is Included in the Branch Diff

### 1) VisTuring metric stack in framework
- Added/updated VisTuring metric modules and calculators under:
  - `src/metrics/visturing/*`
- Added/updated dataset loaders for VisTuring experiments:
  - `src/dataset_loaders/visturing/*`
- Added shared download utility used by VisTuring loaders:
  - `src/utils/download.py`

### 2) Runner integration (CSV execution)
- Updated `src/runner/base.py` so VisTuring metrics are executed based on CSV `metric_*` columns.
- The runner now dispatches to the corresponding Prop calculator(s) instead of a single default property path.

### 3) VisTuring math alignment (reference intent)
- Updated metric implementations so aggregation/order/correlation behavior follows VisTuring reference intent across properties.
- Preserved framework behavior of returning per-layer outputs as serialized JSON values.

### 4) Docs and examples
- Added/updated VisTuring docs:
  - `docs/VISTURING_DEFAULT_CONFIGS.md`
  - `docs/VISTURING_METRICS.md`

### 5) Tests organized by project convention
- Moved VisTuring tests into standard test layout:
  - `tests/unit/test_visturing_metrics.py`
  - `tests/integration/test_visturing_calculators.py`
- Updated shared fixtures and tests docs:
  - `tests/conftest.py`
  - `tests/README.md`

## Validation
- End-to-end run performed with VisTuring CSV test case including Prop1-Prop10 columns.
- Output metrics are produced per requested column and persisted to CSV.
- New VisTuring unit/integration tests pass.

## Notes
- Numeric parity with external VisTuring notebooks may vary when model backbones differ (e.g., PerceptNet vs ViT).
- This PR focuses on integration completeness and reference-consistent computation flow within framework conventions.
