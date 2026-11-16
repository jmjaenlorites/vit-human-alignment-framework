# vit-human-alignment

A Python framework for evaluating the perceptual alignment of Vision Transformers
(ViTs) with human vision, **layer by layer**.

You declare in a single JSON file which models, datasets and metrics to run. The
framework extracts the representation (or an attention-rollout saliency map) of every
layer, computes each perceptual metric on its own scale, and stores resumable,
configuration-hashed results. Optional unit-level sidecars (one row per image, pair or
trial) feed bootstrap confidence intervals and permutation nulls without re-running the
models.

## Features

- **Declarative experiments**: JSON configuration with global `defaults`,
  family-scoped `dataset_defaults` and per-experiment `config`, validated before any
  model is loaded.
- **Layer-wise evaluation**: one score per layer for every model–experiment pair.
- **Perceptual test battery**:
  - saliency: MIT1003, AUC-Judd and Pearson correlation on attention-rollout maps;
  - image quality: TID2013, Spearman correlation with MOS;
  - two-alternative preference: NIGHTS, accuracy;
  - odd-one-out triplets: LEVELS (between-class, class-border, within-class) and
    THINGS, accuracy;
  - low-level psychophysics: the ten properties of the
    [visturing](https://doi.org/10.3389/frai.2025.1665874) test, as twelve
    Pearson/Kendall metrics. Stimuli and ground truth are downloaded automatically
    from Zenodo.
- **Models**: any ViT in [timm](https://github.com/huggingface/pytorch-image-models)
  (`timm::<name>`), its untrained version (`timm-random::<name>`), internal aliases
  (`vit-b16`, …) and Google/AugReg JAX checkpoints.
- **Baselines and uncertainty**: random-weight ViTs, center-Gaussian and uniform
  saliency priors, unit-level sidecars, bootstrap intervals and permutation nulls.
- **Resumable results**: completed runs are skipped on relaunch; a configuration-hash
  mismatch raises an error instead of mixing incompatible results.

## Installation

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # CPU / macOS
uv sync --extra cuda     # Linux x86_64 with CUDA 12 (JAX GPU wheels)
```

## Quickstart

1. Write a configuration file, e.g. `config.json`:

```json
{
  "defaults": {"batch_size": 32},
  "dataset_defaults": {
    "saliency": {"dataset_path": "data/MIT1003", "discard_ratio": 0.95},
    "tid": {"dataset_path": "data/TID2013", "representation_pooling": "flatten"}
  },
  "models": ["timm::vit_base_patch16_224.augreg_in1k"],
  "experiments": [
    {"experiment_id": "mit1003-auc-judd", "metric": "saliency_auc_judd",
     "config": {"head_fusion": "max"}},
    {"experiment_id": "tid2013-spearman", "metric": "tid_spearman_mos"}
  ]
}
```

2. Run it:

```bash
export VIT_ALIGNMENT_JSON_PATH=config.json
export VIT_ALIGNMENT_RESULTS_PATH=results.csv
export VIT_ALIGNMENT_UNIT_RESULTS_DIR=unit_results   # optional: per-unit sidecars
uv run main.py
```

`results.csv` gets one row per model–experiment pair, with the score of every layer,
the run status, duration and configuration hash. Relaunching the same command skips
completed rows.

3. Optionally, add bootstrap intervals and permutation nulls from the sidecars:

```bash
uv run python scripts/bootstrap_unit_results.py unit_results/*.csv \
    --output-dir outputs/bootstrap --include-null
```

The effective configuration of each run is obtained by applying `defaults`, then the
`dataset_defaults` of the metric family (`saliency`, `tid`, `nights`, `levels`,
`things`, `visturing`), then the experiment's `config`. Unknown keys and invalid values
fail early with the list of allowed options. More examples are in `data/`.

## Datasets

MIT1003, TID2013, NIGHTS, LEVELS and THINGS are not redistributed: obtain them from
their original sources and set `dataset_path` (or `levels_path` and `imagenet_path` for
LEVELS) in `dataset_defaults`. The visturing data are downloaded automatically.

## Extending the framework

- **New model**: any timm checkpoint only needs its name in `models`. Other libraries
  need an adapter implementing `forward_features` (one tensor per layer) and, for
  saliency tests, `forward_saliency` (see `src/models/`).
- **New perceptual test**: add a dataset loader (`src/dataset_loaders/`), a metric
  class implementing `reset`, `calculate` and `finalize` (`src/metrics/base.py`),
  its entry in `src/runner/config_schema.py` and its dispatch in the runner. Result
  storage, hashing, sidecars, bootstrap and plotting are reused unchanged.

## Repository layout

```text
main.py                     entry point (reads VIT_ALIGNMENT_* environment variables)
src/models/                 model adapters and resolver (timm::, timm-random::)
src/dataset_loaders/        MIT1003, TID2013, NIGHTS, LEVELS, THINGS, visturing
src/metrics/                metric calculators (reset/calculate/finalize)
src/runner/                 JSON schema, validation, runner and results store
src/utils/                  enums, shared types, dataset download
data/                       example JSON configurations
scripts/                    bootstrap/null summaries, paper figures and tables
review_artifact/            smoke test with fixture data and a fake model
tests/unit, tests/integration
docs/                       design documentation (Spanish)
```

## Tests

```bash
uv run pytest tests/unit                               # fast, no downloads
uv run python review_artifact/smoke_json_runner.py      # end-to-end smoke test
uv run pytest tests/integration                         # slower; some tests need local data
```

## License

Apache License 2.0 (see `LICENSE`). The visturing metrics in `src/metrics/visturing/`
and `src/dataset_loaders/visturing/` are adapted from the
[visturing](https://doi.org/10.3389/frai.2025.1665874) reference implementation, also
released under the Apache License 2.0.

## Reviewer artifact

See `README_REVIEW.md` for the reviewer smoke artifact and the commands used to
reproduce the paper figures and tables.
