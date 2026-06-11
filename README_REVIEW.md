# Reviewer Reproducibility Artifact

This repository includes a lightweight smoke artifact for reviewers who want to
verify the JSON experiment contract without downloading ViT checkpoints or large
benchmark datasets.

The smoke artifact lives in `review_artifact/` and intentionally uses fake model
resolution plus deterministic fake metric outputs. It still runs the real JSON
experiment source and `results.csv` persistence/resume path.

Reviewer-facing contract details are specified in:

- `review_artifact/TECHNICAL_SPEC.md`;
- `review_artifact/MANIFEST.md`;
- `review_artifact/dataset_units.csv`.

## Setup

From the repository root:

```bash
uv sync
```

## Smoke Execution

Run the smoke artifact:

```bash
uv run python review_artifact/smoke_json_runner.py
```

Optional pytest check:

```bash
uv run pytest review_artifact/test_smoke_json_runner.py
```

The script reads `review_artifact/smoke_experiments.json`, writes a temporary
`results.csv`, runs the JSON runner twice, and compares the stable output
columns with `review_artifact/expected_results.csv`.

Expected terminal output is similar to:

```text
Smoke artifact OK
  config: review_artifact/smoke_experiments.json
  results: /tmp/vit-review-smoke-.../results.csv
  normalized: /tmp/vit-review-smoke-.../results.normalized.csv
  first pass: loaded fake model 1 time(s), ran 2 metric(s)
  resume pass: loaded fake model 0 time(s), ran 0 metric(s)
  expected CSV: review_artifact/expected_results.csv
```

The raw `results.csv` also contains volatile timing columns (`started_at` and
`duration_seconds`). The committed expected CSV omits those volatile columns and
checks the stable reviewer-facing contract: run key, model, experiment id,
metric, status, result, error, and config hash.

## What This Covers

- JSON config validation for supported model and metric names.
- Merge behavior for global `defaults`, family-scoped `dataset_defaults`, and
  per-experiment `config`.
- `results.csv` creation through the real `ResultsStore`.
- Resume behavior: rows already marked `done` are skipped on the second run.
- A pytest entry point that reviewers can run without GPUs or benchmark data.

## What This Does Not Cover

- Real ViT checkpoint loading from `timm` or Google/JAX checkpoints.
- GPU execution, VRAM behavior, or backend-specific numerical differences.
- Full benchmark datasets such as the original saliency, TID, Levels, NIGHTS,
  THINGS, or custom psychophysical data.
- Metric numerical validity on real human-alignment benchmarks.

Full benchmark experiments require the original datasets/checkpoints and should
be run on the dedicated experiment server, not through this smoke artifact.

## Running Full Experiments on the Dedicated Server

Full experiments are intended to run on the dedicated shared server. The
agreed sync and execution commands are:

```bash
rsync -av --exclude=".venv" --exclude="ejemplos_pablo" ./ megatron.uv.es:~/vit-human-alignment-framework/

ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

By default, `main.py` runs the JSON model-battery configuration currently wired
in the repository and persists to `results-model-battery.csv`. The same entry
point can run another JSON configuration through environment variables. For the
framework random-ViT baseline configuration:

```bash
ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-baselines.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-baselines.csv /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

The older saliency-only baseline configuration remains available as
`data/experiments.saliency-baselines.json` because it produced the committed
`results-saliency-baselines.csv` table used as the first framework-native
baseline run.

Forced-random timm baselines use a deterministic seed during model creation
(`random_seed=0`) and restore the process RNG state afterwards. Use a fresh
output file when comparing against results produced before that deterministic
contract existed.

The committed `results-framework-baselines-seed0.csv` contains the completed
seeded controls for saliency AUC-Judd, TID Spearman MOS, NIGHTS preference
accuracy, THINGS triplet accuracy, and LEVELS between-class accuracy. The same
run hit a non-destructive `Broken pipe` error on `levels-class-border-random-vit`
before reaching the remaining rows; those unfinished controls should be
recomputed from `data/experiments.framework-baselines.json` before making claims
about all baseline families.

Confirm the dataset paths and checkpoint availability on the server before
launching any full experiment.

The server is shared. Before launching long runs, check that the intended
configuration, output path and resumability state are correct, and inspect only
your own related processes if a previous execution may still be running. Avoid
`rsync --delete`, broad cleanup commands, killing processes, or deleting caches,
checkpoints or result directories unless the target is known to be yours and the
action has been explicitly approved. If permissions, disk space, stuck jobs or
ambiguous paths become blockers, stop and ask before attempting a destructive or
invasive fix.
