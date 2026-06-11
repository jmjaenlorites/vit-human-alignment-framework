# Review artifact manifest

The anonymized review package should include the following files and folders:

```text
AGENTS.md
README.md
README_REVIEW.md
pyproject.toml
uv.lock
pytest.ini
main.py
src/
tests/
review_artifact/
data/experiments.example.json
data/experiments.timm.model-battery.json
data/experiments.framework-baselines.json
data/experiments.saliency-baselines.json
data/baselines/saliency_mit1003_reference.json
docs/baselines_saliency.md
scripts/recompute_saliency_baselines.py
results-model-battery.csv
results-saliency-baselines.csv
results-framework-baselines-seed0.csv
```

Exclude `.git/`, `.venv/`, `__pycache__/`, local notebooks, private datasets,
downloaded checkpoints, user-specific absolute paths, temporary LaTeX files,
and any server logs that could identify users or infrastructure beyond what is
needed for review.

## Reviewer commands

From the repository root:

```bash
uv sync
uv run python review_artifact/smoke_json_runner.py
uv run pytest tests/unit review_artifact/test_smoke_json_runner.py
python3 scripts/recompute_saliency_baselines.py --check-source-files --strict
```

The smoke command must complete without GPUs, real ViT checkpoints, or full
perceptual datasets. The full case-study output tables are included as committed
CSV files so reviewers can inspect the empirical artifact even when licensed
datasets cannot be redistributed.

## Full experiments

Full experiments require the original datasets and checkpoints. In this project
they are run on the dedicated shared server with:

```bash
rsync -av --exclude=".venv" --exclude="ejemplos_pablo" ./ megatron.uv.es:~/vit-human-alignment-framework/

ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

For a specific config:

```bash
ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-baselines.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-baselines-seed0.csv /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

The server is shared. Before launching a long run, check the JSON path, output
path, expected dataset availability, resumability state, and only your own
related running processes. Do not kill processes, delete checkpoints, clean
caches, remove result directories, or repair permissions unless the target is a
clearly owned artifact from a run just launched by this project. For anything
that could affect other users, ask before acting.
