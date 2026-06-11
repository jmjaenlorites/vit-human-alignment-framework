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
data/experiments.framework-case-study.json
data/experiments.framework-baselines.json
data/experiments.framework-saliency-baselines.json
data/experiments.framework-rollout-sensitivity.json
data/experiments.framework-pooling-ablation.json
data/experiments.saliency-baselines.json
data/baselines/saliency_mit1003_reference.json
docs/baselines_saliency.md
scripts/recompute_saliency_baselines.py
scripts/bootstrap_unit_results.py
scripts/framework_paper_outputs.py
results-model-battery.csv
results-saliency-baselines.csv
results-framework-baselines-seed0.csv
results-framework-baselines-seed0-v3.csv
results-framework-case-study-v2.csv
results-framework-rollout-sensitivity.csv
results-framework-pooling-ablation.csv
outputs/bootstrap-framework-baselines/
outputs/bootstrap-framework-case-study/
outputs/bootstrap-framework-rollout-sensitivity/
outputs/bootstrap-framework-pooling-ablation/
outputs/bootstrap-framework-saliency-baselines/
outputs/framework_paper/
review_artifact/unit_result_archives/unit-results-framework-baselines-seed0-v3.tar.gz
review_artifact/unit_result_archives/unit-results-framework-case-study-v2.tar.gz
review_artifact/unit_result_archives/unit-results-framework-rollout-sensitivity.tar.gz
review_artifact/unit_result_archives/unit-results-framework-pooling-ablation.tar.gz
review_artifact/unit_result_archives/unit-results-framework-saliency-baselines.tar.gz
```

Exclude `.git/`, `.venv/`, `__pycache__/`, local notebooks, private datasets,
downloaded checkpoints, raw extracted `unit-results-framework-*/` directories,
user-specific absolute paths, temporary LaTeX files, and any server logs that
could identify users or infrastructure beyond what is needed for review.

## Reviewer commands

From the repository root:

```bash
uv sync
uv run python review_artifact/smoke_json_runner.py
uv run pytest tests/unit review_artifact/test_smoke_json_runner.py
python3 scripts/recompute_saliency_baselines.py --check-source-files --strict
uv run python scripts/bootstrap_unit_results.py --help
tar -xzf review_artifact/unit_result_archives/unit-results-framework-baselines-seed0-v3.tar.gz
tar -xzf review_artifact/unit_result_archives/unit-results-framework-case-study-v2.tar.gz
tar -xzf review_artifact/unit_result_archives/unit-results-framework-rollout-sensitivity.tar.gz
tar -xzf review_artifact/unit_result_archives/unit-results-framework-pooling-ablation.tar.gz
tar -xzf review_artifact/unit_result_archives/unit-results-framework-saliency-baselines.tar.gz
uv run python scripts/bootstrap_unit_results.py unit-results-framework-baselines-seed0-v3/*.csv --output-dir outputs/bootstrap-framework-baselines --include-null
uv run python scripts/bootstrap_unit_results.py unit-results-framework-case-study-v2/*.csv --output-dir outputs/bootstrap-framework-case-study --include-null
uv run python scripts/bootstrap_unit_results.py unit-results-framework-rollout-sensitivity/*.csv --output-dir outputs/bootstrap-framework-rollout-sensitivity --include-null
uv run python scripts/bootstrap_unit_results.py unit-results-framework-pooling-ablation/*.csv --output-dir outputs/bootstrap-framework-pooling-ablation --include-null
uv run python scripts/framework_paper_outputs.py --case-results results-framework-case-study-v2.csv --case-bootstrap-dir outputs/bootstrap-framework-case-study --baseline-results results-framework-baselines-seed0-v3.csv --baseline-bootstrap-dir outputs/bootstrap-framework-baselines --saliency-baseline-results results-framework-saliency-baselines.csv --saliency-baseline-bootstrap-dir outputs/bootstrap-framework-saliency-baselines --rollout-results results-framework-rollout-sensitivity.csv --rollout-bootstrap-dir outputs/bootstrap-framework-rollout-sensitivity --pooling-results results-framework-pooling-ablation.csv --pooling-bootstrap-dir outputs/bootstrap-framework-pooling-ablation --output-dir outputs/framework_paper
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
ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-baselines.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-baselines-seed0-v3.csv VIT_ALIGNMENT_UNIT_RESULTS_DIR=unit-results-framework-baselines-seed0-v3 /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

Saliency-only center/uniform controls use a separate config because those
adapters intentionally do not expose feature representations:

```bash
ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-saliency-baselines.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-saliency-baselines.csv VIT_ALIGNMENT_UNIT_RESULTS_DIR=unit-results-framework-saliency-baselines /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

The compact framework-paper case study uses:

```bash
ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-case-study.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-case-study-v2.csv VIT_ALIGNMENT_UNIT_RESULTS_DIR=unit-results-framework-case-study-v2 /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

Supplementary sensitivity runs use:

```bash
ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-rollout-sensitivity.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-rollout-sensitivity.csv VIT_ALIGNMENT_UNIT_RESULTS_DIR=unit-results-framework-rollout-sensitivity /media/disk/users/jmjaen/.local/bin/uv run main.py"

ssh deep.uv.es "cd ~/vit-human-alignment-framework/ && VIT_ALIGNMENT_JSON_PATH=data/experiments.framework-pooling-ablation.json VIT_ALIGNMENT_RESULTS_PATH=results-framework-pooling-ablation.csv VIT_ALIGNMENT_UNIT_RESULTS_DIR=unit-results-framework-pooling-ablation /media/disk/users/jmjaen/.local/bin/uv run main.py"
```

The server is shared. Before launching a long run, check the JSON path, output
path, expected dataset availability, resumability state, and only your own
related running processes. Do not kill processes, delete checkpoints, clean
caches, remove result directories, or repair permissions unless the target is a
clearly owned artifact from a run just launched by this project. For anything
that could affect other users, ask before acting.
