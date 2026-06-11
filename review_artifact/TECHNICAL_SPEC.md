# Technical contract for review

This note specifies the executable contract exercised by the reviewer smoke
artifact and by full benchmark runs. It is intentionally implementation-facing:
reviewers should be able to map each statement to the repository code.

## JSON experiment schema

The JSON flow reads one object with four top-level fields:

```json
{
  "defaults": {},
  "dataset_defaults": {
    "saliency": {},
    "tid": {},
    "nights": {},
    "things": {},
    "levels": {},
    "visturing": {}
  },
  "models": ["timm::vit_base_patch16_224"],
  "experiments": [
    {
      "experiment_id": "saliency-auc-judd",
      "metric": "saliency_auc_judd",
      "config": {}
    }
  ]
}
```

- `defaults` is an object merged into every experiment.
- `dataset_defaults` is an object keyed by metric family. Valid families are
  `saliency`, `tid`, `nights`, `things`, `levels`, and `visturing`.
- `models` is a non-empty list of model names accepted by the model resolver.
- `experiments` is a non-empty list of experiment objects.
- `experiment_id` is a stable label used in `run_key`.
- `metric` must be one of the names declared in `src/runner/config_schema.py`.
- `config` is optional and contains per-experiment overrides.

The effective metric configuration is:

```text
effective_config = defaults
                 + dataset_defaults[metric_family]
                 + experiment.config
```

Later entries override earlier entries. Family-specific keys must be placed in
the corresponding `dataset_defaults` family, not in a global catch-all section.

## Model name contract

The current resolver accepts:

- internal aliases such as `vit-b16`;
- pretrained timm models as `timm::<model_name>`;
- forced-random timm models as `timm-random::<model_name>`;
- equivalent untrained aliases as `timm-untrained::<model_name>`;
- indexed Google/AugReg checkpoints as `google::<checkpoint_filename>`.

The forced-random prefixes instantiate the same timm architecture and
preprocessing but ignore pretrained weights. They use a deterministic default
seed (`random_seed=0`) during model creation and restore the process RNG state
after initialization. They are framework-native baseline models, not post-hoc
CSV columns.

## Result table schema

JSON runs persist to `results.csv` with the following columns:

| Column | Meaning |
| --- | --- |
| `run_key` | Stable key: `<model_name>__<experiment_id>`. |
| `model_name` | Exact resolver name used by the run. |
| `experiment_id` | Stable experiment label from the JSON file. |
| `metric_name` | Metric implementation name. |
| `status` | `running`, `done`, or `error`. |
| `result` | Serialized metric output. Layer-wise metrics use JSON arrays. |
| `error` | Error message when status is `error`; empty otherwise. |
| `config_hash` | SHA-256 hash of the effective metric configuration. |
| `started_at` | UTC start timestamp for the latest attempt. |
| `duration_seconds` | Runtime measured by the runner for the latest attempt. |

The `result` field stores one value per evaluated layer when the metric is
layer-wise. Nested metric outputs remain JSON-serialized instead of being
flattened into incompatible scalar columns.

## Configuration hash

`config_hash` is computed only from the effective metric configuration, not from
the model name or experiment id:

```python
json.dumps(config, sort_keys=True, separators=(",", ":"))
sha256(normalized_config.encode("utf-8")).hexdigest()
```

This makes resume behaviour auditable. If a `run_key` already exists with
status `done`, it is skipped only when the current effective configuration has
the same hash. A mismatch raises an explicit error instead of silently mixing
results produced with different dataset paths, splits, or metric parameters.

## Adapter API

Model adapters expose a common interface:

```python
forward(
    batch,
    layers: list[int] | None = None,
    return_features: bool = True,
    return_saliency: bool = False,
) -> dict[str, Any]

forward_features(batch, layers=None) -> list[layer_tensor]
forward_saliency(batch, layers=None) -> list[layer_map]
list_layers() -> list[int]
```

For timm ViTs, `forward_features` calls `get_intermediate_layers` with
`reshape=False`, `return_prefix_tokens=True`, and `norm=False`, then concatenates
the prefix token and patch tokens into tensors of shape
`[batch, num_tokens, hidden_dim]`.

For saliency, `forward_saliency` returns one rollout map per requested layer,
with shape `[batch, grid_size, grid_size]`. The maps represent accumulated
attention from the CLS token to patch tokens.

## Attention rollout contract

The timm ViT saliency path captures block attention with forward hooks and uses
the following rollout rule per layer:

1. Compute scaled dot-product attention weights from the block's `qkv`
   projection in evaluation mode.
2. Fuse heads using the configured fusion rule; the current review artifact uses
   `max`.
3. Optionally discard low attention entries while protecting all CLS-token rows
   and columns.
4. Add the residual identity contribution and row-normalize.
5. Left-multiply the accumulated rollout matrix.
6. Read the CLS-to-patch attention vector, reshape it to the patch grid, and
   normalize each map by its maximum value.

This implementation follows the attention-rollout interpretation used for ViT
saliency while keeping the exact choices explicit.

## Metric-family mappings

- **Saliency**: compares rollout maps against human fixation/saliency targets;
  AUC-Judd uses fixation maps and Pearson uses continuous saliency maps.
- **TID**: maps reference/distorted image representations to a distance or
  similarity signal and correlates it with human mean opinion scores (MOS).
- **NIGHTS**: computes preference accuracy by selecting the model option that
  best matches the human preference trial.
- **THINGS**: computes triplet accuracy on human object-similarity judgements.
- **LEVELS**: computes triplet accuracy for between-class, class-border, or
  within-class splits.
- **Custom psychophysical metrics**: compute agreement with controlled
  psychophysical response curves; the framework paper uses one Weber-law
  subtest only as an extensibility example.

## Bootstrap units

Confidence intervals should be computed over the experimental unit, not over
layers:

| Family | Bootstrap unit |
| --- | --- |
| Saliency | image |
| TID | distorted/reference pair |
| NIGHTS | preference trial |
| THINGS | triplet trial |
| LEVELS | triplet trial |
| Custom psychophysics | condition or trial |

The current `results.csv` contract stores aggregate layer-wise values. Future
CI runs should persist optional per-unit outputs in a sidecar artifact, leaving
the stable aggregate result table backward-compatible.

Chance-level controls are metric-family references rather than model outputs.
For binary preference metrics such as NIGHTS the analytical chance level is
0.5. For odd-one-out triplets such as THINGS and LEVELS it is 1/3. Saliency and
correlation metrics require random, center-bias, permutation, or null-model
controls instead of a single universal chance value.

## Execution pseudocode

```text
read JSON payload
validate dataset_defaults families
validate every model name
for each experiment:
    validate metric name
    merge defaults, family defaults, and experiment config
    validate allowed keys and closed-value options
for each model:
    resolve adapter
    for each resolved experiment:
        if results.csv has matching done row and config_hash: skip
        mark row running
        run dataset loader, adapter forward pass, and metric calculator
        write serialized layer-wise result as done
        on error, write error row
    release model
```
