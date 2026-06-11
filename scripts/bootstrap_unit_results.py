#!/usr/bin/env python3
"""Bootstrap unit-level metric sidecars into layer and best-layer summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from scipy.stats import pearsonr


DEFAULT_N_BOOTSTRAPS = 2000
DEFAULT_SEED = 20260611
DEFAULT_LOWER_PERCENTILE = 2.5
DEFAULT_UPPER_PERCENTILE = 97.5

REQUIRED_COLUMNS = [
    "run_key",
    "model_name",
    "experiment_id",
    "metric_name",
    "config_hash",
    "layer_index",
    "unit_index",
    "unit_value",
    "human_value",
    "prediction",
    "is_correct",
]
RUN_COLUMNS = [
    "run_key",
    "model_name",
    "experiment_id",
    "metric_name",
    "config_hash",
]
LAYER_SUMMARY_COLUMNS = [
    *RUN_COLUMNS,
    "layer_index",
    "mode",
    "aggregate",
    "ci_lower",
    "ci_upper",
    "n_units",
    "seed",
    "n_bootstraps",
    "lower_percentile",
    "upper_percentile",
    "value_column",
]
NULL_LAYER_SUMMARY_COLUMNS = [
    *LAYER_SUMMARY_COLUMNS,
    "null_method",
    "chance",
]
BEST_LAYER_SUMMARY_COLUMNS = [
    *RUN_COLUMNS,
    "mode",
    "best_score",
    "ci_lower",
    "ci_upper",
    "best_layer_index",
    "best_layer_median",
    "best_layer_ci_lower",
    "best_layer_ci_upper",
    "best_relative_depth",
    "best_relative_depth_median",
    "best_relative_depth_ci_lower",
    "best_relative_depth_ci_upper",
    "best_layer_counts",
    "n_layers",
    "min_n_units",
    "max_n_units",
    "seed",
    "n_bootstraps",
    "lower_percentile",
    "upper_percentile",
]
NULL_BEST_LAYER_SUMMARY_COLUMNS = [
    *BEST_LAYER_SUMMARY_COLUMNS,
    "null_method",
    "chance",
]

ACCURACY_METRICS = {
    "nights_preference_accuracy",
    "things_triplet_accuracy",
    "levels_triplet_accuracy",
}
TID_METRIC = "tid_spearman_mos"
PSYCHOPHYSICS_PEARSON_METRICS = {
    "visturing_weber_law_pearson",
}
CHANCE_BY_ACCURACY_METRIC = {
    "nights_preference_accuracy": 0.5,
    "things_triplet_accuracy": 1 / 3,
    "levels_triplet_accuracy": 1 / 3,
}
LABEL_COLUMNS = [
    "label",
    "target",
    "target_label",
    "human_label",
    "ground_truth",
    "choice",
    "choice_label",
    "human_choice",
    "human_value",
]


@dataclass(frozen=True)
class BootstrapConfig:
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS
    seed: int = DEFAULT_SEED
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE
    include_null: bool = False


@dataclass(frozen=True)
class BootstrapSummaries:
    layer_summary: pd.DataFrame
    best_layer_summary: pd.DataFrame
    null_layer_summary: pd.DataFrame
    null_best_layer_summary: pd.DataFrame


@dataclass(frozen=True)
class LayerBootstrap:
    layer_index: Any
    layer_numeric: float
    aggregate: float
    values: np.ndarray
    n_units: int
    value_column: str
    null_method: str | None = None
    chance: float | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read unit-level sidecar CSVs and write bootstrap summaries by "
            "run/layer plus best-layer optimism-aware summaries."
        )
    )
    parser.add_argument(
        "sidecar_csvs",
        nargs="+",
        type=Path,
        help="One or more unit-level sidecar CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for default summary CSV paths.",
    )
    parser.add_argument(
        "--layer-summary",
        type=Path,
        help="Output path for per-run/layer bootstrap summary.",
    )
    parser.add_argument(
        "--best-layer-summary",
        type=Path,
        help="Output path for best-layer bootstrap summary.",
    )
    parser.add_argument(
        "--null-layer-summary",
        type=Path,
        help="Output path for per-run/layer null summary; used with --include-null.",
    )
    parser.add_argument(
        "--null-best-layer-summary",
        type=Path,
        help="Output path for best-layer null summary; used with --include-null.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=DEFAULT_N_BOOTSTRAPS,
        help=f"Number of bootstrap replicates. Default: {DEFAULT_N_BOOTSTRAPS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic seed. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=DEFAULT_LOWER_PERCENTILE,
        help=f"Lower confidence percentile. Default: {DEFAULT_LOWER_PERCENTILE}.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=DEFAULT_UPPER_PERCENTILE,
        help=f"Upper confidence percentile. Default: {DEFAULT_UPPER_PERCENTILE}.",
    )
    parser.add_argument(
        "--include-null",
        action="store_true",
        help=(
            "Also write TID permutation and accuracy chance/label-permutation "
            "summaries."
        ),
    )
    return parser.parse_args(argv)


def summarize_sidecars(
    sidecar_paths: Sequence[str | Path],
    config: BootstrapConfig = BootstrapConfig(),
) -> BootstrapSummaries:
    """Load sidecars from disk and return observed/null bootstrap summaries."""
    frames = [pd.read_csv(path) for path in sidecar_paths]
    if not frames:
        raise ValueError("At least one sidecar CSV path is required")
    return summarize_unit_results(pd.concat(frames, ignore_index=True), config)


def summarize_unit_results(
    sidecars: pd.DataFrame,
    config: BootstrapConfig = BootstrapConfig(),
) -> BootstrapSummaries:
    """Return layer and best-layer bootstrap summaries for sidecar rows."""
    _validate_config(config)
    _validate_sidecar_columns(sidecars)

    layer_records: list[dict[str, Any]] = []
    best_layer_records: list[dict[str, Any]] = []
    null_layer_records: list[dict[str, Any]] = []
    null_best_layer_records: list[dict[str, Any]] = []

    for run_key, run_frame in _iter_run_groups(sidecars):
        run_values = dict(zip(RUN_COLUMNS, run_key))
        metric_name = str(run_values["metric_name"])
        mode = _metric_mode(metric_name)
        observed_layers: list[LayerBootstrap] = []
        null_layers: list[LayerBootstrap] = []

        for layer_index, layer_frame in _iter_layer_groups(run_frame):
            observed = _summarize_layer(
                layer_frame=layer_frame,
                layer_index=layer_index,
                mode=mode,
                config=config,
                purpose="observed",
            )
            observed_layers.append(observed)
            layer_records.append(
                _layer_record(run_values, observed, mode=mode, config=config)
            )

            if config.include_null and _supports_null(metric_name):
                null_layer = _summarize_null_layer(
                    layer_frame=layer_frame,
                    layer_index=layer_index,
                    metric_name=metric_name,
                    mode=mode,
                    config=config,
                )
                null_layers.append(null_layer)
                null_layer_records.append(
                    _layer_record(
                        run_values,
                        null_layer,
                        mode=mode,
                        config=config,
                        include_null=True,
                    )
                )

        best_layer_records.append(
            _best_layer_record(run_values, observed_layers, mode=mode, config=config)
        )
        if null_layers:
            null_best_layer_records.append(
                _best_layer_record(
                    run_values,
                    null_layers,
                    mode=mode,
                    config=config,
                    include_null=True,
                )
            )

    return BootstrapSummaries(
        layer_summary=_frame_from_records(layer_records, LAYER_SUMMARY_COLUMNS),
        best_layer_summary=_frame_from_records(
            best_layer_records,
            BEST_LAYER_SUMMARY_COLUMNS,
        ),
        null_layer_summary=_frame_from_records(
            null_layer_records,
            NULL_LAYER_SUMMARY_COLUMNS,
        ),
        null_best_layer_summary=_frame_from_records(
            null_best_layer_records,
            NULL_BEST_LAYER_SUMMARY_COLUMNS,
        ),
    )


def _validate_config(config: BootstrapConfig) -> None:
    if config.n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")
    if not 0 <= config.lower_percentile <= 100:
        raise ValueError("lower_percentile must be between 0 and 100")
    if not 0 <= config.upper_percentile <= 100:
        raise ValueError("upper_percentile must be between 0 and 100")
    if config.lower_percentile >= config.upper_percentile:
        raise ValueError("lower_percentile must be less than upper_percentile")


def _validate_sidecar_columns(sidecars: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(sidecars.columns))
    if missing:
        raise ValueError(
            "Missing required sidecar columns: "
            f"{missing}. Required columns: {', '.join(REQUIRED_COLUMNS)}"
        )


def _iter_run_groups(
    sidecars: pd.DataFrame,
) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
    groups = list(sidecars.groupby(RUN_COLUMNS, sort=False, dropna=False))
    return sorted(
        groups,
        key=lambda item: tuple(_sort_text(value) for value in item[0]),
    )


def _iter_layer_groups(sidecars: pd.DataFrame) -> Iterable[tuple[Any, pd.DataFrame]]:
    groups = list(sidecars.groupby("layer_index", sort=False, dropna=False))
    return sorted(groups, key=lambda item: _layer_sort_key(item[0]))


def _sort_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _layer_sort_key(value: Any) -> tuple[int, float | str]:
    numeric = _to_float(value)
    if np.isfinite(numeric):
        return (0, numeric)
    return (1, _sort_text(value))


def _to_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _metric_mode(metric_name: str) -> str:
    if metric_name in ACCURACY_METRICS:
        return "accuracy"
    if metric_name == TID_METRIC:
        return "tid_spearman"
    if metric_name in PSYCHOPHYSICS_PEARSON_METRICS:
        return "pearson"
    if metric_name.startswith("saliency_"):
        return "saliency"
    raise ValueError(
        f"Unsupported metric for unit bootstrap: {metric_name}. "
        "Supported metrics are saliency_*, tid_spearman_mos, "
        "visturing_weber_law_pearson, "
        f"and {', '.join(sorted(ACCURACY_METRICS))}."
    )


def _supports_null(metric_name: str) -> bool:
    return metric_name == TID_METRIC or metric_name in ACCURACY_METRICS


def _summarize_layer(
    layer_frame: pd.DataFrame,
    layer_index: Any,
    mode: str,
    config: BootstrapConfig,
    purpose: str,
) -> LayerBootstrap:
    rng = _rng_for(config.seed, purpose, *_group_identity(layer_frame), layer_index)
    if mode in {"accuracy", "saliency"}:
        values, value_column = _mean_values(layer_frame, mode)
        aggregate = _nanmean(values)
        bootstrap_values = _bootstrap_mean(values, rng, config.n_bootstraps)
        return LayerBootstrap(
            layer_index=layer_index,
            layer_numeric=_to_float(layer_index),
            aggregate=aggregate,
            values=bootstrap_values,
            n_units=len(values),
            value_column=value_column,
        )

    unit_values, human_values = _paired_numeric_values(layer_frame)
    if mode == "tid_spearman":
        aggregate = _spearman(unit_values, human_values)
        bootstrap_values = _bootstrap_correlation(
            unit_values,
            human_values,
            rng,
            config.n_bootstraps,
            method="spearman",
            permute_human=False,
        )
    else:
        aggregate = _pearson(unit_values, human_values)
        bootstrap_values = _bootstrap_correlation(
            unit_values,
            human_values,
            rng,
            config.n_bootstraps,
            method="pearson",
            permute_human=False,
        )
    return LayerBootstrap(
        layer_index=layer_index,
        layer_numeric=_to_float(layer_index),
        aggregate=aggregate,
        values=bootstrap_values,
        n_units=len(unit_values),
        value_column="unit_value,human_value",
    )


def _summarize_null_layer(
    layer_frame: pd.DataFrame,
    layer_index: Any,
    metric_name: str,
    mode: str,
    config: BootstrapConfig,
) -> LayerBootstrap:
    rng = _rng_for(config.seed, "null", *_group_identity(layer_frame), layer_index)
    if mode == "tid_spearman":
        unit_values, human_values = _paired_numeric_values(layer_frame)
        permuted_human = rng.permutation(human_values)
        aggregate = _spearman(unit_values, permuted_human)
        bootstrap_values = _bootstrap_correlation(
            unit_values,
            human_values,
            rng,
            config.n_bootstraps,
            method="spearman",
            permute_human=True,
        )
        return LayerBootstrap(
            layer_index=layer_index,
            layer_numeric=_to_float(layer_index),
            aggregate=aggregate,
            values=bootstrap_values,
            n_units=len(unit_values),
            value_column="unit_value,permuted_human_value",
            null_method="human_value_permutation",
            chance=None,
        )

    prediction, labels, label_column = _prediction_and_label_values(layer_frame)
    if len(prediction) > 0:
        aggregate = _permuted_accuracy(prediction, labels, rng)
        bootstrap_values = _bootstrap_label_permutation_accuracy(
            prediction,
            labels,
            rng,
            config.n_bootstraps,
        )
        return LayerBootstrap(
            layer_index=layer_index,
            layer_numeric=_to_float(layer_index),
            aggregate=aggregate,
            values=bootstrap_values,
            n_units=len(prediction),
            value_column=f"prediction,{label_column}",
            null_method="label_permutation",
            chance=CHANCE_BY_ACCURACY_METRIC.get(metric_name),
        )

    observed_values, _value_column = _mean_values(layer_frame, mode)
    chance = CHANCE_BY_ACCURACY_METRIC[metric_name]
    bootstrap_values = _bootstrap_chance(
        chance,
        len(observed_values),
        rng,
        config.n_bootstraps,
    )
    return LayerBootstrap(
        layer_index=layer_index,
        layer_numeric=_to_float(layer_index),
        aggregate=chance,
        values=bootstrap_values,
        n_units=len(observed_values),
        value_column="known_chance",
        null_method="chance",
        chance=chance,
    )


def _group_identity(layer_frame: pd.DataFrame) -> tuple[Any, ...]:
    first = layer_frame.iloc[0]
    return tuple(first[column] for column in RUN_COLUMNS)


def _mean_values(layer_frame: pd.DataFrame, mode: str) -> tuple[np.ndarray, str]:
    if mode == "accuracy":
        is_correct = _coerce_bool_numeric(layer_frame["is_correct"])
        if is_correct.notna().any():
            return is_correct.dropna().to_numpy(dtype=float), "is_correct"

    values = pd.to_numeric(layer_frame["unit_value"], errors="coerce")
    return values.dropna().to_numpy(dtype=float), "unit_value"


def _coerce_bool_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "t": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "false": 0.0,
            "f": 0.0,
            "no": 0.0,
            "n": 0.0,
        }
    )
    return numeric.where(numeric.notna(), mapped.astype("float64"))


def _paired_numeric_values(layer_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    unit_values = pd.to_numeric(layer_frame["unit_value"], errors="coerce")
    human_values = pd.to_numeric(layer_frame["human_value"], errors="coerce")
    mask = unit_values.notna() & human_values.notna()
    return (
        unit_values[mask].to_numpy(dtype=float),
        human_values[mask].to_numpy(dtype=float),
    )


def _prediction_and_label_values(
    layer_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str]:
    prediction = _clean_label_series(layer_frame["prediction"])
    for label_column in LABEL_COLUMNS:
        if label_column not in layer_frame.columns:
            continue
        labels = _clean_label_series(layer_frame[label_column])
        mask = prediction.notna() & labels.notna()
        if mask.any():
            return (
                prediction[mask].to_numpy(dtype=str),
                labels[mask].to_numpy(dtype=str),
                label_column,
            )
    return np.array([]), np.array([]), ""


def _clean_label_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_label_value)


def _normalize_label_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstraps: int,
) -> np.ndarray:
    if len(values) == 0:
        return _nan_array(n_bootstraps)

    bootstraps = np.empty(n_bootstraps, dtype=float)
    for idx in range(n_bootstraps):
        sample_idx = rng.integers(0, len(values), size=len(values))
        bootstraps[idx] = float(np.mean(values[sample_idx]))
    return bootstraps


def _bootstrap_correlation(
    unit_values: np.ndarray,
    human_values: np.ndarray,
    rng: np.random.Generator,
    n_bootstraps: int,
    *,
    method: str,
    permute_human: bool,
) -> np.ndarray:
    if len(unit_values) == 0:
        return _nan_array(n_bootstraps)

    bootstraps = np.empty(n_bootstraps, dtype=float)
    for idx in range(n_bootstraps):
        sample_idx = rng.integers(0, len(unit_values), size=len(unit_values))
        sample_unit_values = unit_values[sample_idx]
        sample_human_values = human_values[sample_idx]
        if permute_human:
            sample_human_values = rng.permutation(sample_human_values)
        if method == "spearman":
            bootstraps[idx] = _spearman(sample_unit_values, sample_human_values)
        elif method == "pearson":
            bootstraps[idx] = _pearson(sample_unit_values, sample_human_values)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
    return bootstraps


def _bootstrap_label_permutation_accuracy(
    prediction: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_bootstraps: int,
) -> np.ndarray:
    if len(prediction) == 0:
        return _nan_array(n_bootstraps)

    bootstraps = np.empty(n_bootstraps, dtype=float)
    for idx in range(n_bootstraps):
        sample_idx = rng.integers(0, len(prediction), size=len(prediction))
        sample_prediction = prediction[sample_idx]
        sample_labels = rng.permutation(labels[sample_idx])
        bootstraps[idx] = float(np.mean(sample_prediction == sample_labels))
    return bootstraps


def _bootstrap_chance(
    chance: float,
    n_units: int,
    rng: np.random.Generator,
    n_bootstraps: int,
) -> np.ndarray:
    if n_units <= 0:
        return _nan_array(n_bootstraps)
    return rng.binomial(n_units, chance, size=n_bootstraps).astype(float) / n_units


def _permuted_accuracy(
    prediction: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> float:
    if len(prediction) == 0:
        return float("nan")
    return float(np.mean(prediction == rng.permutation(labels)))


def _spearman(unit_values: np.ndarray, human_values: np.ndarray) -> float:
    if len(unit_values) < 2 or len(human_values) < 2:
        return float("nan")
    if _is_constant(unit_values) or _is_constant(human_values):
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        correlation, _p_value = spearmanr(unit_values, human_values)
    if not np.isfinite(correlation):
        return float("nan")
    return float(correlation)


def _pearson(unit_values: np.ndarray, human_values: np.ndarray) -> float:
    if len(unit_values) < 2 or len(human_values) < 2:
        return float("nan")
    if _is_constant(unit_values) or _is_constant(human_values):
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        correlation, _p_value = pearsonr(unit_values, human_values)
    if not np.isfinite(correlation):
        return float("nan")
    return float(correlation)


def _is_constant(values: np.ndarray) -> bool:
    if len(values) == 0:
        return True
    return bool(np.all(values == values[0]))


def _nanmean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _nan_array(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=float)


def _layer_record(
    run_values: dict[str, Any],
    layer: LayerBootstrap,
    *,
    mode: str,
    config: BootstrapConfig,
    include_null: bool = False,
) -> dict[str, Any]:
    record = {
        **run_values,
        "layer_index": layer.layer_index,
        "mode": mode,
        "aggregate": layer.aggregate,
        "ci_lower": _finite_percentile(layer.values, config.lower_percentile),
        "ci_upper": _finite_percentile(layer.values, config.upper_percentile),
        "n_units": layer.n_units,
        "seed": config.seed,
        "n_bootstraps": config.n_bootstraps,
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "value_column": layer.value_column,
    }
    if include_null:
        record["null_method"] = layer.null_method
        record["chance"] = layer.chance
    return record


def _best_layer_record(
    run_values: dict[str, Any],
    layers: list[LayerBootstrap],
    *,
    mode: str,
    config: BootstrapConfig,
    include_null: bool = False,
) -> dict[str, Any]:
    if not layers:
        raise ValueError(f"No layer summaries available for run {run_values['run_key']}")

    aggregates = np.array([layer.aggregate for layer in layers], dtype=float)
    best_position = _nanargmax_or_none(aggregates)
    best_layer_index = (
        layers[best_position].layer_index if best_position is not None else None
    )
    best_score = (
        aggregates[best_position] if best_position is not None else float("nan")
    )

    layer_numeric_values = np.array(
        [layer.layer_numeric for layer in layers],
        dtype=float,
    )
    relative_depths = _relative_depths(layer_numeric_values)
    best_relative_depth = (
        relative_depths[best_position]
        if best_position is not None and np.isfinite(relative_depths[best_position])
        else float("nan")
    )
    best_scores, best_layers, best_relative_depths = _best_bootstrap_distribution(
        layers,
        relative_depths,
        config.n_bootstraps,
    )
    layer_counts = Counter(
        _json_key(value) for value in best_layers if value is not None
    )

    record = {
        **run_values,
        "mode": mode,
        "best_score": float(best_score),
        "ci_lower": _finite_percentile(best_scores, config.lower_percentile),
        "ci_upper": _finite_percentile(best_scores, config.upper_percentile),
        "best_layer_index": best_layer_index,
        "best_layer_median": _finite_percentile(
            np.array([_to_float(value) for value in best_layers], dtype=float),
            50,
        ),
        "best_layer_ci_lower": _finite_percentile(
            np.array([_to_float(value) for value in best_layers], dtype=float),
            config.lower_percentile,
        ),
        "best_layer_ci_upper": _finite_percentile(
            np.array([_to_float(value) for value in best_layers], dtype=float),
            config.upper_percentile,
        ),
        "best_relative_depth": best_relative_depth,
        "best_relative_depth_median": _finite_percentile(best_relative_depths, 50),
        "best_relative_depth_ci_lower": _finite_percentile(
            best_relative_depths,
            config.lower_percentile,
        ),
        "best_relative_depth_ci_upper": _finite_percentile(
            best_relative_depths,
            config.upper_percentile,
        ),
        "best_layer_counts": json.dumps(dict(sorted(layer_counts.items()))),
        "n_layers": len(layers),
        "min_n_units": min(layer.n_units for layer in layers),
        "max_n_units": max(layer.n_units for layer in layers),
        "seed": config.seed,
        "n_bootstraps": config.n_bootstraps,
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
    }
    if include_null:
        record["null_method"] = _common_value(layer.null_method for layer in layers)
        record["chance"] = _common_value(layer.chance for layer in layers)
    return record


def _best_bootstrap_distribution(
    layers: list[LayerBootstrap],
    relative_depths: np.ndarray,
    n_bootstraps: int,
) -> tuple[np.ndarray, list[Any | None], np.ndarray]:
    score_values = np.empty(n_bootstraps, dtype=float)
    layer_values: list[Any | None] = []
    relative_values = np.empty(n_bootstraps, dtype=float)
    bootstrap_matrix = np.column_stack([layer.values for layer in layers])

    for idx in range(n_bootstraps):
        row = bootstrap_matrix[idx]
        best_position = _nanargmax_or_none(row)
        if best_position is None:
            score_values[idx] = float("nan")
            layer_values.append(None)
            relative_values[idx] = float("nan")
            continue
        score_values[idx] = row[best_position]
        layer_values.append(layers[best_position].layer_index)
        relative_values[idx] = relative_depths[best_position]

    return score_values, layer_values, relative_values


def _relative_depths(layer_numeric_values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(layer_numeric_values)):
        return _nan_array(len(layer_numeric_values))
    min_layer = float(np.min(layer_numeric_values))
    max_layer = float(np.max(layer_numeric_values))
    if max_layer == min_layer:
        return np.zeros(len(layer_numeric_values), dtype=float)
    return (layer_numeric_values - min_layer) / (max_layer - min_layer)


def _nanargmax_or_none(values: np.ndarray) -> int | None:
    if len(values) == 0 or np.all(np.isnan(values)):
        return None
    return int(np.nanargmax(values))


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return float("nan")
    return float(np.percentile(finite_values, percentile))


def _common_value(values: Iterable[Any]) -> Any:
    values_list = list(values)
    if not values_list:
        return None
    first = values_list[0]
    if all(value == first for value in values_list):
        return first
    return "mixed"


def _json_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _rng_for(seed: int, *parts: Any) -> np.random.Generator:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    child_seed = int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)
    return np.random.default_rng(child_seed)


def _frame_from_records(
    records: list[dict[str, Any]],
    columns: list[str],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns]


def write_summaries(
    summaries: BootstrapSummaries,
    *,
    output_dir: Path,
    layer_summary_path: Path | None = None,
    best_layer_summary_path: Path | None = None,
    null_layer_summary_path: Path | None = None,
    null_best_layer_summary_path: Path | None = None,
    include_null: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_summary_path = (
        layer_summary_path or output_dir / "unit_layer_bootstrap_summary.csv"
    )
    best_layer_summary_path = (
        best_layer_summary_path or output_dir / "unit_best_layer_bootstrap_summary.csv"
    )
    _write_csv(summaries.layer_summary, layer_summary_path)
    _write_csv(summaries.best_layer_summary, best_layer_summary_path)
    written = [layer_summary_path, best_layer_summary_path]

    if include_null:
        null_layer_summary_path = (
            null_layer_summary_path or output_dir / "unit_layer_null_summary.csv"
        )
        null_best_layer_summary_path = (
            null_best_layer_summary_path
            or output_dir / "unit_best_layer_null_summary.csv"
        )
        _write_csv(summaries.null_layer_summary, null_layer_summary_path)
        _write_csv(summaries.null_best_layer_summary, null_best_layer_summary_path)
        written.extend([null_layer_summary_path, null_best_layer_summary_path])

    return written


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BootstrapConfig(
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        include_null=args.include_null,
    )
    summaries = summarize_sidecars(args.sidecar_csvs, config)
    written = write_summaries(
        summaries,
        output_dir=args.output_dir,
        layer_summary_path=args.layer_summary,
        best_layer_summary_path=args.best_layer_summary,
        null_layer_summary_path=args.null_layer_summary,
        null_best_layer_summary_path=args.null_best_layer_summary,
        include_null=args.include_null,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
