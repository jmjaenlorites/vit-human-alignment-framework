#!/usr/bin/env python3
"""Generate framework-paper tables and figures from result CSVs and sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


MODEL_GROUPS = {
    "training_recipe": [
        "timm::vit_base_patch16_224.augreg_in1k",
        "timm::vit_base_patch16_224.mae",
        "timm::vit_base_patch16_224.dino",
        "timm::vit_base_patch16_clip_224.laion2b_ft_in1k",
    ],
    "dataset_scale": [
        "timm::vit_base_patch16_224.augreg_in1k",
        "timm::vit_base_patch16_224.augreg_in21k",
    ],
    "image_resolution": [
        "timm::vit_base_patch16_224.augreg_in1k",
        "timm::vit_base_patch16_384.augreg_in1k",
    ],
}

GROUP_TITLES = {
    "training_recipe": "Training recipe",
    "dataset_scale": "Dataset scale",
    "image_resolution": "Image resolution",
}

MODEL_LABELS = {
    "timm::vit_base_patch16_224.augreg_in1k": "AugReg IN1k",
    "timm::vit_base_patch16_224.mae": "MAE",
    "timm::vit_base_patch16_224.dino": "DINO",
    "timm::vit_base_patch16_clip_224.laion2b_ft_in1k": "CLIP LAION2B->IN1k",
    "timm::vit_base_patch16_224.augreg_in21k": "AugReg IN21k",
    "timm::vit_base_patch16_384.augreg_in1k": "AugReg IN1k 384",
    "timm-random::vit_base_patch16_224": "Random ViT-B/16",
    "saliency-baseline::uniform": "Uniform saliency",
    "saliency-baseline::center-gaussian": "Center Gaussian saliency",
}

METRIC_LABELS = {
    "saliency-auc-judd": "Saliency AUC-Judd",
    "tid-spearman-mos": "TID Spearman MOS",
    "nights-preference-accuracy": "NIGHTS preference",
    "things-triplet-accuracy": "THINGS triplet",
    "levels-between-class": "LEVELS between class",
    "levels-class-border": "LEVELS class border",
    "levels-within-class": "LEVELS within class",
    "psychophysics-weber-law-pearson": "Custom psychophysical metric",
    "saliency-pearson": "Saliency PCC",
}

CENTRAL_METRICS = [
    "saliency-auc-judd",
    "tid-spearman-mos",
    "nights-preference-accuracy",
    "levels-between-class",
]

CONTRASTS = [
    {
        "factor": "Training recipe",
        "contrast": "MAE - AugReg IN1k",
        "reference": "timm::vit_base_patch16_224.augreg_in1k",
        "comparison": "timm::vit_base_patch16_224.mae",
    },
    {
        "factor": "Training recipe",
        "contrast": "DINO - AugReg IN1k",
        "reference": "timm::vit_base_patch16_224.augreg_in1k",
        "comparison": "timm::vit_base_patch16_224.dino",
    },
    {
        "factor": "Training recipe",
        "contrast": "CLIP - AugReg IN1k",
        "reference": "timm::vit_base_patch16_224.augreg_in1k",
        "comparison": "timm::vit_base_patch16_clip_224.laion2b_ft_in1k",
    },
    {
        "factor": "Dataset scale",
        "contrast": "AugReg IN21k - AugReg IN1k",
        "reference": "timm::vit_base_patch16_224.augreg_in1k",
        "comparison": "timm::vit_base_patch16_224.augreg_in21k",
    },
    {
        "factor": "Image resolution",
        "contrast": "AugReg 384px - AugReg 224px",
        "reference": "timm::vit_base_patch16_224.augreg_in1k",
        "comparison": "timm::vit_base_patch16_384.augreg_in1k",
    },
]

SUPPLEMENT_METRICS = [
    "saliency-auc-judd",
    "tid-spearman-mos",
    "nights-preference-accuracy",
    "levels-between-class",
    "levels-class-border",
    "levels-within-class",
    "psychophysics-weber-law-pearson",
]

BASELINE_METRIC_NAMES = {
    "saliency_auc_judd",
    "tid_spearman_mos",
    "nights_preference_accuracy",
    "things_triplet_accuracy",
    "levels_triplet_accuracy",
}

ACCURACY_METRIC_NAMES = {
    "nights_preference_accuracy",
    "things_triplet_accuracy",
    "levels_triplet_accuracy",
}

DELTA_TABLE_COLUMNS = [
    "factor",
    "contrast",
    "metric_label",
    "reference_model",
    "comparison_model",
    "final_delta",
    "final_ci_lower",
    "final_ci_upper",
    "final_support",
    "best_delta",
    "best_ci_lower",
    "best_ci_upper",
    "best_support",
    "n_units",
    "reference_n_layers",
    "comparison_n_layers",
    "seed",
    "n_bootstraps",
]

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "P", "X"]

# Main-text grids are drawn at their printed size (journal text width) so that
# font sizes in points are the ones the reader sees; no title is embedded because
# the LaTeX caption names the figure.
MAIN_FIGURE_SIZE_IN = (5.15, 2.7)
MAIN_FIGURE_RC = {
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}
# A null reference further below the plotted curves than their own range is listed
# in the legend with its value instead of drawn, so it does not flatten the panel.
NULL_OFF_SCALE_GAP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create figures and tables for the framework paper."
    )
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--case-bootstrap-dir", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--baseline-bootstrap-dir", type=Path)
    parser.add_argument("--saliency-baseline-results", type=Path)
    parser.add_argument("--saliency-baseline-bootstrap-dir", type=Path)
    parser.add_argument("--rollout-results", type=Path)
    parser.add_argument("--rollout-bootstrap-dir", type=Path)
    parser.add_argument("--pooling-results", type=Path)
    parser.add_argument("--pooling-bootstrap-dir", type=Path)
    parser.add_argument("--case-unit-dir", type=Path)
    parser.add_argument("--contrast-n-bootstraps", type=int, default=2000)
    parser.add_argument("--contrast-seed", type=int, default=20260611)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/framework_paper"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    case_layerwise = load_layerwise_results(args.case_results)
    case_layer_bootstrap = load_layer_bootstrap(args.case_bootstrap_dir)
    case_best_bootstrap = load_best_bootstrap(args.case_bootstrap_dir)
    case_layerwise = attach_layer_ci(case_layerwise, case_layer_bootstrap)
    selected_scores = summarize_scores(case_layerwise, case_best_bootstrap)
    selected_scores.to_csv(tables_dir / "framework_selected_model_scores_with_ci.csv", index=False)
    write_framework_run_summary(
        case_results=args.case_results,
        baseline_results=args.baseline_results,
        saliency_baseline_results=args.saliency_baseline_results,
        rollout_results=args.rollout_results,
        pooling_results=args.pooling_results,
        sidecar_archives_dir=Path("review_artifact/unit_result_archives"),
        output_path=tables_dir / "framework_run_summary.csv",
    )
    write_artifact_verification_table(
        output_path=tables_dir / "framework_artifact_verification.csv",
    )

    if args.case_unit_dir and args.case_unit_dir.exists():
        contrast_delta_table = build_contrast_delta_table(
            args.case_unit_dir,
            n_bootstraps=args.contrast_n_bootstraps,
            seed=args.contrast_seed,
        )
        contrast_delta_table.to_csv(
            tables_dir / "framework_prespecified_contrast_deltas.csv",
            index=False,
        )

    baseline_table = build_baseline_table(
        baseline_results=args.baseline_results,
        baseline_bootstrap_dir=args.baseline_bootstrap_dir,
        saliency_results=args.saliency_baseline_results,
        saliency_bootstrap_dir=args.saliency_baseline_bootstrap_dir,
    )
    baseline_layerwise = load_baseline_layerwise(
        baseline_results=args.baseline_results,
        baseline_bootstrap_dir=args.baseline_bootstrap_dir,
        saliency_results=args.saliency_baseline_results,
        saliency_bootstrap_dir=args.saliency_baseline_bootstrap_dir,
    )
    if baseline_table is not None:
        baseline_table.to_csv(tables_dir / "framework_baselines_with_ci.csv", index=False)
        baseline_context = build_baseline_context_table(selected_scores, baseline_table)
        baseline_context.to_csv(tables_dir / "framework_baseline_context.csv", index=False)
        plot_baseline_context(
            baseline_context,
            output_path=figures_dir / "framework_main_baseline_context_2x2.png",
        )

    for group_name, models in MODEL_GROUPS.items():
        plot_metric_grid(
            case_layerwise,
            models=models,
            metrics=CENTRAL_METRICS,
            title=None,
            output_path=figures_dir / f"framework_main_{group_name}_2x2.png",
            nrows=2,
            ncols=2,
            baseline_layerwise=baseline_layerwise,
            baseline_summary=baseline_table,
            include_baselines=True,
            compact=True,
        )
        plot_metric_grid(
            case_layerwise,
            models=models,
            metrics=SUPPLEMENT_METRICS,
            title=f"{GROUP_TITLES[group_name]}: selected metric battery",
            output_path=figures_dir / f"framework_supplement_{group_name}_4x2.png",
            nrows=4,
            ncols=2,
            baseline_layerwise=baseline_layerwise,
            baseline_summary=baseline_table,
            include_baselines=True,
        )

    write_optional_summary(
        results_path=args.rollout_results,
        bootstrap_dir=args.rollout_bootstrap_dir,
        output_path=tables_dir / "framework_rollout_sensitivity.csv",
    )
    write_optional_summary(
        results_path=args.pooling_results,
        bootstrap_dir=args.pooling_bootstrap_dir,
        output_path=tables_dir / "framework_pooling_ablation.csv",
    )


def load_layerwise_results(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for record in raw.to_dict("records"):
        if record.get("status") != "done":
            continue
        parsed = json.loads(record["result"])
        if not isinstance(parsed, list):
            parsed = [parsed]
        n_layers = len(parsed)
        denom = max(n_layers - 1, 1)
        for layer_index, value in enumerate(parsed):
            rows.append(
                {
                    "run_key": record["run_key"],
                    "model_name": record["model_name"],
                    "model_label": MODEL_LABELS.get(record["model_name"], record["model_name"]),
                    "experiment_id": normalize_experiment_id(record["experiment_id"]),
                    "metric_name": record["metric_name"],
                    "layer_index": layer_index,
                    "n_layers": n_layers,
                    "relative_depth": layer_index / denom,
                    "raw_score": scalarize_layer(value, record["metric_name"]),
                }
            )
    return pd.DataFrame(rows)


def normalize_experiment_id(experiment_id: str) -> str:
    if experiment_id.endswith("-random-vit"):
        experiment_id = experiment_id.removesuffix("-random-vit")
    if experiment_id == "saliency-auc-judd-baseline":
        return "saliency-auc-judd"
    if experiment_id == "saliency-pearson-baseline":
        return "saliency-pearson"
    if experiment_id == "visturing-weber-law-pearson":
        return "psychophysics-weber-law-pearson"
    if experiment_id == "psychophysics-weber-law-pearson":
        return experiment_id
    if experiment_id == "psychophysics-weber-law-pearson-random-vit":
        return "psychophysics-weber-law-pearson"
    return experiment_id


def scalarize_layer(value: Any, metric_name: str) -> float:
    leaves = collect_numeric_leaves(value, target_stat(metric_name))
    if not leaves:
        return float("nan")
    return float(np.mean(leaves))


def target_stat(metric_name: str) -> str | None:
    if "kendall" in metric_name:
        return "kendall"
    if "spearman" in metric_name:
        return "spearman"
    if "pearson" in metric_name:
        return "pearson"
    return None


def collect_numeric_leaves(value: Any, target: str | None) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        number = float(value)
        return [number] if math.isfinite(number) else []
    if isinstance(value, list):
        leaves: list[float] = []
        for item in value:
            leaves.extend(collect_numeric_leaves(item, target))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for key, item in value.items():
            key_matches = target is None or key == target or key.startswith(f"{target}_")
            if key_matches:
                leaves.extend(collect_numeric_leaves(item, target))
            elif isinstance(item, dict | list):
                leaves.extend(collect_numeric_leaves(item, target))
        return leaves
    return []


def load_layer_bootstrap(bootstrap_dir: Path | None) -> pd.DataFrame:
    if bootstrap_dir is None:
        return pd.DataFrame()
    path = bootstrap_dir / "unit_layer_bootstrap_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["experiment_id"] = frame["experiment_id"].map(normalize_experiment_id)
    return drop_nonreported_ci_metrics(frame)


def load_best_bootstrap(bootstrap_dir: Path | None) -> pd.DataFrame:
    if bootstrap_dir is None:
        return pd.DataFrame()
    path = bootstrap_dir / "unit_best_layer_bootstrap_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["experiment_id"] = frame["experiment_id"].map(normalize_experiment_id)
    return drop_nonreported_ci_metrics(frame)


def load_null_best_bootstrap(bootstrap_dir: Path | None) -> pd.DataFrame:
    if bootstrap_dir is None:
        return pd.DataFrame()
    path = bootstrap_dir / "unit_best_layer_null_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["experiment_id"] = frame["experiment_id"].map(normalize_experiment_id)
    return drop_nonreported_ci_metrics(frame)


def drop_nonreported_ci_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["metric_name"] != "visturing_weber_law_pearson"].copy()


def attach_layer_ci(layerwise: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    if bootstrap.empty:
        layerwise["ci_lower"] = np.nan
        layerwise["ci_upper"] = np.nan
        return layerwise
    merged = layerwise.merge(
        bootstrap[
            [
                "model_name",
                "experiment_id",
                "metric_name",
                "layer_index",
                "ci_lower",
                "ci_upper",
                "n_units",
            ]
        ],
        on=["model_name", "experiment_id", "metric_name", "layer_index"],
        how="left",
    )
    return merged


def summarize_scores(layerwise: pd.DataFrame, best_bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in layerwise.groupby(["model_name", "experiment_id", "metric_name"], sort=False):
        model_name, experiment_id, metric_name = key
        finite = group.dropna(subset=["raw_score"])
        if finite.empty:
            best = group.sort_values("layer_index").tail(1).iloc[0]
        else:
            best = finite.loc[finite["raw_score"].idxmax()]
        final = group.sort_values("layer_index").tail(1).iloc[0]
        best_ci = lookup_best_ci(best_bootstrap, model_name, experiment_id, metric_name)
        final_ci = lookup_layer_ci(final)
        rows.append(
            {
                "model_name": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "experiment_id": experiment_id,
                "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
                "metric_name": metric_name,
                "n_layers": int(group["n_layers"].max()),
                "best_score": best["raw_score"],
                "best_layer": int(best["layer_index"]),
                "best_relative_depth": best["relative_depth"],
                "best_ci_lower": best_ci[0],
                "best_ci_upper": best_ci[1],
                "best_n_units": _optional_int(best.get("n_units", np.nan)),
                "final_score": final["raw_score"],
                "final_layer": int(final["layer_index"]),
                "final_relative_depth": final["relative_depth"],
                "final_ci_lower": final_ci[0],
                "final_ci_upper": final_ci[1],
                "final_n_units": _optional_int(final.get("n_units", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _optional_int(value: Any) -> int | float:
    if pd.isna(value):
        return float("nan")
    return int(value)


def lookup_best_ci(
    best_bootstrap: pd.DataFrame,
    model_name: str,
    experiment_id: str,
    metric_name: str,
) -> tuple[float, float]:
    if best_bootstrap.empty:
        return float("nan"), float("nan")
    match = best_bootstrap[
        (best_bootstrap["model_name"] == model_name)
        & (best_bootstrap["experiment_id"] == experiment_id)
        & (best_bootstrap["metric_name"] == metric_name)
    ]
    if match.empty:
        return float("nan"), float("nan")
    row = match.iloc[0]
    return float(row["ci_lower"]), float(row["ci_upper"])


def lookup_layer_ci(row: pd.Series) -> tuple[float, float]:
    return float(row.get("ci_lower", np.nan)), float(row.get("ci_upper", np.nan))


def plot_metric_grid(
    layerwise: pd.DataFrame,
    *,
    models: list[str],
    metrics: list[str],
    title: str | None,
    output_path: Path,
    nrows: int,
    ncols: int,
    baseline_layerwise: pd.DataFrame | None = None,
    baseline_summary: pd.DataFrame | None = None,
    include_baselines: bool = False,
    compact: bool = False,
) -> None:
    """Plot a metric grid; ``compact`` draws it at main-text print size."""
    if compact:
        rc = MAIN_FIGURE_RC
        figsize = MAIN_FIGURE_SIZE_IN
    else:
        rc = {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
        figsize = (6.4 * ncols, 4.15 * nrows)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=compact)
        _plot_metric_grid_contents(
            fig,
            axes,
            layerwise=layerwise,
            models=models,
            metrics=metrics,
            title=title,
            output_path=output_path,
            baseline_layerwise=baseline_layerwise,
            baseline_summary=baseline_summary,
            include_baselines=include_baselines,
            compact=compact,
        )


def _plot_metric_grid_contents(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    layerwise: pd.DataFrame,
    models: list[str],
    metrics: list[str],
    title: str | None,
    output_path: Path,
    baseline_layerwise: pd.DataFrame | None,
    baseline_summary: pd.DataFrame | None,
    include_baselines: bool,
    compact: bool = False,
) -> None:
    axes_flat = np.ravel(axes)
    ncols = axes.shape[1] if np.ndim(axes) == 2 else len(axes_flat)
    colors = plt.get_cmap("tab10").colors
    linewidth, markersize = (1.1, 2.6) if compact else (1.7, 4.2)
    for panel_idx, (ax, experiment_id) in enumerate(zip(axes_flat, metrics)):
        subset = layerwise[
            (layerwise["experiment_id"] == experiment_id)
            & (layerwise["model_name"].isin(models))
        ]
        for model_idx, model_name in enumerate(models):
            curve = subset[subset["model_name"] == model_name].sort_values("layer_index")
            if curve.empty:
                continue
            color = colors[model_idx % len(colors)]
            label = MODEL_LABELS.get(model_name, model_name).replace("->", "\u2192")
            ax.plot(
                curve["relative_depth"],
                curve["raw_score"],
                color=color,
                marker=MARKERS[model_idx % len(MARKERS)],
                linestyle=LINE_STYLES[model_idx % len(LINE_STYLES)],
                linewidth=linewidth,
                markersize=markersize,
                label=label,
            )
            if curve["ci_lower"].notna().any() and curve["ci_upper"].notna().any():
                ax.fill_between(
                    curve["relative_depth"],
                    curve["ci_lower"],
                    curve["ci_upper"],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        if include_baselines:
            add_metric_references(
                ax,
                experiment_id=experiment_id,
                baseline_layerwise=baseline_layerwise,
                baseline_summary=baseline_summary,
                compact=compact,
            )
        panel_label = chr(ord("a") + panel_idx)
        ax.set_title(
            f"({panel_label}) {METRIC_LABELS.get(experiment_id, experiment_id)}",
            pad=3 if compact else 6,
        )
        # In compact grids the axes are shared, so only the outer panels are labelled.
        is_bottom_row = panel_idx >= len(metrics) - ncols
        if not compact or is_bottom_row:
            ax.set_xlabel("Relative depth")
        if not compact or panel_idx % ncols == 0:
            ax.set_ylabel("Score")
        ax.grid(alpha=0.25, linewidth=0.4 if compact else 0.5)
        if compact:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.set_xlim(-0.02, 1.02)
    for ax in axes_flat[len(metrics):]:
        ax.axis("off")
    handles, labels = collect_unique_legend_items(axes_flat)
    legend_ncol = min(len(labels), 4 if compact else 5)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_ncol,
            frameon=False,
            columnspacing=0.9 if compact else 2.0,
            handlelength=2.4 if compact else 2.0,
        )
    if title:
        fig.suptitle(title, y=0.995, fontsize=15)
    if compact:
        legend_rows = math.ceil(len(labels) / legend_ncol) if handles else 0
        fig.tight_layout(rect=(0, 0.035 + 0.045 * legend_rows, 1, 1), h_pad=0.6, w_pad=1.0)
    else:
        fig.tight_layout(rect=(0, 0.045, 1, 0.98 if title else 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def collect_unique_legend_items(
    axes: np.ndarray,
) -> tuple[list[Any], list[str]]:
    handles_by_label: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label.startswith("_"):
                continue
            handles_by_label.setdefault(label, handle)
    return list(handles_by_label.values()), list(handles_by_label.keys())


def add_metric_references(
    ax: plt.Axes,
    *,
    experiment_id: str,
    baseline_layerwise: pd.DataFrame | None,
    baseline_summary: pd.DataFrame | None,
    compact: bool = False,
) -> None:
    if baseline_layerwise is not None and not baseline_layerwise.empty:
        draw_random_vit_curve(ax, experiment_id, baseline_layerwise, compact=compact)
        if experiment_id == "saliency-auc-judd":
            draw_spatial_saliency_lines(ax, baseline_layerwise)
    if baseline_summary is not None and not baseline_summary.empty:
        draw_null_reference(ax, experiment_id, baseline_summary)
    chance = chance_for_experiment(experiment_id)
    if chance is not None:
        ax.axhline(
            chance,
            color="#6f6f6f",
            linestyle=(0, (1, 2)),
            linewidth=1.4,
            alpha=0.8,
            label="Chance",
        )


def draw_random_vit_curve(
    ax: plt.Axes,
    experiment_id: str,
    baseline_layerwise: pd.DataFrame,
    *,
    compact: bool = False,
) -> None:
    curve = baseline_layerwise[
        (baseline_layerwise["experiment_id"] == experiment_id)
        & (baseline_layerwise["model_name"] == "timm-random::vit_base_patch16_224")
    ].sort_values("layer_index")
    if curve.empty:
        return
    ax.plot(
        curve["relative_depth"],
        curve["raw_score"],
        color="#111111",
        marker="x",
        linestyle=(0, (5, 2)),
        linewidth=1.2 if compact else 1.8,
        markersize=3.0 if compact else 4.8,
        label="Random ViT",
        zorder=4,
    )


def draw_spatial_saliency_lines(
    ax: plt.Axes,
    baseline_layerwise: pd.DataFrame,
) -> None:
    styles = {
        "saliency-baseline::center-gaussian": ("Center Gaussian", "#8c2d04", "-."),
        "saliency-baseline::uniform": ("Uniform map", "#7b5aa6", ":"),
    }
    for model_name, (label, color, linestyle) in styles.items():
        curve = baseline_layerwise[
            (baseline_layerwise["experiment_id"] == "saliency-auc-judd")
            & (baseline_layerwise["model_name"] == model_name)
        ]
        if curve.empty:
            continue
        ax.axhline(
            float(curve["raw_score"].iloc[0]),
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.85,
            label=label,
        )


def draw_null_reference(
    ax: plt.Axes,
    experiment_id: str,
    baseline_summary: pd.DataFrame,
) -> None:
    rows = baseline_summary[
        (baseline_summary["experiment_id"] == experiment_id)
        & baseline_summary["null_score"].notna()
    ]
    if rows.empty:
        return
    row = rows.iloc[0]
    null_score = float(row["null_score"])
    label = null_label(str(row["metric_name"]))
    data_low, data_high = ax.dataLim.y0, ax.dataLim.y1
    if null_score < data_low - NULL_OFF_SCALE_GAP * (data_high - data_low):
        # Invisible handle: the legend entry carries the value of the off-scale null.
        ax.plot([], [], linestyle="none", label=f"{label.replace('permutation', 'perm.')} {null_score:.3f} (off-scale)")
        return
    ax.axhline(
        null_score,
        color="#525252",
        linestyle=(0, (3, 2)),
        linewidth=1.4,
        alpha=0.85,
        label=label,
    )


def build_baseline_table(
    *,
    baseline_results: Path | None,
    baseline_bootstrap_dir: Path | None,
    saliency_results: Path | None,
    saliency_bootstrap_dir: Path | None,
) -> pd.DataFrame | None:
    frames = []
    if baseline_results and baseline_results.exists():
        random_layerwise = load_layerwise_results(baseline_results)
        random_layer_bootstrap = load_layer_bootstrap(baseline_bootstrap_dir)
        random_best_bootstrap = load_best_bootstrap(baseline_bootstrap_dir)
        random_null_bootstrap = load_null_best_bootstrap(baseline_bootstrap_dir)
        random_layerwise = attach_layer_ci(random_layerwise, random_layer_bootstrap)
        random_summary = summarize_scores(random_layerwise, random_best_bootstrap)
        frames.append(attach_null_summary(random_summary, random_null_bootstrap))
    if saliency_results and saliency_results.exists():
        sal_layerwise = load_layerwise_results(saliency_results)
        sal_layer_bootstrap = load_layer_bootstrap(saliency_bootstrap_dir)
        sal_best_bootstrap = load_best_bootstrap(saliency_bootstrap_dir)
        sal_layerwise = attach_layer_ci(sal_layerwise, sal_layer_bootstrap)
        frames.append(summarize_scores(sal_layerwise, sal_best_bootstrap))
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    table = table[table["metric_name"].isin(BASELINE_METRIC_NAMES)].copy()
    table["reference"] = table["metric_name"].map(reference_for_metric)
    return table.sort_values(["metric_label", "model_name"])


def load_baseline_layerwise(
    *,
    baseline_results: Path | None,
    baseline_bootstrap_dir: Path | None,
    saliency_results: Path | None,
    saliency_bootstrap_dir: Path | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if baseline_results and baseline_results.exists():
        random_layerwise = load_layerwise_results(baseline_results)
        random_layer_bootstrap = load_layer_bootstrap(baseline_bootstrap_dir)
        frames.append(attach_layer_ci(random_layerwise, random_layer_bootstrap))
    if saliency_results and saliency_results.exists():
        sal_layerwise = load_layerwise_results(saliency_results)
        sal_layer_bootstrap = load_layer_bootstrap(saliency_bootstrap_dir)
        frames.append(attach_layer_ci(sal_layerwise, sal_layer_bootstrap))
    if not frames:
        return pd.DataFrame()
    layerwise = pd.concat(frames, ignore_index=True)
    return layerwise[layerwise["metric_name"].isin(BASELINE_METRIC_NAMES)].copy()


def build_baseline_context_table(
    selected_scores: pd.DataFrame,
    baseline_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment_id in CENTRAL_METRICS:
        trained = selected_scores[selected_scores["experiment_id"] == experiment_id]
        if trained.empty:
            continue
        rows.append(
            {
                "experiment_id": experiment_id,
                "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
                "entry": "Trained ViTs",
                "kind": "range",
                "score": float(trained["best_score"].median()),
                "score_min": float(trained["best_score"].min()),
                "score_max": float(trained["best_score"].max()),
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }
        )
        baseline_rows = baseline_table[baseline_table["experiment_id"] == experiment_id]
        for _, baseline in baseline_rows.iterrows():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
                    "entry": short_baseline_label(str(baseline["model_name"])),
                    "kind": "point",
                    "score": float(baseline["best_score"]),
                    "score_min": np.nan,
                    "score_max": np.nan,
                    "ci_lower": float(baseline["best_ci_lower"]),
                    "ci_upper": float(baseline["best_ci_upper"]),
                }
            )
            if not pd.isna(baseline.get("null_score", np.nan)):
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
                        "entry": null_label(str(baseline["metric_name"])),
                        "kind": "null",
                        "score": float(baseline["null_score"]),
                        "score_min": np.nan,
                        "score_max": np.nan,
                        "ci_lower": float(baseline["null_ci_lower"]),
                        "ci_upper": float(baseline["null_ci_upper"]),
                    }
                )
        chance = chance_for_experiment(experiment_id)
        if chance is not None:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
                    "entry": f"Chance ({chance:.3g})",
                    "kind": "chance",
                    "score": chance,
                    "score_min": np.nan,
                    "score_max": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_baseline_context(context: pd.DataFrame, *, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes_flat = np.ravel(axes)
    colors = {
        "range": "#4c78a8",
        "point": "#f58518",
        "null": "#6f6f6f",
        "chance": "#9a9a9a",
    }
    for panel_idx, (ax, experiment_id) in enumerate(zip(axes_flat, CENTRAL_METRICS)):
        panel = context[context["experiment_id"] == experiment_id].copy()
        if panel.empty:
            ax.axis("off")
            continue
        panel = panel.reset_index(drop=True)
        y_positions = np.arange(len(panel))[::-1]
        for y, (_, row) in zip(y_positions, panel.iterrows()):
            kind = row["kind"]
            color = colors.get(kind, "black")
            if kind == "range":
                ax.hlines(
                    y,
                    row["score_min"],
                    row["score_max"],
                    color=color,
                    linewidth=5,
                    alpha=0.55,
                )
                ax.plot(row["score"], y, marker="o", color=color, markersize=7)
            elif kind == "chance":
                ax.axvline(row["score"], color=color, linestyle=":", linewidth=1.6)
                ax.plot(row["score"], y, marker="|", color=color, markersize=11)
            else:
                interval_low = min(row["score"], row["ci_lower"], row["ci_upper"])
                interval_high = max(row["score"], row["ci_lower"], row["ci_upper"])
                lower = row["score"] - interval_low
                upper = interval_high - row["score"]
                if pd.isna(lower) or pd.isna(upper):
                    ax.plot(row["score"], y, marker="o", color=color, markersize=7)
                else:
                    ax.errorbar(
                        row["score"],
                        y,
                        xerr=np.array([[lower], [upper]]),
                        fmt="o",
                        color=color,
                        capsize=3,
                        linewidth=1.3,
                        markersize=6,
                    )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(panel["entry"])
        ax.set_xlabel("Best-layer score")
        panel_label = chr(ord("a") + panel_idx)
        ax.set_title(f"({panel_label}) {METRIC_LABELS.get(experiment_id, experiment_id)}")
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.margins(x=0.08, y=0.18)
    fig.suptitle("Baseline context for the four central metrics", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def short_baseline_label(model_name: str) -> str:
    if model_name == "timm-random::vit_base_patch16_224":
        return "Random ViT"
    if model_name == "saliency-baseline::uniform":
        return "Uniform"
    if model_name == "saliency-baseline::center-gaussian":
        return "Center Gaussian"
    return MODEL_LABELS.get(model_name, model_name)


def null_label(metric_name: str) -> str:
    if metric_name == "tid_spearman_mos":
        return "MOS permutation"
    if metric_name in ACCURACY_METRIC_NAMES:
        return "Label permutation"
    return "Null"


def chance_for_experiment(experiment_id: str) -> float | None:
    if experiment_id == "nights-preference-accuracy":
        return 0.5
    if experiment_id in {
        "things-triplet-accuracy",
        "levels-between-class",
        "levels-class-border",
        "levels-within-class",
    }:
        return 1 / 3
    return None


def write_optional_summary(
    *,
    results_path: Path | None,
    bootstrap_dir: Path | None,
    output_path: Path,
) -> None:
    if results_path is None or not results_path.exists():
        return
    layerwise = load_layerwise_results(results_path)
    layer_bootstrap = load_layer_bootstrap(bootstrap_dir)
    best_bootstrap = load_best_bootstrap(bootstrap_dir)
    layerwise = attach_layer_ci(layerwise, layer_bootstrap)
    summarize_scores(layerwise, best_bootstrap).to_csv(output_path, index=False)


def write_framework_run_summary(
    *,
    case_results: Path,
    baseline_results: Path | None,
    saliency_baseline_results: Path | None,
    rollout_results: Path | None,
    pooling_results: Path | None,
    sidecar_archives_dir: Path,
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    result_specs = [
        ("case_study", case_results),
        ("random_vit_baselines", baseline_results),
        ("saliency_spatial_baselines", saliency_baseline_results),
        ("rollout_sensitivity", rollout_results),
        ("pooling_ablation", pooling_results),
    ]
    for label, path in result_specs:
        if path is None or not path.exists():
            continue
        frame = pd.read_csv(path)
        done = frame[frame["status"] == "done"].copy()
        duration = pd.to_numeric(done.get("duration_seconds"), errors="coerce")
        rows.append(
            {
                "artifact_component": label,
                "result_csv": str(path),
                "result_rows": int(len(frame)),
                "done_rows": int(len(done)),
                "unique_models": int(done["model_name"].nunique()) if not done.empty else 0,
                "unique_experiments": (
                    int(done["experiment_id"].nunique()) if not done.empty else 0
                ),
                "unique_metric_implementations": (
                    int(done["metric_name"].nunique()) if not done.empty else 0
                ),
                "duration_seconds_median": _nan_percentile(duration, 50),
                "duration_seconds_iqr_low": _nan_percentile(duration, 25),
                "duration_seconds_iqr_high": _nan_percentile(duration, 75),
                "duration_seconds_total": float(duration.sum(skipna=True)),
            }
        )
    if sidecar_archives_dir.exists():
        archives = sorted(sidecar_archives_dir.glob("*.tar.gz"))
        rows.append(
            {
                "artifact_component": "unit_sidecar_archives",
                "result_csv": str(sidecar_archives_dir),
                "result_rows": int(len(archives)),
                "done_rows": int(len(archives)),
                "unique_models": "",
                "unique_experiments": "",
                "unique_metric_implementations": "",
                "duration_seconds_median": "",
                "duration_seconds_iqr_low": "",
                "duration_seconds_iqr_high": "",
                "duration_seconds_total": "",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_artifact_verification_table(*, output_path: Path) -> None:
    package_summary = _review_package_summary()
    rows = [
        {
            "artifact_item": "Anonymized review package",
            "location": "Online Resource 1",
            "verification": package_summary
            or "Packaged with MANIFEST.md; SHA256 generated at packaging time",
        },
        {
            "artifact_item": "Smoke command",
            "location": "review_artifact/smoke_json_runner.py",
            "verification": "First pass runs two metrics; resume pass runs zero metrics",
        },
        {
            "artifact_item": "Expected smoke output",
            "location": "review_artifact/expected_results.csv",
            "verification": "Stable results.csv columns compared after volatile timing columns are removed",
        },
        {
            "artifact_item": "Unit sidecar archives",
            "location": "review_artifact/unit_result_archives/",
            "verification": "Compressed sidecars for case study, baselines and sensitivity checks",
        },
        {
            "artifact_item": "Bootstrap/null summaries",
            "location": "outputs/bootstrap-framework-*/",
            "verification": "Generated with 2000 resamples and seed 20260611",
        },
        {
            "artifact_item": "Figure/table regeneration",
            "location": "scripts/framework_paper_outputs.py",
            "verification": "Regenerates main figures, supplementary grids, baseline context and contrast deltas",
        },
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _review_package_summary() -> str:
    path = Path("outputs/review_artifact/review_artifact_package.csv")
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    if frame.empty:
        return ""
    row = frame.iloc[0]
    return (
        f"{row['archive']}; SHA256={row['sha256']}; "
        f"{int(row['file_count'])} files; {int(row['size_bytes'])} bytes"
    )


def build_contrast_delta_table(
    sidecar_dir: Path,
    *,
    n_bootstraps: int = 2000,
    seed: int = 20260611,
) -> pd.DataFrame:
    """Return pre-specified paired bootstrap deltas for central paper contrasts."""
    sidecar_index = _index_sidecars(sidecar_dir)
    rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for experiment_id in CENTRAL_METRICS:
            reference_path = sidecar_index.get((contrast["reference"], experiment_id))
            comparison_path = sidecar_index.get((contrast["comparison"], experiment_id))
            if reference_path is None or comparison_path is None:
                rows.append(_missing_delta_row(contrast, experiment_id, seed, n_bootstraps))
                continue
            reference = pd.read_csv(reference_path)
            comparison = pd.read_csv(comparison_path)
            rows.append(
                summarize_contrast_delta(
                    reference,
                    comparison,
                    factor=contrast["factor"],
                    contrast=contrast["contrast"],
                    experiment_id=experiment_id,
                    reference_model=contrast["reference"],
                    comparison_model=contrast["comparison"],
                    n_bootstraps=n_bootstraps,
                    seed=seed,
                )
            )
    return pd.DataFrame(rows, columns=DELTA_TABLE_COLUMNS)


def summarize_contrast_delta(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    factor: str,
    contrast: str,
    experiment_id: str,
    reference_model: str,
    comparison_model: str,
    n_bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    reference_data, comparison_data = _aligned_metric_data(reference, comparison)
    rng = _contrast_rng(seed, factor, contrast, experiment_id)
    final_deltas = np.empty(n_bootstraps, dtype=float)
    best_deltas = np.empty(n_bootstraps, dtype=float)
    n_units = len(reference_data["unit_index"])
    for bootstrap_idx in range(n_bootstraps):
        sample = rng.integers(0, n_units, size=n_units)
        final_deltas[bootstrap_idx] = (
            _aggregate_layer(comparison_data, comparison_data["final_pos"], sample)
            - _aggregate_layer(reference_data, reference_data["final_pos"], sample)
        )
        best_deltas[bootstrap_idx] = _best_layer_score(
            comparison_data,
            sample,
        ) - _best_layer_score(reference_data, sample)

    all_units = np.arange(n_units)
    final_delta = (
        _aggregate_layer(comparison_data, comparison_data["final_pos"], all_units)
        - _aggregate_layer(reference_data, reference_data["final_pos"], all_units)
    )
    best_delta = _best_layer_score(comparison_data, all_units) - _best_layer_score(
        reference_data,
        all_units,
    )
    final_ci = _nan_ci(final_deltas)
    best_ci = _nan_ci(best_deltas)
    return {
        "factor": factor,
        "contrast": contrast,
        "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
        "reference_model": MODEL_LABELS.get(reference_model, reference_model),
        "comparison_model": MODEL_LABELS.get(comparison_model, comparison_model),
        "final_delta": final_delta,
        "final_ci_lower": final_ci[0],
        "final_ci_upper": final_ci[1],
        "final_support": _support_label(*final_ci),
        "best_delta": best_delta,
        "best_ci_lower": best_ci[0],
        "best_ci_upper": best_ci[1],
        "best_support": _support_label(*best_ci),
        "n_units": n_units,
        "reference_n_layers": len(reference_data["layer_index"]),
        "comparison_n_layers": len(comparison_data["layer_index"]),
        "seed": seed,
        "n_bootstraps": n_bootstraps,
    }


def _index_sidecars(sidecar_dir: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in sorted(sidecar_dir.glob("*.csv")):
        header = pd.read_csv(path, nrows=1)
        if header.empty:
            continue
        row = header.iloc[0]
        index[(str(row["model_name"]), normalize_experiment_id(str(row["experiment_id"])))] = path
    return index


def _missing_delta_row(
    contrast: dict[str, str],
    experiment_id: str,
    seed: int,
    n_bootstraps: int,
) -> dict[str, Any]:
    return {
        "factor": contrast["factor"],
        "contrast": contrast["contrast"],
        "metric_label": METRIC_LABELS.get(experiment_id, experiment_id),
        "reference_model": MODEL_LABELS.get(contrast["reference"], contrast["reference"]),
        "comparison_model": MODEL_LABELS.get(contrast["comparison"], contrast["comparison"]),
        "final_delta": np.nan,
        "final_ci_lower": np.nan,
        "final_ci_upper": np.nan,
        "final_support": "missing",
        "best_delta": np.nan,
        "best_ci_lower": np.nan,
        "best_ci_upper": np.nan,
        "best_support": "missing",
        "n_units": np.nan,
        "reference_n_layers": np.nan,
        "comparison_n_layers": np.nan,
        "seed": seed,
        "n_bootstraps": n_bootstraps,
    }


def _aligned_metric_data(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common_units = sorted(set(reference["unit_index"]) & set(comparison["unit_index"]))
    if not common_units:
        raise ValueError("Contrast sidecars do not share unit_index values")
    return (
        _metric_data(reference, common_units),
        _metric_data(comparison, common_units),
    )


def _metric_data(frame: pd.DataFrame, unit_index: list[Any]) -> dict[str, Any]:
    metric_name = str(frame["metric_name"].iloc[0])
    mode = _contrast_metric_mode(metric_name)
    layers = sorted(frame["layer_index"].unique(), key=float)
    values: list[np.ndarray] = []
    human_values: list[np.ndarray] = []
    for layer in layers:
        layer_frame = frame[frame["layer_index"] == layer].set_index("unit_index")
        if mode in {"mean", "accuracy"}:
            column = "is_correct" if mode == "accuracy" else "unit_value"
            values.append(pd.to_numeric(layer_frame.loc[unit_index, column]).to_numpy(float))
            human_values.append(np.array([], dtype=float))
        else:
            values.append(pd.to_numeric(layer_frame.loc[unit_index, "unit_value"]).to_numpy(float))
            human_values.append(
                pd.to_numeric(layer_frame.loc[unit_index, "human_value"]).to_numpy(float)
            )
    values_array = np.vstack(values)
    human_array = np.vstack(human_values) if mode == "tid_spearman" else None
    rank_values = np.vstack([rankdata(row, method="average") for row in values_array])
    rank_human = (
        np.vstack([rankdata(row, method="average") for row in human_array])
        if human_array is not None
        else None
    )
    return {
        "metric_name": metric_name,
        "mode": mode,
        "unit_index": unit_index,
        "layer_index": layers,
        "values": values_array,
        "rank_values": rank_values,
        "rank_human": rank_human,
        "final_pos": len(layers) - 1,
    }


def _contrast_metric_mode(metric_name: str) -> str:
    if metric_name in ACCURACY_METRIC_NAMES:
        return "accuracy"
    if metric_name == "tid_spearman_mos":
        return "tid_spearman"
    if metric_name.startswith("saliency_"):
        return "mean"
    raise ValueError(f"Unsupported contrast metric: {metric_name}")


def _aggregate_layer(data: dict[str, Any], layer_pos: int, sample: np.ndarray) -> float:
    if data["mode"] == "tid_spearman":
        return _fast_spearman(
            data["rank_values"][layer_pos, sample],
            data["rank_human"][layer_pos, sample],
        )
    return float(np.nanmean(data["values"][layer_pos, sample]))


def _best_layer_score(data: dict[str, Any], sample: np.ndarray) -> float:
    scores = np.array(
        [
            _aggregate_layer(data, layer_pos, sample)
            for layer_pos in range(len(data["layer_index"]))
        ],
        dtype=float,
    )
    if np.isfinite(scores).any():
        return float(np.nanmax(scores))
    return float("nan")


def _fast_spearman(rank_x: np.ndarray, rank_y: np.ndarray) -> float:
    finite = np.isfinite(rank_x) & np.isfinite(rank_y)
    if finite.sum() < 2:
        return float("nan")
    x = rank_x[finite]
    y = rank_y[finite]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _contrast_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256("::".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**32))


def _nan_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


def _support_label(ci_lower: float, ci_upper: float) -> str:
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return "missing"
    if ci_lower > 0:
        return "directional+"
    if ci_upper < 0:
        return "directional-"
    return "ambiguous"


def _nan_percentile(values: pd.Series, percentile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return float("nan")
    return float(np.percentile(finite, percentile))


def attach_null_summary(summary: pd.DataFrame, null_best: pd.DataFrame) -> pd.DataFrame:
    if null_best.empty:
        summary["null_score"] = np.nan
        summary["null_ci_lower"] = np.nan
        summary["null_ci_upper"] = np.nan
        summary["null_method"] = ""
        summary["chance"] = np.nan
        return summary
    null_cols = [
        "model_name",
        "experiment_id",
        "metric_name",
        "best_score",
        "ci_lower",
        "ci_upper",
        "null_method",
        "chance",
    ]
    merged = summary.merge(
        null_best[null_cols].rename(
            columns={
                "best_score": "null_score",
                "ci_lower": "null_ci_lower",
                "ci_upper": "null_ci_upper",
            }
        ),
        on=["model_name", "experiment_id", "metric_name"],
        how="left",
    )
    return merged


def reference_for_metric(metric_name: str) -> str:
    if metric_name == "nights_preference_accuracy":
        return "chance=0.5; label permutation when sidecar labels exist"
    if metric_name in {"things_triplet_accuracy", "levels_triplet_accuracy"}:
        return "chance=1/3; label permutation when sidecar labels exist"
    if metric_name == "tid_spearman_mos":
        return "MOS permutation"
    if metric_name.startswith("saliency_"):
        return "uniform/center baseline and random ViT"
    return ""


if __name__ == "__main__":
    main()
