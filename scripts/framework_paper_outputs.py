#!/usr/bin/env python3
"""Generate framework-paper tables and figures from result CSVs and sidecars."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

SUPPLEMENT_METRICS = [
    "saliency-auc-judd",
    "tid-spearman-mos",
    "nights-preference-accuracy",
    "things-triplet-accuracy",
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

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "P", "X"]


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

    for group_name, models in MODEL_GROUPS.items():
        plot_metric_grid(
            case_layerwise,
            models=models,
            metrics=CENTRAL_METRICS,
            title=GROUP_TITLES[group_name],
            output_path=figures_dir / f"framework_main_{group_name}_2x2.png",
            nrows=2,
            ncols=2,
        )
        plot_metric_grid(
            case_layerwise,
            models=models,
            metrics=SUPPLEMENT_METRICS,
            title=f"{GROUP_TITLES[group_name]}: selected metric battery",
            output_path=figures_dir / f"framework_supplement_{group_name}_4x2.png",
            nrows=4,
            ncols=2,
        )

    baseline_table = build_baseline_table(
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
    title: str,
    output_path: Path,
    nrows: int,
    ncols: int,
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.2 * nrows))
    axes_flat = np.ravel(axes)
    colors = plt.get_cmap("tab10").colors
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
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(
                curve["relative_depth"],
                curve["raw_score"],
                color=color,
                marker=MARKERS[model_idx % len(MARKERS)],
                linestyle=LINE_STYLES[model_idx % len(LINE_STYLES)],
                linewidth=1.7,
                markersize=4.2,
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
        panel_label = chr(ord("a") + panel_idx)
        ax.set_title(f"({panel_label}) {METRIC_LABELS.get(experiment_id, experiment_id)}")
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.02, 1.02)
    for ax in axes_flat[len(metrics):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


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
