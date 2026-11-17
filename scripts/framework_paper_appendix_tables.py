#!/usr/bin/env python3
"""Write the LaTeX table rows of the paper appendix and Table 6 from the result CSVs.

Usage: uv run python scripts/framework_paper_appendix_tables.py \
    [--tables-dir outputs/framework_paper/tables] [--output-dir outputs/framework_paper/latex]
"""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--tables-dir", type=Path, default=Path("outputs/framework_paper/tables"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/framework_paper/latex"))
args = parser.parse_args()
tables = args.tables_dir
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("timm::vit_base_patch16_224.augreg_in1k", "AugReg IN1k"),
    ("timm::vit_base_patch16_224.mae", "MAE"),
    ("timm::vit_base_patch16_224.dino", "DINO"),
    ("timm::vit_base_patch16_clip_224.laion2b_ft_in1k", "CLIP"),
    ("timm::vit_base_patch16_224.augreg_in21k", "AugReg IN21k"),
    ("timm::vit_base_patch16_384.augreg_in1k", "AugReg 384"),
]
TESTS = [
    ("saliency-auc-judd", "Saliency AUC-Judd (MIT1003)"),
    ("tid-spearman-mos", "TID2013 Spearman MOS"),
    ("nights-preference-accuracy", "NIGHTS preference accuracy"),
    ("levels-between-class", "LEVELS between-class accuracy"),
    ("levels-class-border", "LEVELS class-border accuracy"),
    ("levels-within-class", "LEVELS within-class accuracy"),
    ("psychophysics-weber-law-pearson", "Weber law (visturing Property~2), Pearson"),
]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def signed(x):
    return "--" if pd.isna(x) else f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")


def with_ci(v, lo, hi, fmt=f3):
    if pd.isna(lo) or pd.isna(hi):
        return fmt(v)
    return f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"


# Full benchmark results.
scores = pd.read_csv(tables / "framework_selected_model_scores_with_ci.csv")
lines = []
for exp, title in TESTS:
    lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for model, label in MODELS:
        r = scores[(scores.model_name == model) & (scores.experiment_id == exp)]
        assert len(r) == 1, (model, exp)
        r = r.iloc[0]
        lines.append(
            f"\\quad {label} & {with_ci(r.best_score, r.best_ci_lower, r.best_ci_upper)} & "
            f"{r.best_relative_depth:.2f} & "
            f"{with_ci(r.final_score, r.final_ci_lower, r.final_ci_upper)} \\\\"
        )
(out / "full_results_rows.tex").write_text("\n".join(lines) + "\n")

# Pre-specified contrasts.
deltas = pd.read_csv(tables / "framework_prespecified_contrast_deltas.csv")
CONTRAST_LABELS = {
    "MAE - AugReg IN1k": "MAE $-$ AugReg IN1k",
    "DINO - AugReg IN1k": "DINO $-$ AugReg IN1k",
    "CLIP - AugReg IN1k": "CLIP $-$ AugReg IN1k",
    "AugReg IN21k - AugReg IN1k": "AugReg IN21k $-$ AugReg IN1k",
    "AugReg 384px - AugReg 224px": "AugReg 384 $-$ AugReg IN1k",
}
METRIC_SHORT = {
    "Saliency AUC-Judd": "Saliency AUC-Judd",
    "TID Spearman MOS": "TID Spearman MOS",
    "NIGHTS preference": "NIGHTS preference",
    "LEVELS between class": "LEVELS between-class",
}
lines = []
for contrast, clabel in CONTRAST_LABELS.items():
    sub = deltas[deltas.contrast == contrast]
    assert len(sub) == 4, contrast
    lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{clabel}}}}} \\\\")
    for _, r in sub.iterrows():
        lines.append(
            f"\\quad {METRIC_SHORT[r.metric_label]} & "
            f"{with_ci(r.final_delta, r.final_ci_lower, r.final_ci_upper, signed)} & "
            f"{with_ci(r.best_delta, r.best_ci_lower, r.best_ci_upper, signed)} \\\\"
        )
(out / "contrast_rows.tex").write_text("\n".join(lines) + "\n")

# Sensitivity checks.
rollout = pd.read_csv(tables / "framework_rollout_sensitivity.csv")
pooling = pd.read_csv(tables / "framework_pooling_ablation.csv")
GROUPS = [
    ("Saliency AUC-Judd: attention rollout settings", rollout, [
        ("saliency-auc-rollout-max-discard095", "max fusion, discard 0.95 (default)"),
        ("saliency-auc-rollout-max-discard000", "max fusion, no discard"),
        ("saliency-auc-rollout-mean-discard095", "mean fusion, discard 0.95"),
        ("saliency-auc-rollout-mean-discard000", "mean fusion, no discard"),
    ]),
    ("TID2013 Spearman MOS: representation pooling", pooling, [
        ("tid-spearman-mos-pooling-flatten", "flatten [CLS+patches] (default)"),
        ("tid-spearman-mos-pooling-cls", "CLS token only"),
        ("tid-spearman-mos-pooling-mean-patch", "mean over patches"),
    ]),
    ("THINGS triplet accuracy: representation pooling", pooling, [
        ("things-triplet-accuracy-pooling-flatten", "flatten [CLS+patches] (default)"),
        ("things-triplet-accuracy-pooling-cls", "CLS token only"),
        ("things-triplet-accuracy-pooling-mean-patch", "mean over patches"),
    ]),
]
lines = []
for title, frame, rows in GROUPS:
    lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for exp, label in rows:
        r = frame[frame.experiment_id == exp]
        assert len(r) == 1, exp
        r = r.iloc[0]
        lines.append(
            f"\\quad {label} & {with_ci(r.best_score, r.best_ci_lower, r.best_ci_upper)} & "
            f"{with_ci(r.final_score, r.final_ci_lower, r.final_ci_upper)} \\\\"
        )
(out / "sensitivity_rows.tex").write_text("\n".join(lines) + "\n")

# Baselines (Table 6 body).
base = pd.read_csv(tables / "framework_baselines_with_ci.csv")
BASE_ROWS = [
    ("timm-random::vit_base_patch16_224", "saliency-auc-judd", "AUC-Judd", "Random ViT"),
    ("saliency-baseline::uniform", "saliency-auc-judd", "AUC-Judd", "Uniform"),
    ("saliency-baseline::center-gaussian", "saliency-auc-judd", "AUC-Judd", "Center"),
    ("timm-random::vit_base_patch16_224", "tid-spearman-mos", "TID Spearman", "Random ViT"),
    ("timm-random::vit_base_patch16_224", "nights-preference-accuracy", "NIGHTS acc.", "Random ViT"),
    ("timm-random::vit_base_patch16_224", "levels-between-class", "LEVELS between", "Random ViT"),
    ("timm-random::vit_base_patch16_224", "levels-class-border", "LEVELS border", "Random ViT"),
    ("timm-random::vit_base_patch16_224", "levels-within-class", "LEVELS within", "Random ViT"),
]
lines = []
for model, exp, metric, control in BASE_ROWS:
    r = base[(base.model_name == model) & (base.experiment_id == exp)]
    assert len(r) == 1, (model, exp)
    r = r.iloc[0]
    null = "--" if pd.isna(r.null_score) else with_ci(r.null_score, r.null_ci_lower, r.null_ci_upper)
    lines.append(
        f"{metric} & {control} & {with_ci(r.best_score, r.best_ci_lower, r.best_ci_upper)} & "
        f"{with_ci(r.final_score, r.final_ci_lower, r.final_ci_upper)} & {null} \\\\"
    )
(out / "baseline_rows.tex").write_text("\n".join(lines) + "\n")
print(f"Wrote table rows to {out}")
