#!/usr/bin/env python3
"""Create a clean anonymized review artifact archive."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import zipfile
from pathlib import Path

import pandas as pd


INCLUDE_PATHS = [
    "LICENSE",
    "README.md",
    "README_REVIEW.md",
    "pyproject.toml",
    "uv.lock",
    "pytest.ini",
    "main.py",
    "src",
    "tests/README.md",
    "tests/conftest.py",
    "tests/unit",
    "review_artifact",
    "data/experiments.example.json",
    "data/experiments.timm.model-battery.json",
    "data/experiments.framework-case-study.json",
    "data/experiments.framework-baselines.json",
    "data/experiments.framework-saliency-baselines.json",
    "data/experiments.framework-rollout-sensitivity.json",
    "data/experiments.framework-pooling-ablation.json",
    "data/experiments.saliency-baselines.json",
    "data/baselines/saliency_mit1003_reference.json",
    "docs/baselines_saliency.md",
    "scripts/recompute_saliency_baselines.py",
    "scripts/bootstrap_unit_results.py",
    "scripts/framework_paper_outputs.py",
    "scripts/framework_paper_appendix_tables.py",
    "scripts/package_review_artifact.py",
    "results-model-battery.csv",
    "results-saliency-baselines.csv",
    "results-framework-baselines-seed0.csv",
    "results-framework-baselines-seed0-v3.csv",
    "results-framework-case-study-v2.csv",
    "results-framework-rollout-sensitivity.csv",
    "results-framework-pooling-ablation.csv",
    "results-framework-saliency-baselines.csv",
    "outputs/bootstrap-framework-baselines",
    "outputs/bootstrap-framework-case-study",
    "outputs/bootstrap-framework-rollout-sensitivity",
    "outputs/bootstrap-framework-pooling-ablation",
    "outputs/bootstrap-framework-saliency-baselines",
    "outputs/framework_paper",
]

EXCLUDE_PATTERNS = [
    ".git/*",
    ".venv/*",
    ".pytest_cache/*",
    "*/__pycache__/*",
    "*.pyc",
    "*.pt",
    "*.npy",
    "*.tmp",
    ".DS_Store",
    "._*",
    "*/._*",
    "tests/data/*",
    "tests/fixtures/*",
    "unit-results-framework-*/*",
    "outputs/review_artifact/*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/review_artifact"),
    )
    parser.add_argument(
        "--archive-name",
        default="vit-human-alignment-review-artifact.zip",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / args.archive_name
    files = _collect_files(Path("."))
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.as_posix())
    sha256 = _sha256(archive_path)
    sha_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sha_path.write_text(f"{sha256}  {archive_path.name}\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "archive": archive_path.name,
                "sha256": sha256,
                "size_bytes": archive_path.stat().st_size,
                "file_count": len(files),
            }
        ]
    ).to_csv(args.output_dir / "review_artifact_package.csv", index=False)
    print(f"Wrote {archive_path}")
    print(f"SHA256 {sha256}")
    print(f"Included {len(files)} files")
    return 0


def _collect_files(root: Path) -> list[Path]:
    collected: set[Path] = set()
    for include in INCLUDE_PATHS:
        path = root / include
        if not path.exists():
            continue
        if path.is_file():
            if not _is_excluded(path):
                collected.add(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and not _is_excluded(child):
                collected.add(child)
    return sorted(collected, key=lambda item: item.as_posix())


def _is_excluded(path: Path) -> bool:
    text = path.as_posix()
    return any(fnmatch.fnmatch(text, pattern) for pattern in EXCLUDE_PATTERNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
