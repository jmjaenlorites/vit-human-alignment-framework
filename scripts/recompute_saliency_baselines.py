#!/usr/bin/env python3
"""Inspect or prepare MIT1003 saliency baseline recomputation metadata.

The default mode is intentionally non-computational: it only summarizes the
reference artifact. Use this as a guarded handoff point before implementing or
running framework-native baseline recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PATH = (
    REPO_ROOT / "data" / "baselines" / "saliency_mit1003_reference.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MIT1003 saliency baseline provenance without running "
            "heavy model or dataset computation."
        )
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Path to the saliency baseline provenance JSON.",
    )
    parser.add_argument(
        "--check-source-files",
        action="store_true",
        help="Check whether legacy source files still exist and match hashes.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any checked source file is missing or mismatched.",
    )
    parser.add_argument(
        "--emit-recompute-template",
        type=Path,
        help=(
            "Write a framework_recomputed JSON template. This does not compute "
            "metrics and will not contact remote servers."
        ),
    )
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_reference(reference: dict[str, Any]) -> None:
    dataset = reference["dataset"]
    print(f"Artifact: {reference['artifact_id']} ({reference['artifact_status']})")
    print(
        f"Dataset: {dataset['name']} "
        f"expected_image_count={dataset['expected_image_count']}"
    )
    print("Baselines:")
    for baseline in reference["baselines"]:
        values = baseline["values"]
        auc = values["saliency_auc_judd"]["value"]
        pearson = values["saliency_pearson_correlation_coefficient"]["value"]
        print(
            f"  - {baseline['legacy_name']}: "
            f"saliency_auc_judd={auc}, "
            f"saliency_pearson_correlation_coefficient={pearson}, "
            f"status={baseline['status']}, "
            f"framework_recomputed_status={baseline['framework_recomputed_status']}"
        )
    print(
        "Policy: use framework_recomputed values for paper tables when available; "
        "reference values are provenance/sanity-check inputs only."
    )


def check_source_files(reference: dict[str, Any], *, strict: bool) -> int:
    failures = 0
    for source in reference["provenance"]["legacy_source_files"]:
        path = Path(source["path"])
        if not path.exists():
            failures += 1
            print(f"MISSING {path}")
            continue
        actual_sha256 = sha256_file(path)
        actual_bytes = path.stat().st_size
        sha_ok = actual_sha256 == source["sha256"]
        bytes_ok = actual_bytes == source["bytes"]
        status = "OK" if sha_ok and bytes_ok else "MISMATCH"
        if status != "OK":
            failures += 1
        print(
            f"{status} {path} "
            f"sha256={actual_sha256} bytes={actual_bytes}"
        )
    return 1 if strict and failures else 0


def recompute_template(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": reference["schema_version"],
        "artifact_id": "saliency_mit1003_framework_recomputed",
        "created_at": None,
        "dataset": reference["dataset"],
        "status": "framework_recomputed",
        "notes": (
            "Fill this template only after recomputing values through this "
            "repository. Do not paste legacy reference values here."
        ),
        "baselines": [
            {
                "baseline_id": baseline["baseline_id"],
                "legacy_name": baseline["legacy_name"],
                "framework_recomputed_status": "pending",
                "values": {
                    "saliency_auc_judd": None,
                    "saliency_pearson_correlation_coefficient": None,
                },
            }
            for baseline in reference["baselines"]
        ],
    }


def write_template(reference: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(recompute_template(reference), handle, indent=2)
        handle.write("\n")
    print(f"Wrote non-computational recompute template: {path}")


def main() -> int:
    args = parse_args()
    reference = load_reference(args.reference_json)
    summarize_reference(reference)

    exit_code = 0
    if args.check_source_files:
        exit_code = check_source_files(reference, strict=args.strict)
    if args.emit_recompute_template:
        write_template(reference, args.emit_recompute_template)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
