#!/usr/bin/env python
"""Build the frozen global symbolic feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from TaikoChartEstimator.research.data import load_manifest
from TaikoChartEstimator.research.features import (
    BASE_FEATURE_NAMES,
    COURSE_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    build_feature_records,
    save_feature_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/icassp2027_core/data/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027_core/data/global_features.jsonl"),
    )
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    records = build_feature_records(manifest, cache_dir=args.cache_dir)
    save_feature_records(records, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_table_sha256": digest,
        "records": len(records),
        "records_by_split": dict(
            sorted(Counter(record["primary_split"] for record in records).items())
        ),
        "base_features": list(BASE_FEATURE_NAMES),
        "course_features": list(COURSE_FEATURE_NAMES),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
