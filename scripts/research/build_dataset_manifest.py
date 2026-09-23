#!/usr/bin/env python
"""Build and audit the frozen ICASSP 2027 dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from TaikoChartEstimator.research.data import build_dataset_manifest, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/icassp2027_core/data/dataset_manifest.json"
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split-seed", type=int, default=2027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset_manifest(
        cache_dir=args.cache_dir,
        split_seed=args.split_seed,
    )
    save_manifest(manifest, args.output)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
