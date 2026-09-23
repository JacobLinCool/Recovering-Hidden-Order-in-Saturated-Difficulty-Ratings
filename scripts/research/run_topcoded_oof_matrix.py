#!/usr/bin/env python
"""Run or resume the frozen top-coded OOF CPU matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from TaikoChartEstimator.research.evidence import (
    file_sha256,
    research_source_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "paper/manuscript/artifact/public/global_primary_config.json"
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("paper/manuscript/artifact/public/EXPERIMENT_SPEC.md"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("experiments/icassp2027_core/data/global_features.jsonl"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("experiments/icassp2027_core/data/dataset_manifest.json"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/raw/runs"
        ),
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _matching_success_exists(
    root: Path,
    key: str,
    *,
    seed: int,
    config_hash: str,
    spec_hash: str,
    feature_hash: str,
    dataset_manifest_hash: str,
    source_hash: str,
) -> bool:
    for attempt in sorted((root / key / f"seed-{seed}").glob("attempt-*")):
        status_path = attempt / "status.json"
        manifest_path = attempt / "run_manifest.json"
        if not status_path.exists() or not manifest_path.exists():
            continue
        status = _json(status_path)
        manifest = _json(manifest_path)
        if status.get("status") != "success":
            continue
        if (
            manifest.get("status") != "success"
            or manifest.get("experiment_id")
            != "icassp2027_topcoded_oof_v1"
            or manifest.get("condition") != key
            or int(manifest.get("seed", -1)) != seed
        ):
            continue
        if not manifest.get("parameters", {}).get("primary_evidence", False):
            continue
        required_outputs = (
            "fold_assignments.jsonl",
            "validation_search.jsonl",
            "oof_predictions.jsonl",
            "fold_diagnostics.jsonl",
            "model_records.json",
            "aggregate_metrics.json",
        )
        if any(not (attempt / name).is_file() for name in required_outputs):
            continue
        expected_models = {
            attempt / "models" / f"fold-{fold}.joblib"
            for fold in range(5)
        }
        if any(not path.is_file() for path in expected_models):
            continue
        inputs = manifest.get("inputs", {})
        observed_source_hash = (
            manifest.get("environment", {})
            .get("research_source", {})
            .get("aggregate_sha256")
        )
        if (
            inputs.get("config_sha256") == config_hash
            and inputs.get("spec_sha256") == spec_hash
            and inputs.get("features_sha256") == feature_hash
            and inputs.get("dataset_manifest_sha256")
            == dataset_manifest_hash
            and observed_source_hash == source_hash
        ):
            return True
    return False


def main() -> None:
    args = parse_args()
    config = _json(args.config)
    if not config.get("primary_evidence", False):
        raise ValueError("matrix runner requires a primary-evidence config")
    config_hash = file_sha256(args.config)
    spec_hash = file_sha256(args.spec)
    feature_hash = file_sha256(args.features)
    dataset_manifest_hash = file_sha256(args.dataset_manifest)
    source_hash = research_source_snapshot()["aggregate_sha256"]
    seed = int(config["seed"])

    completed: list[str] = []
    skipped: list[str] = []
    for scenario in config["scenarios"]:
        for condition in config["conditions"]:
            key = f"{scenario}--{condition}"
            if _matching_success_exists(
                args.runs,
                key,
                seed=seed,
                config_hash=config_hash,
                spec_hash=spec_hash,
                feature_hash=feature_hash,
                dataset_manifest_hash=dataset_manifest_hash,
                source_hash=source_hash,
            ):
                skipped.append(key)
                continue
            command = [
                sys.executable,
                "scripts/research/run_topcoded_oof.py",
                "--scenario",
                scenario,
                "--condition",
                condition,
                "--config",
                str(args.config),
                "--spec",
                str(args.spec),
                "--features",
                str(args.features),
                "--dataset-manifest",
                str(args.dataset_manifest),
                "--output-root",
                str(args.runs),
            ]
            subprocess.run(command, check=True)
            completed.append(key)

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped_matching_successes": skipped,
                "expected_total": (
                    len(config["scenarios"]) * len(config["conditions"])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
