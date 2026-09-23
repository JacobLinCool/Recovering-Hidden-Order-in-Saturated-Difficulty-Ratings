#!/usr/bin/env python
"""Build strict, deterministic canonical tables for top-coded OOF evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    research_source_snapshot,
)
from TaikoChartEstimator.research.features import load_feature_records
from TaikoChartEstimator.research.oof import build_song_group_folds

EXPERIMENT_ID = "icassp2027_topcoded_oof_v1"
CANONICAL_SCHEMA_VERSION = "topcoded_oof_canonical_v1"
SCENARIOS = ("native", "cap7", "cap8", "cap9")
CONDITIONS = (
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
    "ordinal_logit",
    "ordinal_probit",
    "ordinal_logit_reduction",
)
SCENARIO_CAPS = {
    "native": 10.0,
    "cap7": 7.0,
    "cap8": 8.0,
    "cap9": 9.0,
}
EXPECTED_CHART_COUNT = 1_023
EXPECTED_SONG_GROUP_COUNT = 1_019
EXPECTED_OUTER_FOLDS = 5
EXPECTED_RUN_COUNT = len(SCENARIOS) * len(CONDITIONS)
EXPECTED_SCORE_COUNT = EXPECTED_RUN_COUNT * EXPECTED_CHART_COUNT
EXPECTED_FOLD_DIAGNOSTIC_COUNT = EXPECTED_RUN_COUNT * EXPECTED_OUTER_FOLDS

_RAW_OUTPUTS = {
    "fold_assignments": "fold_assignments.jsonl",
    "validation_search": "validation_search.jsonl",
    "oof_predictions": "oof_predictions.jsonl",
    "fold_diagnostics": "fold_diagnostics.jsonl",
    "model_records": "model_records.json",
    "aggregate_metrics": "aggregate_metrics.json",
    "model_directory": "models",
}
_COHORT_HASH_FIELDS = (
    "config_sha256",
    "spec_sha256",
    "features_sha256",
    "dataset_manifest_sha256",
)


class CanonicalEvidenceError(RuntimeError):
    """Raised when raw evidence cannot support a canonical primary table."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/raw/runs"
        ),
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
        "--output",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/canonical"
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalEvidenceError(
            f"cannot read JSON object {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CanonicalEvidenceError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CanonicalEvidenceError(
            f"cannot read JSONL table {path}: {error}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CanonicalEvidenceError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise CanonicalEvidenceError(
                f"{path}:{line_number} must contain one JSON object"
            )
        rows.append(value)
    return rows


def _require_exact_primary_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise CanonicalEvidenceError("primary config has the wrong experiment_id")
    if config.get("primary_evidence") is not True:
        raise CanonicalEvidenceError("canonical builder requires primary_evidence=true")
    if int(config.get("seed", -1)) != 2027:
        raise CanonicalEvidenceError("primary config seed must remain frozen at 2027")
    if int(config.get("outer_folds", -1)) != EXPECTED_OUTER_FOLDS:
        raise CanonicalEvidenceError("primary config must use five outer folds")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise CanonicalEvidenceError("primary config condition matrix drifted")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, dict) or tuple(scenarios) != SCENARIOS:
        raise CanonicalEvidenceError("primary config scenario matrix drifted")
    observed_caps = {
        scenario: float(parameters["cap"])
        for scenario, parameters in scenarios.items()
        if isinstance(parameters, dict) and "cap" in parameters
    }
    if observed_caps != SCENARIO_CAPS:
        raise CanonicalEvidenceError(
            f"primary config caps drifted: observed {observed_caps}"
        )


def _oni_feature_panel(
    features_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = sorted(
        (
            dict(row)
            for row in load_feature_records(features_path)
            if row["difficulty"] == "oni"
        ),
        key=lambda row: str(row["chart_id"]),
    )
    if len(rows) != EXPECTED_CHART_COUNT:
        raise CanonicalEvidenceError(
            f"expected {EXPECTED_CHART_COUNT} Oni charts, found {len(rows)}"
        )
    groups = {str(row["normalized_title"]) for row in rows}
    if len(groups) != EXPECTED_SONG_GROUP_COUNT:
        raise CanonicalEvidenceError(
            "expected "
            f"{EXPECTED_SONG_GROUP_COUNT} Oni title groups, found {len(groups)}"
        )
    by_chart = {str(row["chart_id"]): row for row in rows}
    if len(by_chart) != len(rows):
        raise CanonicalEvidenceError("Oni feature panel has duplicate chart IDs")
    return rows, by_chart


def _successful_primary_attempts(
    condition_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    successes: list[tuple[Path, dict[str, Any]]] = []
    if not condition_root.exists():
        return successes
    for attempt in sorted(condition_root.glob("seed-*/attempt-*")):
        if not attempt.is_dir():
            continue
        status_path = attempt / "status.json"
        manifest_path = attempt / "run_manifest.json"
        if not status_path.is_file() or not manifest_path.is_file():
            continue
        status = _read_json(status_path)
        manifest = _read_json(manifest_path)
        if status.get("status") != "success":
            continue
        if manifest.get("status") != "success":
            continue
        if manifest.get("parameters", {}).get("primary_evidence") is not True:
            continue
        successes.append((attempt, manifest))
    return successes


def _validate_selected_manifest(
    *,
    attempt: Path,
    manifest: Mapping[str, Any],
    scenario: str,
    condition: str,
    seed: int,
    expected_input_hashes: Mapping[str, str],
    expected_source_hash: str,
) -> dict[str, Any]:
    condition_key = f"{scenario}--{condition}"
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise CanonicalEvidenceError(
            f"{attempt} has the wrong experiment_id"
        )
    if manifest.get("condition") != condition_key:
        raise CanonicalEvidenceError(
            f"{attempt} has condition {manifest.get('condition')!r}, "
            f"expected {condition_key!r}"
        )
    if int(manifest.get("seed", -1)) != seed:
        raise CanonicalEvidenceError(f"{attempt} has the wrong seed")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise CanonicalEvidenceError(f"{attempt} has no parameters object")
    if parameters.get("primary_evidence") is not True:
        raise CanonicalEvidenceError(f"{attempt} is not primary evidence")
    if parameters.get("scenario") != scenario:
        raise CanonicalEvidenceError(f"{attempt} has scenario parameter drift")
    if parameters.get("condition") != condition:
        raise CanonicalEvidenceError(f"{attempt} has condition parameter drift")
    if not math.isclose(
        float(parameters.get("active_cap", float("nan"))),
        SCENARIO_CAPS[scenario],
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise CanonicalEvidenceError(f"{attempt} has active-cap drift")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise CanonicalEvidenceError(f"{attempt} has no inputs object")
    for field, expected in expected_input_hashes.items():
        if inputs.get(field) != expected:
            raise CanonicalEvidenceError(
                f"{attempt} has mismatched {field}: "
                f"{inputs.get(field)!r} != {expected!r}"
            )
    for field in _COHORT_HASH_FIELDS:
        value = inputs.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise CanonicalEvidenceError(
                f"{attempt} has invalid or missing {field}"
            )

    research_source = (
        manifest.get("environment", {}).get("research_source", {})
    )
    source_hash = research_source.get("aggregate_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise CanonicalEvidenceError(
            f"{attempt} has no valid research-source hash"
        )
    if source_hash != expected_source_hash:
        raise CanonicalEvidenceError(
            f"{attempt} has mismatched research_source_sha256: "
            f"{source_hash!r} != {expected_source_hash!r}"
        )
    if not isinstance(research_source.get("files"), dict):
        raise CanonicalEvidenceError(
            f"{attempt} has no research-source file inventory"
        )
    if manifest.get("outputs") != _RAW_OUTPUTS:
        raise CanonicalEvidenceError(f"{attempt} raw-output contract drifted")
    for relative in _RAW_OUTPUTS.values():
        if not (attempt / relative).exists():
            raise CanonicalEvidenceError(
                f"{attempt} is terminal-success but lacks output {relative}"
            )
    model_files = {
        attempt / "models" / f"fold-{fold}.joblib"
        for fold in range(EXPECTED_OUTER_FOLDS)
    }
    if any(not path.is_file() for path in model_files):
        raise CanonicalEvidenceError(
            f"{attempt} lacks one or more fitted outer-fold models"
        )
    completed = manifest.get("completed_at_utc")
    if not isinstance(completed, str) or not completed:
        raise CanonicalEvidenceError(
            f"{attempt} lacks a completion timestamp"
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CanonicalEvidenceError(f"{attempt} lacks a run_id")
    return {
        "scenario": scenario,
        "condition": condition,
        "condition_key": condition_key,
        "seed": seed,
        "attempt": attempt.as_posix(),
        "run_id": run_id,
        "completed_at_utc": completed,
        **{field: str(inputs[field]) for field in _COHORT_HASH_FIELDS},
        "research_source_sha256": source_hash,
    }


def unique_complete_source_hash(
    matching_by_cell: Mapping[Any, Mapping[str, Any]],
) -> str:
    """Return the sole source hash represented in every matrix cell."""

    if not matching_by_cell:
        raise CanonicalEvidenceError("source-cohort selection has no matrix cells")
    complete = set.intersection(
        *(set(by_source) for by_source in matching_by_cell.values())
    )
    if len(complete) != 1:
        raise CanonicalEvidenceError(
            "expected exactly one complete research-source cohort, "
            f"found {sorted(complete)}"
        )
    return next(iter(complete))


def select_latest_successes(
    *,
    runs_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    spec_path: Path,
    features_path: Path,
    dataset_manifest_path: Path,
) -> list[dict[str, Any]]:
    """Select one complete, internally consistent historical source cohort.

    Raw runs are immutable historical evidence. Later analysis development may
    change the working tree, so selection is anchored to a source hash shared
    by all 28 cells rather than to the current checkout. Multiple complete
    source cohorts are rejected as ambiguous.
    """

    seed = int(config["seed"])
    expected_hashes = {
        "config_sha256": file_sha256(config_path),
        "spec_sha256": file_sha256(spec_path),
        "features_sha256": file_sha256(features_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
    }
    matching_by_cell: dict[
        tuple[str, str], dict[str, list[tuple[Path, dict[str, Any]]]]
    ] = {}
    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            condition_key = f"{scenario}--{condition}"
            successes = _successful_primary_attempts(runs_root / condition_key)
            if not successes:
                raise CanonicalEvidenceError(
                    f"no successful primary attempt for {condition_key}"
                )
            matching: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
            for attempt, manifest in successes:
                inputs = manifest.get("inputs", {})
                source_hash = (
                    manifest.get("environment", {})
                    .get("research_source", {})
                    .get("aggregate_sha256")
                )
                parameters = manifest.get("parameters", {})
                if (
                    manifest.get("experiment_id") == EXPERIMENT_ID
                    and manifest.get("condition") == condition_key
                    and manifest.get("seed") == seed
                    and parameters.get("scenario") == scenario
                    and parameters.get("condition") == condition
                    and all(
                        inputs.get(field) == expected
                        for field, expected in expected_hashes.items()
                    )
                    and isinstance(source_hash, str)
                    and len(source_hash) == 64
                ):
                    matching.setdefault(source_hash, []).append(
                        (attempt, manifest)
                    )
            if not matching:
                raise CanonicalEvidenceError(
                    "no successful primary attempt matching the frozen "
                    "config/spec/features/dataset inputs for "
                    f"{condition_key}"
                )
            matching_by_cell[(scenario, condition)] = matching

    expected_source_hash = unique_complete_source_hash(matching_by_cell)
    selected: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            attempt, manifest = matching_by_cell[(scenario, condition)][
                expected_source_hash
            ][-1]
            selected.append(
                _validate_selected_manifest(
                    attempt=attempt,
                    manifest=manifest,
                    scenario=scenario,
                    condition=condition,
                    seed=seed,
                    expected_input_hashes=expected_hashes,
                    expected_source_hash=expected_source_hash,
                )
            )
    if len(selected) != EXPECTED_RUN_COUNT:
        raise CanonicalEvidenceError(
            f"expected {EXPECTED_RUN_COUNT} selected runs, got {len(selected)}"
        )
    cohort_fields = (*_COHORT_HASH_FIELDS, "research_source_sha256")
    for field in cohort_fields:
        values = {str(run[field]) for run in selected}
        if len(values) != 1:
            raise CanonicalEvidenceError(
                f"selected primary runs have mismatched {field}: {sorted(values)}"
            )
    return selected


def _as_int(value: Any, *, name: str, context: str) -> int:
    if isinstance(value, bool):
        raise CanonicalEvidenceError(f"{context} has invalid {name}: {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise CanonicalEvidenceError(
            f"{context} has invalid {name}: {value!r}"
        ) from error
    if isinstance(value, float) and converted != value:
        raise CanonicalEvidenceError(f"{context} has non-integral {name}: {value!r}")
    return converted


def _as_finite_float(value: Any, *, name: str, context: str) -> float:
    if isinstance(value, bool):
        raise CanonicalEvidenceError(f"{context} has invalid {name}: {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise CanonicalEvidenceError(
            f"{context} has invalid {name}: {value!r}"
        ) from error
    if not math.isfinite(converted):
        raise CanonicalEvidenceError(
            f"{context} has non-finite {name}: {value!r}"
        )
    return converted


def _validate_fold_assignments(
    rows: Sequence[Mapping[str, Any]],
    *,
    features_by_chart: Mapping[str, Mapping[str, Any]],
    expected_assignment_by_group: Mapping[str, tuple[int, tuple[int, ...]]],
    context: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, set[str]]]]:
    if len(rows) != EXPECTED_CHART_COUNT:
        raise CanonicalEvidenceError(
            f"{context} has {len(rows)} fold rows, expected {EXPECTED_CHART_COUNT}"
        )
    by_chart: dict[str, dict[str, Any]] = {}
    group_assignments: dict[str, tuple[int, tuple[int, ...]]] = {}
    for raw in rows:
        chart_id = str(raw.get("chart_id", ""))
        if chart_id in by_chart:
            raise CanonicalEvidenceError(
                f"{context} has duplicate fold assignment for {chart_id}"
            )
        feature = features_by_chart.get(chart_id)
        if feature is None:
            raise CanonicalEvidenceError(
                f"{context} has unknown fold-assignment chart {chart_id}"
            )
        normalized_title = str(raw.get("normalized_title", ""))
        outer_fold = _as_int(
            raw.get("outer_test_fold"),
            name="outer_test_fold",
            context=context,
        )
        if outer_fold not in range(EXPECTED_OUTER_FOLDS):
            raise CanonicalEvidenceError(
                f"{context} assigns {chart_id} to invalid fold {outer_fold}"
            )
        validation_value = raw.get("inner_validation_for_outer_folds")
        if not isinstance(validation_value, list):
            raise CanonicalEvidenceError(
                f"{context} has invalid validation-fold list for {chart_id}"
            )
        validation_folds = tuple(
            _as_int(value, name="validation fold", context=context)
            for value in validation_value
        )
        if (
            tuple(sorted(set(validation_folds))) != validation_folds
            or any(fold not in range(EXPECTED_OUTER_FOLDS) for fold in validation_folds)
            or outer_fold in validation_folds
        ):
            raise CanonicalEvidenceError(
                f"{context} has invalid validation folds for {chart_id}: "
                f"{validation_folds}"
            )
        expected_metadata = {
            "song_id": str(feature["song_id"]),
            "normalized_title": str(feature["normalized_title"]),
            "original_star": int(feature["star"]),
        }
        observed_metadata = {
            "song_id": str(raw.get("song_id", "")),
            "normalized_title": normalized_title,
            "original_star": _as_int(
                raw.get("original_star"),
                name="original_star",
                context=context,
            ),
        }
        if observed_metadata != expected_metadata:
            raise CanonicalEvidenceError(
                f"{context} metadata drift for {chart_id}: "
                f"{observed_metadata} != {expected_metadata}"
            )
        assignment = (outer_fold, validation_folds)
        previous = group_assignments.setdefault(normalized_title, assignment)
        if previous != assignment:
            raise CanonicalEvidenceError(
                f"{context} splits title group {normalized_title!r} "
                "across incompatible folds"
            )
        expected_assignment = expected_assignment_by_group.get(normalized_title)
        if expected_assignment != assignment:
            raise CanonicalEvidenceError(
                f"{context} assignment for title {normalized_title!r} "
                f"does not reproduce the frozen splitter: "
                f"{assignment} != {expected_assignment}"
            )
        by_chart[chart_id] = {
            "chart_id": chart_id,
            **observed_metadata,
            "outer_test_fold": outer_fold,
            "inner_validation_for_outer_folds": list(validation_folds),
        }
    if set(by_chart) != set(features_by_chart):
        raise CanonicalEvidenceError(f"{context} fold coverage is incomplete")

    all_groups = set(group_assignments)
    reconstructed: dict[int, dict[str, set[str]]] = {}
    for outer_fold in range(EXPECTED_OUTER_FOLDS):
        test = {
            group
            for group, (test_fold, _) in group_assignments.items()
            if test_fold == outer_fold
        }
        validation = {
            group
            for group, (_, validation_folds) in group_assignments.items()
            if outer_fold in validation_folds
        }
        train = all_groups - test - validation
        if not train or not validation or not test:
            raise CanonicalEvidenceError(
                f"{context} fold {outer_fold} has an empty partition"
            )
        if train & validation or train & test or validation & test:
            raise CanonicalEvidenceError(
                f"{context} fold {outer_fold} leaks normalized-title groups"
            )
        if train | validation | test != all_groups:
            raise CanonicalEvidenceError(
                f"{context} fold {outer_fold} does not cover all title groups"
            )
        reconstructed[outer_fold] = {
            "train": train,
            "validation": validation,
            "test": test,
        }
    return [by_chart[key] for key in sorted(by_chart)], reconstructed


def _expected_candidates(
    condition: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if condition in {
        "ordinary_huber",
        "topcoded_huber",
        "ordinal_logit",
        "ordinal_probit",
    }:
        return [{"alpha": float(value)} for value in config["alpha_grid"]]
    if condition == "ridge":
        return [
            {"alpha": float(value)} for value in config["ridge_alpha_grid"]
        ]
    if condition == "gbdt":
        return [
            {
                "learning_rate": float(row["learning_rate"]),
                "max_leaf_nodes": int(row["max_leaf_nodes"]),
            }
            for row in config["gbdt_grid"]
        ]
    if condition == "ordinal_logit_reduction":
        return [
            {"C": float(value)}
            for value in config["ordinal_reduction_c_grid"]
        ]
    raise CanonicalEvidenceError(f"unsupported condition {condition!r}")


def _serialized_parameters(parameters: Mapping[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _validate_search_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    context: str,
) -> tuple[list[dict[str, Any]], dict[int, tuple[dict[str, Any], float]]]:
    expected = _expected_candidates(condition, config)
    expected_serialized = {_serialized_parameters(row) for row in expected}
    expected_count = EXPECTED_OUTER_FOLDS * len(expected)
    if len(rows) != expected_count:
        raise CanonicalEvidenceError(
            f"{context} has {len(rows)} search rows, expected {expected_count}"
        )
    by_fold: dict[int, list[tuple[dict[str, Any], float, bool]]] = {
        fold: [] for fold in range(EXPECTED_OUTER_FOLDS)
    }
    canonical: list[dict[str, Any]] = []
    for raw in rows:
        if raw.get("scenario") != scenario or raw.get("condition") != condition:
            raise CanonicalEvidenceError(f"{context} search-row identity drift")
        fold = _as_int(raw.get("outer_fold"), name="outer_fold", context=context)
        if fold not in by_fold:
            raise CanonicalEvidenceError(f"{context} has invalid search fold {fold}")
        parameters = raw.get("parameters")
        if not isinstance(parameters, dict):
            raise CanonicalEvidenceError(f"{context} has invalid parameters object")
        normalized_parameters = json.loads(_serialized_parameters(parameters))
        mae = _as_finite_float(
            raw.get("validation_observed_mae"),
            name="validation_observed_mae",
            context=context,
        )
        selected = raw.get("selected")
        if not isinstance(selected, bool):
            raise CanonicalEvidenceError(f"{context} has non-boolean selected flag")
        by_fold[fold].append((normalized_parameters, mae, selected))
        canonical.append(
            {
                "scenario": scenario,
                "condition": condition,
                "outer_fold": fold,
                "parameters": normalized_parameters,
                "validation_observed_mae": mae,
                "selected": selected,
                **provenance,
            }
        )

    selected_by_fold: dict[int, tuple[dict[str, Any], float]] = {}
    for fold, candidates in by_fold.items():
        serialized = [_serialized_parameters(row[0]) for row in candidates]
        if len(serialized) != len(set(serialized)):
            raise CanonicalEvidenceError(
                f"{context} fold {fold} has duplicate hyperparameter candidates"
            )
        if set(serialized) != expected_serialized:
            raise CanonicalEvidenceError(
                f"{context} fold {fold} hyperparameter grid drifted"
            )
        selected_rows = [row for row in candidates if row[2]]
        if len(selected_rows) != 1:
            raise CanonicalEvidenceError(
                f"{context} fold {fold} must select exactly one candidate"
            )
        winner = min(
            candidates,
            key=lambda row: (row[1], _serialized_parameters(row[0])),
        )
        if not winner[2]:
            raise CanonicalEvidenceError(
                f"{context} fold {fold} selected a non-minimal candidate"
            )
        selected_by_fold[fold] = (winner[0], winner[1])
    canonical.sort(
        key=lambda row: (
            int(row["outer_fold"]),
            _serialized_parameters(row["parameters"]),
        )
    )
    return canonical, selected_by_fold


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
    seed: int,
    cap: float,
    assignments_by_chart: Mapping[str, Mapping[str, Any]],
    features_by_chart: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    context: str,
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_CHART_COUNT:
        raise CanonicalEvidenceError(
            f"{context} has {len(rows)} predictions, expected {EXPECTED_CHART_COUNT}"
        )
    canonical: dict[str, dict[str, Any]] = {}
    for raw in rows:
        chart_id = str(raw.get("chart_id", ""))
        if chart_id in canonical:
            raise CanonicalEvidenceError(
                f"{context} has duplicate OOF prediction for {chart_id}"
            )
        assignment = assignments_by_chart.get(chart_id)
        feature = features_by_chart.get(chart_id)
        if assignment is None or feature is None:
            raise CanonicalEvidenceError(
                f"{context} predicts an unknown chart {chart_id}"
            )
        if (
            raw.get("scenario") != scenario
            or raw.get("condition") != condition
            or _as_int(raw.get("seed"), name="seed", context=context) != seed
        ):
            raise CanonicalEvidenceError(
                f"{context} prediction identity drift for {chart_id}"
            )
        outer_fold = _as_int(
            raw.get("outer_fold"),
            name="outer_fold",
            context=context,
        )
        if outer_fold != assignment["outer_test_fold"]:
            raise CanonicalEvidenceError(
                f"{context} prediction fold drift for {chart_id}"
            )
        original_star = _as_finite_float(
            raw.get("original_star"),
            name="original_star",
            context=context,
        )
        observed_star = _as_finite_float(
            raw.get("observed_star"),
            name="observed_star",
            context=context,
        )
        active_cap = _as_finite_float(
            raw.get("active_cap"),
            name="active_cap",
            context=context,
        )
        latent = _as_finite_float(
            raw.get("latent_score"),
            name="latent_score",
            context=context,
        )
        expected = _as_finite_float(
            raw.get("expected_observed_label"),
            name="expected_observed_label",
            context=context,
        )
        clipped = _as_finite_float(
            raw.get("clipped_observed_prediction"),
            name="clipped_observed_prediction",
            context=context,
        )
        expected_metadata = {
            "song_id": str(feature["song_id"]),
            "title": str(feature["title"]),
            "normalized_title": str(feature["normalized_title"]),
            "difficulty": "oni",
            "difficulty_id": 3,
            "original_star": float(feature["star"]),
            "primary_split": str(feature["primary_split"]),
        }
        observed_metadata = {
            "song_id": str(raw.get("song_id", "")),
            "title": str(raw.get("title", "")),
            "normalized_title": str(raw.get("normalized_title", "")),
            "difficulty": raw.get("difficulty"),
            "difficulty_id": _as_int(
                raw.get("difficulty_id"),
                name="difficulty_id",
                context=context,
            ),
            "original_star": original_star,
            "primary_split": str(raw.get("primary_split", "")),
        }
        if observed_metadata != expected_metadata:
            raise CanonicalEvidenceError(
                f"{context} prediction metadata drift for {chart_id}"
            )
        expected_observed = min(original_star, cap)
        if (
            not math.isclose(active_cap, cap, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(
                observed_star,
                expected_observed,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or raw.get("is_top_coded") is not (original_star >= cap)
            or raw.get("is_strictly_hidden") is not (original_star > cap)
        ):
            raise CanonicalEvidenceError(
                f"{context} top-coding contract drift for {chart_id}"
            )
        expected_clipped = min(max(expected, 1.0), cap)
        if not math.isclose(
            clipped,
            expected_clipped,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CanonicalEvidenceError(
                f"{context} clipped prediction drift for {chart_id}"
            )
        canonical[chart_id] = {
            "scenario": scenario,
            "condition": condition,
            "seed": seed,
            "outer_fold": outer_fold,
            "chart_id": chart_id,
            **observed_metadata,
            "observed_star": observed_star,
            "active_cap": active_cap,
            "is_top_coded": bool(raw["is_top_coded"]),
            "is_strictly_hidden": bool(raw["is_strictly_hidden"]),
            "latent_score": latent,
            "expected_observed_label": expected,
            "clipped_observed_prediction": clipped,
            **provenance,
        }
    if set(canonical) != set(features_by_chart):
        raise CanonicalEvidenceError(f"{context} OOF coverage is incomplete")
    return [canonical[key] for key in sorted(canonical)]


def _validate_fold_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
    reconstructed: Mapping[int, Mapping[str, set[str]]],
    features_by_chart: Mapping[str, Mapping[str, Any]],
    selected_by_fold: Mapping[int, tuple[dict[str, Any], float]],
    provenance: Mapping[str, Any],
    context: str,
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_OUTER_FOLDS:
        raise CanonicalEvidenceError(
            f"{context} has {len(rows)} diagnostics, "
            f"expected {EXPECTED_OUTER_FOLDS}"
        )
    by_fold: dict[int, dict[str, Any]] = {}
    for raw in rows:
        if raw.get("scenario") != scenario or raw.get("condition") != condition:
            raise CanonicalEvidenceError(f"{context} diagnostic identity drift")
        fold = _as_int(raw.get("outer_fold"), name="outer_fold", context=context)
        if fold in by_fold or fold not in range(EXPECTED_OUTER_FOLDS):
            raise CanonicalEvidenceError(
                f"{context} has duplicate or invalid diagnostic fold {fold}"
            )
        partitions = reconstructed[fold]
        chart_partition = {
            name: {
                chart_id
                for chart_id, feature in features_by_chart.items()
                if str(feature["normalized_title"]) in groups
            }
            for name, groups in partitions.items()
        }
        expected_counts = {
            "inner_train_charts": len(chart_partition["train"]),
            "inner_validation_charts": len(chart_partition["validation"]),
            "outer_train_charts": (
                len(chart_partition["train"])
                + len(chart_partition["validation"])
            ),
            "outer_test_charts": len(chart_partition["test"]),
            "inner_train_groups": len(partitions["train"]),
            "inner_validation_groups": len(partitions["validation"]),
            "outer_test_groups": len(partitions["test"]),
        }
        observed_counts = {
            field: _as_int(raw.get(field), name=field, context=context)
            for field in expected_counts
        }
        if observed_counts != expected_counts:
            raise CanonicalEvidenceError(
                f"{context} fold {fold} count drift: "
                f"{observed_counts} != {expected_counts}"
            )
        selected_parameters, selected_mae = selected_by_fold[fold]
        if raw.get("selected_parameters") != selected_parameters:
            raise CanonicalEvidenceError(
                f"{context} fold {fold} selected-parameter drift"
            )
        diagnostic_mae = _as_finite_float(
            raw.get("validation_observed_mae"),
            name="validation_observed_mae",
            context=context,
        )
        if not math.isclose(
            diagnostic_mae,
            selected_mae,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise CanonicalEvidenceError(
                f"{context} fold {fold} validation-MAE drift"
            )
        test_mae = _as_finite_float(
            raw.get("test_observed_mae"),
            name="test_observed_mae",
            context=context,
        )
        by_fold[fold] = {
            "scenario": scenario,
            "condition": condition,
            "outer_fold": fold,
            **observed_counts,
            "selected_parameters": selected_parameters,
            "validation_observed_mae": diagnostic_mae,
            "test_observed_mae": test_mae,
            "verified_title_overlap_counts": {
                "inner_train_vs_inner_validation": len(
                    partitions["train"] & partitions["validation"]
                ),
                "inner_train_vs_outer_test": len(
                    partitions["train"] & partitions["test"]
                ),
                "inner_validation_vs_outer_test": len(
                    partitions["validation"] & partitions["test"]
                ),
            },
            "verified_title_partition_union_count": len(
                partitions["train"]
                | partitions["validation"]
                | partitions["test"]
            ),
            **provenance,
        }
    if set(by_fold) != set(range(EXPECTED_OUTER_FOLDS)):
        raise CanonicalEvidenceError(
            f"{context} diagnostic fold coverage is incomplete"
        )
    return [by_fold[fold] for fold in range(EXPECTED_OUTER_FOLDS)]


def _cohort_hashes(selected: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    fields = (*_COHORT_HASH_FIELDS, "research_source_sha256")
    return {field: str(selected[0][field]) for field in fields}


def _provenance(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(run["run_id"]),
        "raw_attempt": str(run["attempt"]),
    }


def _expected_assignment_by_group(
    features_by_chart: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, tuple[int, tuple[int, ...]]]:
    folds = build_song_group_folds(
        [
            str(feature["normalized_title"])
            for feature in features_by_chart.values()
        ],
        n_folds=int(config["outer_folds"]),
        seed=int(config["seed"]),
        validation_fraction=float(config["inner_validation_fraction"]),
    )
    outer_by_group = {
        group: int(fold.outer_fold)
        for fold in folds
        for group in fold.test_groups
    }
    validation_by_group: dict[str, list[int]] = {
        group: [] for group in outer_by_group
    }
    for fold in folds:
        for group in fold.validation_groups:
            validation_by_group[group].append(int(fold.outer_fold))
    return {
        group: (outer_by_group[group], tuple(sorted(validation_by_group[group])))
        for group in sorted(outer_by_group)
    }


def build_canonical_tables(
    *,
    runs_root: Path,
    features_path: Path,
    config_path: Path,
    spec_path: Path,
    dataset_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    _require_exact_primary_config(config)
    _, features_by_chart = _oni_feature_panel(features_path)
    expected_assignment_by_group = _expected_assignment_by_group(
        features_by_chart,
        config,
    )
    selected = select_latest_successes(
        runs_root=runs_root,
        config=config,
        config_path=config_path,
        spec_path=spec_path,
        features_path=features_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    canonical_assignments: list[dict[str, Any]] | None = None
    model_scores: list[dict[str, Any]] = []
    validation_search: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for run in selected:
        scenario = str(run["scenario"])
        condition = str(run["condition"])
        attempt = Path(str(run["attempt"]))
        context = f"{scenario}--{condition} ({attempt})"
        assignments, reconstructed = _validate_fold_assignments(
            _read_jsonl(attempt / "fold_assignments.jsonl"),
            features_by_chart=features_by_chart,
            expected_assignment_by_group=expected_assignment_by_group,
            context=context,
        )
        if canonical_assignments is None:
            canonical_assignments = assignments
        elif assignments != canonical_assignments:
            raise CanonicalEvidenceError(
                f"{context} fold assignments differ from the canonical cohort"
            )
        assignments_by_chart = {
            str(row["chart_id"]): row for row in assignments
        }
        provenance = _provenance(run)
        search_rows, selected_by_fold = _validate_search_rows(
            _read_jsonl(attempt / "validation_search.jsonl"),
            scenario=scenario,
            condition=condition,
            config=config,
            provenance=provenance,
            context=context,
        )
        score_rows = _validate_prediction_rows(
            _read_jsonl(attempt / "oof_predictions.jsonl"),
            scenario=scenario,
            condition=condition,
            seed=int(config["seed"]),
            cap=SCENARIO_CAPS[scenario],
            assignments_by_chart=assignments_by_chart,
            features_by_chart=features_by_chart,
            provenance=provenance,
            context=context,
        )
        diagnostic_rows = _validate_fold_diagnostics(
            _read_jsonl(attempt / "fold_diagnostics.jsonl"),
            scenario=scenario,
            condition=condition,
            reconstructed=reconstructed,
            features_by_chart=features_by_chart,
            selected_by_fold=selected_by_fold,
            provenance=provenance,
            context=context,
        )
        validation_search.extend(search_rows)
        model_scores.extend(score_rows)
        fold_diagnostics.extend(diagnostic_rows)

    if canonical_assignments is None:
        raise CanonicalEvidenceError("no canonical fold assignments were selected")
    if len(model_scores) != EXPECTED_SCORE_COUNT:
        raise CanonicalEvidenceError(
            f"expected {EXPECTED_SCORE_COUNT} score rows, got {len(model_scores)}"
        )
    if len(fold_diagnostics) != EXPECTED_FOLD_DIAGNOSTIC_COUNT:
        raise CanonicalEvidenceError(
            "expected "
            f"{EXPECTED_FOLD_DIAGNOSTIC_COUNT} fold diagnostics, "
            f"got {len(fold_diagnostics)}"
        )
    score_keys = [
        (row["scenario"], row["condition"], row["chart_id"])
        for row in model_scores
    ]
    if len(score_keys) != len(set(score_keys)):
        raise CanonicalEvidenceError("canonical model-score grain is not unique")
    diagnostic_keys = [
        (row["scenario"], row["condition"], row["outer_fold"])
        for row in fold_diagnostics
    ]
    if len(diagnostic_keys) != len(set(diagnostic_keys)):
        raise CanonicalEvidenceError("canonical fold-diagnostic grain is not unique")

    scenario_order = {value: index for index, value in enumerate(SCENARIOS)}
    condition_order = {value: index for index, value in enumerate(CONDITIONS)}
    model_scores.sort(
        key=lambda row: (
            scenario_order[str(row["scenario"])],
            condition_order[str(row["condition"])],
            str(row["chart_id"]),
        )
    )
    validation_search.sort(
        key=lambda row: (
            scenario_order[str(row["scenario"])],
            condition_order[str(row["condition"])],
            int(row["outer_fold"]),
            _serialized_parameters(row["parameters"]),
        )
    )
    fold_diagnostics.sort(
        key=lambda row: (
            scenario_order[str(row["scenario"])],
            condition_order[str(row["condition"])],
            int(row["outer_fold"]),
        )
    )

    output_path.mkdir(parents=True, exist_ok=True)
    selected_path = output_path / "latest_successful_runs.json"
    assignments_path = output_path / "fold_assignments.jsonl"
    scores_path = output_path / "model_scores.jsonl"
    search_path = output_path / "validation_search.jsonl"
    diagnostics_path = output_path / "fold_diagnostics.jsonl"
    hashes = _cohort_hashes(selected)
    atomic_write_json(
        selected_path,
        {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "selection_rule": (
                "lexically latest attempt with terminal success in both "
                "status and manifest, primary_evidence=true, exact frozen "
                "config/spec/features/dataset hashes, and membership "
                "in the unique complete 28-cell research-source cohort"
            ),
            "expected_scenarios": list(SCENARIOS),
            "expected_conditions": list(CONDITIONS),
            "expected_run_count": EXPECTED_RUN_COUNT,
            "cohort_hashes": hashes,
            "runs": selected,
        },
    )
    atomic_write_jsonl(assignments_path, canonical_assignments)
    atomic_write_jsonl(scores_path, model_scores)
    atomic_write_jsonl(search_path, validation_search)
    atomic_write_jsonl(diagnostics_path, fold_diagnostics)

    outputs = (
        selected_path,
        assignments_path,
        scores_path,
        search_path,
        diagnostics_path,
    )
    completion_times = sorted(str(run["completed_at_utc"]) for run in selected)
    row_counts = {
        "fold_assignments.jsonl": len(canonical_assignments),
        "model_scores.jsonl": len(model_scores),
        "validation_search.jsonl": len(validation_search),
        "fold_diagnostics.jsonl": len(fold_diagnostics),
    }
    manifest = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": completion_times[-1],
        "generated_at_semantics": (
            "latest selected raw completion timestamp; deterministic from "
            "the selected evidence cohort"
        ),
        "transformation_script": (
            "scripts/research/build_topcoded_oof_tables.py"
        ),
        "transformation_command": (
            "PYTHONPATH=. uv run python "
            "scripts/research/build_topcoded_oof_tables.py"
        ),
        "source_config": config_path.as_posix(),
        "source_spec": spec_path.as_posix(),
        "source_feature_table": features_path.as_posix(),
        "source_dataset_manifest": dataset_manifest_path.as_posix(),
        "cohort_hashes": hashes,
        "source_attempts": [str(run["attempt"]) for run in selected],
        "expected_matrix": {
            "scenarios": list(SCENARIOS),
            "conditions": list(CONDITIONS),
            "run_count": EXPECTED_RUN_COUNT,
            "outer_folds": EXPECTED_OUTER_FOLDS,
            "oni_charts": EXPECTED_CHART_COUNT,
            "normalized_title_groups": EXPECTED_SONG_GROUP_COUNT,
        },
        "row_grain": {
            "fold_assignments.jsonl": "one row per Oni chart",
            "model_scores.jsonl": (
                "one row per scenario/condition/Oni chart"
            ),
            "validation_search.jsonl": (
                "one row per scenario/condition/outer fold/candidate"
            ),
            "fold_diagnostics.jsonl": (
                "one row per scenario/condition/outer fold"
            ),
        },
        "row_counts": row_counts,
        "output_sha256": {path.name: file_sha256(path) for path in outputs},
        "exclusions": {
            "smoke_attempts": (
                "condition directories suffixed __smoke are outside the "
                "frozen primary matrix"
            ),
            "failed_or_incomplete_attempts": (
                "attempts without terminal success in both status.json and "
                "run_manifest.json are retained raw and excluded"
            ),
            "non_primary_attempts": (
                "attempts with primary_evidence other than literal true are "
                "retained raw and excluded"
            ),
            "superseded_attempts": (
                "older successful primary attempts in a matrix cell are "
                "retained raw and excluded by the latest-success rule"
            ),
            "hash_mismatches": (
                "attempts with config/spec/feature/dataset/"
                "research-source hash drift cannot enter the current cohort"
            ),
        },
    }
    atomic_write_json(output_path / "MANIFEST.json", manifest)
    return {
        "selected_runs": len(selected),
        "fold_assignment_rows": len(canonical_assignments),
        "model_score_rows": len(model_scores),
        "validation_search_rows": len(validation_search),
        "fold_diagnostic_rows": len(fold_diagnostics),
        "cohort_hashes": hashes,
    }


def main() -> None:
    args = parse_args()
    result = build_canonical_tables(
        runs_root=args.runs,
        features_path=args.features,
        config_path=args.config,
        spec_path=args.spec,
        dataset_manifest_path=args.dataset_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
