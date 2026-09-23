#!/usr/bin/env python
"""Analyze canonical top-coded OOF evidence and generate claim artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
)
from TaikoChartEstimator.research.oof import (
    build_hidden_tail_panel,
    hidden_tail_rank_metrics,
    paired_song_group_bootstrap_difference,
    top_code_targets,
)
from TaikoChartEstimator.research.release import (
    GLOBAL_CHART_COUNT,
    GLOBAL_CONDITIONS,
    GLOBAL_EXPERIMENT_ID,
    GLOBAL_OUTER_FOLDS,
    GLOBAL_SCENARIOS,
    GlobalReleaseInputs,
    validate_global_release_manifest,
)

EXPERIMENT_ID = GLOBAL_EXPERIMENT_ID
SCENARIOS = GLOBAL_SCENARIOS
CONTROLLED_SCENARIOS = ("cap7", "cap8", "cap9")
CONDITIONS = GLOBAL_CONDITIONS
CAPS = {"native": 10.0, "cap7": 7.0, "cap8": 8.0, "cap9": 9.0}
EXPECTED_CHARTS = GLOBAL_CHART_COUNT
EXPECTED_SCORE_ROWS = len(SCENARIOS) * len(CONDITIONS) * EXPECTED_CHARTS
STAR_SCALE_CONDITIONS = {
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
}
PRIMARY_FIRST = "topcoded_huber"
PRIMARY_SECOND = "ordinary_huber"


class AnalysisEvidenceError(RuntimeError):
    """Raised when canonical inputs cannot support the frozen analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/configs/primary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027_topcoded_oof_v1/reports"),
    )
    return parser.parse_args()


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisEvidenceError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisEvidenceError(f"{path} must contain one JSON object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AnalysisEvidenceError(f"cannot read JSONL table {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisEvidenceError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise AnalysisEvidenceError(
                f"{path}:{line_number} must contain one JSON object"
            )
        rows.append(value)
    return rows


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise AnalysisEvidenceError(f"{name} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisEvidenceError(f"{name} must be numeric") from error
    if not math.isfinite(converted):
        raise AnalysisEvidenceError(f"{name} must be finite")
    return converted


def _validate_inputs(
    *,
    release_manifest: Path,
    config_path: Path,
) -> tuple[dict[str, Any], GlobalReleaseInputs, list[dict[str, Any]]]:
    config = _object(config_path)
    if (
        config.get("experiment_id") != EXPERIMENT_ID
        or config.get("primary_evidence") is not True
        or tuple(config.get("conditions", ())) != CONDITIONS
        or tuple(config.get("scenarios", {})) != SCENARIOS
    ):
        raise AnalysisEvidenceError("primary config differs from the frozen matrix")
    try:
        release = validate_global_release_manifest(release_manifest)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise AnalysisEvidenceError(str(error)) from error
    scores = _rows(release.model_scores)
    if len(scores) != EXPECTED_SCORE_ROWS:
        raise AnalysisEvidenceError(
            f"expected {EXPECTED_SCORE_ROWS} canonical scores, found {len(scores)}"
        )
    keys = [
        (str(row["scenario"]), str(row["condition"]), str(row["chart_id"]))
        for row in scores
    ]
    if len(keys) != len(set(keys)):
        raise AnalysisEvidenceError("canonical score grain is not unique")
    if {key[0] for key in keys} != set(SCENARIOS):
        raise AnalysisEvidenceError("canonical scenario coverage is incomplete")
    if {key[1] for key in keys} != set(CONDITIONS):
        raise AnalysisEvidenceError("canonical condition coverage is incomplete")
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for scenario, condition, _ in keys:
        counts[(scenario, condition)] += 1
    if any(
        counts[(scenario, condition)] != EXPECTED_CHARTS
        for scenario in SCENARIOS
        for condition in CONDITIONS
    ):
        raise AnalysisEvidenceError("a scenario-condition lacks complete OOF coverage")
    assignments = _rows(release.fold_assignments)
    assignments_by_chart = {
        str(row["chart_id"]): row for row in assignments
    }
    if len(assignments_by_chart) != EXPECTED_CHARTS:
        raise AnalysisEvidenceError("release fold assignments are not unique")
    folds_by_title: dict[str, set[int]] = defaultdict(set)
    for row in assignments:
        folds_by_title[str(row["normalized_title"])].add(
            int(row["outer_test_fold"])
        )
    if (
        any(len(folds) != 1 for folds in folds_by_title.values())
        or len(folds_by_title) != 1_019
        or {fold for folds in folds_by_title.values() for fold in folds}
        != set(range(GLOBAL_OUTER_FOLDS))
    ):
        raise AnalysisEvidenceError("release fold assignments violate title grouping")
    mismatched_scores = [
        str(row["chart_id"])
        for row in scores
        if str(row["chart_id"]) not in assignments_by_chart
        or str(row["normalized_title"])
        != str(assignments_by_chart[str(row["chart_id"])]["normalized_title"])
        or int(row["outer_fold"])
        != int(assignments_by_chart[str(row["chart_id"])]["outer_test_fold"])
        or float(row["original_star"])
        != float(assignments_by_chart[str(row["chart_id"])]["original_star"])
    ]
    if mismatched_scores:
        raise AnalysisEvidenceError(
            "release OOF scores disagree with fold assignments: "
            f"{mismatched_scores[:10]}"
        )
    return config, release, scores


def _score_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["condition"]))].append(dict(row))
    return {
        key: sorted(values, key=lambda row: str(row["chart_id"]))
        for key, values in grouped.items()
    }


def _hidden_panel_and_predictions(
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    scenario: str,
) -> tuple[Any, dict[str, np.ndarray]]:
    reference = list(index[(scenario, CONDITIONS[0])])
    chart_ids = [str(row["chart_id"]) for row in reference]
    targets = np.asarray(
        [_finite_float(row["original_star"], name="original_star") for row in reference],
        dtype=np.float64,
    )
    cap = CAPS[scenario]
    coded = top_code_targets(
        targets,
        np.full(len(targets), 3, dtype=np.int64),
        cap=cap,
        selected_course_ids={3},
    )
    panel = build_hidden_tail_panel(
        coded,
        [str(row["normalized_title"]) for row in reference],
    )
    predictions: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        rows = list(index[(scenario, condition)])
        if [str(row["chart_id"]) for row in rows] != chart_ids:
            raise AnalysisEvidenceError(
                f"{scenario}/{condition} chart order differs across conditions"
            )
        predictions[condition] = np.asarray(
            [
                _finite_float(row["latent_score"], name="latent_score")
                for row in rows
            ],
            dtype=np.float64,
        )
    return panel, predictions


def _hidden_tail_analysis(
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for scenario in CONTROLLED_SCENARIOS:
        panel, predictions = _hidden_panel_and_predictions(index, scenario)
        by_condition: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            row = {
                "schema_version": "topcoded_hidden_tail_v1",
                "scenario": scenario,
                "active_cap": CAPS[scenario],
                "condition": condition,
                "panel_definition": "original_star_at_or_above_active_cap",
                "score_field": "latent_score",
                "is_primary_panel": scenario == "cap8",
                **hidden_tail_rank_metrics(panel, predictions[condition]),
            }
            metrics.append(row)
            by_condition[condition] = row

        for second in CONDITIONS:
            if second == PRIMARY_FIRST:
                continue
            first_metrics = by_condition[PRIMARY_FIRST]
            second_metrics = by_condition[second]
            common = {
                "schema_version": "topcoded_hidden_contrast_v1",
                "scenario": scenario,
                "active_cap": CAPS[scenario],
                "first_condition": PRIMARY_FIRST,
                "second_condition": second,
                "contrast_direction": "first_minus_second",
                "is_primary": scenario == "cap8" and second == PRIMARY_SECOND,
                "is_sensitivity": (
                    scenario in {"cap7", "cap9"} and second == PRIMARY_SECOND
                ),
                "spearman_point_difference": (
                    float(first_metrics["spearman_rho"])
                    - float(second_metrics["spearman_rho"])
                ),
                "kendall_tau_b_point_difference": (
                    float(first_metrics["kendall_tau_b"])
                    - float(second_metrics["kendall_tau_b"])
                ),
                "strict_pair_accuracy_point_difference": (
                    float(first_metrics["strict_comparable_pair_accuracy"])
                    - float(second_metrics["strict_comparable_pair_accuracy"])
                ),
            }
            if second == PRIMARY_SECOND:
                bootstrap = paired_song_group_bootstrap_difference(
                    panel,
                    predictions[PRIMARY_FIRST],
                    predictions[PRIMARY_SECOND],
                    replicates=int(config["bootstrap_replicates"]),
                    seed=int(config["bootstrap_seed"]),
                )
                spearman = bootstrap["spearman_difference"]
                strict_pair = bootstrap["strict_pair_accuracy_difference"]
                contrasts.append(
                    {
                        **common,
                        "uncertainty_status": "paired_title_bootstrap",
                        "bootstrap_method": "paired_normalized_title_resampling_v1",
                        "bootstrap_replicates": int(bootstrap["bootstrap_replicates"]),
                        "bootstrap_seed": int(bootstrap["bootstrap_seed"]),
                        "confidence_level": float(bootstrap["confidence_level"]),
                        "title_clusters": int(bootstrap["song_group_count"]),
                        "spearman_bootstrap_mean_difference": spearman[
                            "bootstrap_mean_difference"
                        ],
                        "spearman_bootstrap_ci_low": spearman["bootstrap_ci_low"],
                        "spearman_bootstrap_ci_high": spearman["bootstrap_ci_high"],
                        "spearman_valid_replicates": int(spearman["valid_replicates"]),
                        "spearman_undefined_replicates": int(
                            spearman["undefined_replicates"]
                        ),
                        "spearman_undefined_reason": spearman["undefined_reason"],
                        "strict_pair_accuracy_bootstrap_mean_difference": strict_pair[
                            "bootstrap_mean_difference"
                        ],
                        "strict_pair_accuracy_bootstrap_ci_low": strict_pair[
                            "bootstrap_ci_low"
                        ],
                        "strict_pair_accuracy_bootstrap_ci_high": strict_pair[
                            "bootstrap_ci_high"
                        ],
                        "strict_pair_accuracy_valid_replicates": int(
                            strict_pair["valid_replicates"]
                        ),
                        "strict_pair_accuracy_undefined_replicates": int(
                            strict_pair["undefined_replicates"]
                        ),
                        "strict_pair_accuracy_undefined_reason": strict_pair[
                            "undefined_reason"
                        ],
                    }
                )
            else:
                contrasts.append(
                    {
                        **common,
                        "uncertainty_status": "point_estimate_only",
                        "bootstrap_method": None,
                        "bootstrap_replicates": None,
                        "reason_no_interval": (
                            "not_a_prespecified_topcoded_vs_ordinary_contrast"
                        ),
                    }
                )
    metrics.sort(
        key=lambda row: (
            CONTROLLED_SCENARIOS.index(str(row["scenario"])),
            CONDITIONS.index(str(row["condition"])),
        )
    )
    contrasts.sort(
        key=lambda row: (
            CONTROLLED_SCENARIOS.index(str(row["scenario"])),
            CONDITIONS.index(str(row["second_condition"])),
        )
    )
    return metrics, contrasts


def _rank_statistic(
    targets: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    if np.unique(targets).size < 2 or np.unique(scores).size < 2:
        raise AnalysisEvidenceError("native catalog rank statistic is undefined")
    spearman = float(spearmanr(targets, scores).statistic)
    kendall = float(kendalltau(targets, scores, variant="b").statistic)
    if not math.isfinite(spearman) or not math.isfinite(kendall):
        raise AnalysisEvidenceError("native catalog rank statistic is non-finite")
    return spearman, kendall


def _native_catalog_analysis(
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rows = list(index[("native", condition)])
        targets = np.asarray(
            [float(row["original_star"]) for row in rows],
            dtype=np.float64,
        )
        latent = np.asarray(
            [float(row["latent_score"]) for row in rows],
            dtype=np.float64,
        )
        expected = np.asarray(
            [float(row["expected_observed_label"]) for row in rows],
            dtype=np.float64,
        )
        official_prediction = np.clip(expected, 1.0, 10.0)
        spearman, kendall = _rank_statistic(targets, latent)
        top = targets >= 10.0
        row: dict[str, Any] = {
            "schema_version": "topcoded_native_catalog_v1",
            "scenario": "native",
            "condition": condition,
            "chart_count": len(rows),
            "title_group_count": len(
                {str(candidate["normalized_title"]) for candidate in rows}
            ),
            "official_star_mae": float(
                np.abs(official_prediction - targets).mean()
            ),
            "official_star_rmse": float(
                np.sqrt(np.square(official_prediction - targets).mean())
            ),
            "spearman_rho": spearman,
            "kendall_tau_b": kendall,
            "ranking_score_field": "latent_score",
            "official_prediction_field": "expected_observed_label_clipped_1_10",
            "top_category_count": int(top.sum()),
        }
        if condition in STAR_SCALE_CONDITIONS:
            shortfall = np.maximum(10.0 - latent[top], 0.0)
            row.update(
                {
                    "top_category_lower_bound_violation_rate": float(
                        np.mean(shortfall > 0.0)
                    ),
                    "top_category_mean_shortfall": float(shortfall.mean()),
                    "lower_bound_diagnostic_status": "defined_star_scale_score",
                }
            )
        else:
            row.update(
                {
                    "top_category_lower_bound_violation_rate": None,
                    "top_category_mean_shortfall": None,
                    "lower_bound_diagnostic_status": (
                        "not_applicable_latent_score_not_in_star_units"
                    ),
                }
            )
        results.append(row)
    return results



def main() -> None:
    args = parse_args()
    config, release, scores = _validate_inputs(
        release_manifest=args.release_manifest,
        config_path=args.config,
    )
    index = _score_index(scores)
    hidden_metrics, hidden_contrasts = _hidden_tail_analysis(index, config)
    native_metrics = _native_catalog_analysis(index)

    output = args.output
    output_paths = {
        "hidden_tail_metrics": output / "hidden_tail_metrics.jsonl",
        "hidden_tail_contrasts": output / "hidden_tail_contrasts.jsonl",
        "native_catalog_metrics": output / "native_catalog_metrics.jsonl",
    }
    atomic_write_jsonl(output_paths["hidden_tail_metrics"], hidden_metrics)
    atomic_write_jsonl(output_paths["hidden_tail_contrasts"], hidden_contrasts)
    atomic_write_jsonl(output_paths["native_catalog_metrics"], native_metrics)

    manifest = {
        "schema_version": "topcoded_oof_analysis",
        "experiment_id": EXPERIMENT_ID,
        "analysis_script": "scripts/research/analyze_topcoded_oof.py",
        "release_manifest": {
            "path": args.release_manifest.as_posix(),
            "sha256": file_sha256(release.manifest_path),
        },
        "input_files": {
            "config": {
                "path": args.config.as_posix(),
                "sha256": file_sha256(args.config),
            },
            "fold_assignments": {
                "path": str(
                    release.manifest["files"]["fold_assignments"]["path"]
                ),
                "sha256": file_sha256(release.fold_assignments),
            },
            "model_scores": {
                "path": str(release.manifest["files"]["model_scores"]["path"]),
                "sha256": file_sha256(release.model_scores),
            },
        },
        "output_files": {
            name: {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
            for name, path in sorted(output_paths.items())
        },
        "bootstrap": {
            "method": "paired_normalized_title_resampling_v1",
            "replicates": int(config["bootstrap_replicates"]),
            "seed": int(config["bootstrap_seed"]),
        },
        "analysis_scope": {
            "controlled_hidden_tail_scenarios": list(CONTROLLED_SCENARIOS),
            "model_evidence_source": "frozen_title_grouped_oof_predictions",
        },
    }
    atomic_write_json(output / "analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "hidden_metric_rows": len(hidden_metrics),
                "hidden_contrast_rows": len(hidden_contrasts),
                "native_metric_rows": len(native_metrics),
                "output_sha256": {
                    name: file_sha256(path)
                    for name, path in sorted(output_paths.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
