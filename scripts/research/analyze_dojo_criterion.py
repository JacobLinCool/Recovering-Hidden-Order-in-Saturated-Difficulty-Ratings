#!/usr/bin/env python
"""Evaluate frozen OOF scores against Dan-i Dojo curriculum placements."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from TaikoChartEstimator.research.dojo import (
    HIGH_DAN_ORDER,
    DojoDataError,
    estimate_with_uncertainty,
    leave_one_year_out,
    macro_year_spearman,
    match_course_placements,
    primary_ten_star_panel,
    validate_course_placements,
    with_analysis_tier,
    within_year_pair_accuracy,
)
from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
)
from TaikoChartEstimator.research.release import (
    validate_global_release_manifest,
    validate_sequence_release_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/icassp2027_dojo/config.json"),
    )
    parser.add_argument(
        "--placements",
        type=Path,
        default=Path("experiments/icassp2027_dojo/data/course_placements.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("experiments/icassp2027_dojo/data/SOURCE_MANIFEST.json"),
    )
    parser.add_argument(
        "--global-release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--sequence-release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_sequence_oof/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027_dojo/reports"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DojoDataError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_sequence_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                **row,
                "original_star": float(row["evaluation_original_label"]),
                "title": str(row.get("title", "")),
                "difficulty": "oni",
            }
        )
    return normalized


def validate_source_manifest(
    manifest: Mapping[str, Any], placement_path: Path
) -> None:
    if manifest.get("schema_version") != "icassp2027_dojo_source_manifest_v1":
        raise DojoDataError("unsupported Dojo source manifest")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise DojoDataError("Dojo source manifest has no output record")
    if Path(str(output.get("path", ""))) != placement_path:
        raise DojoDataError("Dojo source manifest points to a different placement file")
    if output.get("sha256") != file_sha256(placement_path):
        raise DojoDataError("course placement snapshot differs from source manifest")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping) or len(source_files) != 7:
        raise DojoDataError("Dojo source manifest needs six course files and songs")
    for record in source_files.values():
        if not isinstance(record, Mapping):
            raise DojoDataError("invalid source file record")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise DojoDataError(f"Dojo raw source is missing or stale: {path}")


def require_expected_counts(
    placements: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, int]:
    high = [row for row in placements if row["dan"] in HIGH_DAN_ORDER]
    high_ten = [row for row in high if int(row["official_star"]) == 10]
    high_ten_oni = [row for row in high_ten if row["difficulty"] == "oni"]
    counts = {
        "total_placements": len(placements),
        "high_dan_placements": len(high),
        "high_dan_ten_star_placements": len(high_ten),
        "high_dan_ten_star_oni_placements": len(high_ten_oni),
    }
    expected = {
        "total_placements": int(config["expected_total_placements"]),
        "high_dan_placements": int(config["expected_high_dan_placements"]),
        "high_dan_ten_star_placements": int(
            config["expected_high_dan_ten_star_placements"]
        ),
        "high_dan_ten_star_oni_placements": int(
            config["expected_high_dan_ten_star_oni_placements"]
        ),
    }
    if counts != expected:
        raise DojoDataError(f"Dojo source counts drifted: {counts} != {expected}")
    return counts


def _primary_per_year(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["year"])].append(row)
    output: list[dict[str, Any]] = []
    for year, year_rows in sorted(grouped.items()):
        tiers = np.asarray([float(row["analysis_tier"]) for row in year_rows])
        scores = np.asarray([float(row["latent_score"]) for row in year_rows])
        rho = float(stats.spearmanr(tiers, scores).statistic)
        pair_accuracy, pair_count = within_year_pair_accuracy(year_rows)
        output.append(
            {
                "year": year,
                "placement_count": len(year_rows),
                "tier_count": int(np.unique(tiers).size),
                "spearman_rho": rho,
                "within_year_pair_accuracy": pair_accuracy,
                "pair_count": pair_count,
                "minimum_score": float(np.min(scores)),
                "maximum_score": float(np.max(scores)),
            }
        )
    return output


def _position_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (int(row["year"]), str(row["dan"]))
        position = int(row["position"])
        if position in grouped[key]:
            raise DojoDataError(f"duplicate position within course: {key}, {position}")
        grouped[key][position] = float(row["latent_score"])
    differences = [
        positions[3] - positions[1]
        for positions in grouped.values()
        if 1 in positions and 3 in positions
    ]
    if not differences:
        return {
            "complete_course_count": 0,
            "mean_position_3_minus_1": None,
            "median_position_3_minus_1": None,
            "positive_fraction": None,
        }
    return {
        "complete_course_count": len(differences),
        "mean_position_3_minus_1": float(np.mean(differences)),
        "median_position_3_minus_1": float(np.median(differences)),
        "positive_fraction": float(np.mean(np.asarray(differences) > 0.0)),
    }


def _control_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_rows = with_analysis_tier(rows, high_only=False)
    all_pair, all_pair_count = within_year_pair_accuracy(all_rows)
    lower_pair, lower_pair_count = within_year_pair_accuracy(
        all_rows,
        require_equal_star=True,
        maximum_star=9,
    )
    equal_star_pair, equal_star_pair_count = within_year_pair_accuracy(
        all_rows,
        require_equal_star=True,
    )
    star_rows = [{**row, "latent_score": float(row["official_star"])} for row in all_rows]
    star_pair, star_pair_count = within_year_pair_accuracy(star_rows)
    return {
        "matched_oni_placements_all_stars": len(all_rows),
        "official_star_distribution": dict(
            sorted(Counter(int(row["official_star"]) for row in all_rows).items())
        ),
        "full_curriculum_macro_year_spearman": macro_year_spearman(all_rows),
        "full_curriculum_score_pair_accuracy": all_pair,
        "full_curriculum_score_pair_count": all_pair_count,
        "full_curriculum_official_star_pair_accuracy": star_pair,
        "full_curriculum_official_star_pair_count": star_pair_count,
        "equal_star_score_pair_accuracy": equal_star_pair,
        "equal_star_pair_count": equal_star_pair_count,
        "below_ceiling_equal_star_score_pair_accuracy": lower_pair,
        "below_ceiling_equal_star_pair_count": lower_pair_count,
    }


def analyze_condition(
    matched: Sequence[Mapping[str, Any]],
    *,
    family: str,
    condition: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    primary = with_analysis_tier(primary_ten_star_panel(matched), high_only=True)
    expected_primary = int(config["expected_primary_matched_placements"])
    if len(primary) != expected_primary:
        raise DojoDataError(
            f"{family}/{condition} primary panel has {len(primary)} rows, "
            f"expected {expected_primary}"
        )
    year_count = len({int(row["year"]) for row in primary})
    if year_count != int(config["expected_primary_years"]):
        raise DojoDataError(f"{family}/{condition} primary years drifted")
    bootstrap_replicates = int(config["bootstrap_replicates"])
    permutation_replicates = int(config["permutation_replicates"])
    seed = int(config["random_seed"])
    spearman = estimate_with_uncertainty(
        primary,
        metric="macro_year_spearman",
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed,
    )
    pair = estimate_with_uncertainty(
        primary,
        metric="within_year_pair_accuracy",
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed + 1,
    )
    _, pair_count = within_year_pair_accuracy(primary)
    per_year = _primary_per_year(primary)
    if len(per_year) != year_count or not all(
        math.isfinite(float(row["spearman_rho"])) for row in per_year
    ):
        raise DojoDataError(f"{family}/{condition} has incomplete yearly estimates")
    summary = {
        "schema_version": "icassp2027_dojo_criterion_summary_v1",
        "family": family,
        "condition": condition,
        "primary_panel": "high_dan_official_10_oni",
        "placement_count": len(primary),
        "unique_chart_count": len({str(row["chart_id"]) for row in primary}),
        "year_count": year_count,
        "tier_count": len({int(row["analysis_tier"]) for row in primary}),
        "pair_count": pair_count,
        "macro_year_spearman": spearman.as_dict(),
        "within_year_pair_accuracy": pair.as_dict(),
        "controls": _control_summary(matched),
        "position_exploration": _position_summary(primary),
    }
    per_year_rows = [
        {
            "schema_version": "icassp2027_dojo_per_year_v1",
            "family": family,
            "condition": condition,
            **row,
        }
        for row in per_year
    ]
    leave_out_rows = [
        {
            "schema_version": "icassp2027_dojo_leave_one_year_out_v1",
            "family": family,
            "condition": condition,
            **row,
        }
        for row in leave_one_year_out(primary)
    ]
    return summary, per_year_rows, leave_out_rows


def _panel_identity(rows: Sequence[Mapping[str, Any]]) -> set[tuple[int, str, int, str]]:
    return {
        (int(row["year"]), str(row["dan"]), int(row["position"]), str(row["normalized_title"]))
        for row in primary_ten_star_panel(rows)
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    placements = read_jsonl(args.placements)
    validate_course_placements(placements)
    source_manifest = read_json(args.source_manifest)
    validate_source_manifest(source_manifest, args.placements)
    source_counts = require_expected_counts(placements, config)
    global_release = validate_global_release_manifest(args.global_release_manifest)
    sequence_release = validate_sequence_release_manifest(
        args.sequence_release_manifest
    )
    global_rows = read_jsonl(global_release.model_scores)
    sequence_rows = normalize_sequence_rows(
        read_jsonl(sequence_release.ensemble_scores)
    )

    conditions = (
        (
            "global",
            str(config["primary_global_condition"]),
            global_rows,
        ),
        (
            "sequence",
            str(config["replication_sequence_condition"]),
            sequence_rows,
        ),
    )
    all_matched: list[dict[str, Any]] = []
    all_exclusions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    per_year_rows: list[dict[str, Any]] = []
    leave_out_rows: list[dict[str, Any]] = []
    panels: list[set[tuple[int, str, int, str]]] = []
    for family, condition, score_rows in conditions:
        matched, exclusions = match_course_placements(
            placements,
            score_rows,
            scenario="native",
            condition=condition,
        )
        for row in matched:
            row["family"] = family
        for row in exclusions:
            row["family"] = family
            row["condition"] = condition
        summary, yearly, leave_out = analyze_condition(
            matched,
            family=family,
            condition=condition,
            config=config,
        )
        all_matched.extend(matched)
        all_exclusions.extend(exclusions)
        summaries.append(summary)
        per_year_rows.extend(yearly)
        leave_out_rows.extend(leave_out)
        panels.append(_panel_identity(matched))
    if panels[0] != panels[1]:
        raise DojoDataError("global and sequence criteria use different primary panels")

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "matched_placements": args.output / "matched_placements.jsonl",
        "exclusions": args.output / "exclusions.jsonl",
        "criterion_summary": args.output / "criterion_summary.jsonl",
        "per_year_metrics": args.output / "per_year_metrics.jsonl",
        "leave_one_year_out": args.output / "leave_one_year_out.jsonl",
    }
    atomic_write_jsonl(output_paths["matched_placements"], all_matched)
    atomic_write_jsonl(output_paths["exclusions"], all_exclusions)
    atomic_write_jsonl(output_paths["criterion_summary"], summaries)
    atomic_write_jsonl(output_paths["per_year_metrics"], per_year_rows)
    atomic_write_jsonl(output_paths["leave_one_year_out"], leave_out_rows)
    story = {
        "schema_version": "icassp2027_dojo_story_v1",
        "source_counts": source_counts,
        "primary_panel_identity_count": len(panels[0]),
        "conditions": {
            f"{row['family']}:{row['condition']}": row for row in summaries
        },
    }
    story_path = args.output / "primary_story_summary.json"
    atomic_write_json(story_path, story)
    output_paths["primary_story_summary"] = story_path
    manifest = {
        "schema_version": "icassp2027_dojo_analysis_manifest_v1",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "inputs": {
            "placements": {
                "path": args.placements.as_posix(),
                "sha256": file_sha256(args.placements),
            },
            "source_manifest": {
                "path": args.source_manifest.as_posix(),
                "sha256": file_sha256(args.source_manifest),
            },
            "global_scores": {
                "path": str(
                    global_release.manifest["files"]["model_scores"]["path"]
                ),
                "sha256": file_sha256(global_release.model_scores),
            },
            "sequence_scores": {
                "path": str(
                    sequence_release.manifest["files"]["ensemble_scores"]["path"]
                ),
                "sha256": file_sha256(sequence_release.ensemble_scores),
            },
        },
        "release_manifests": {
            "global": {
                "path": args.global_release_manifest.as_posix(),
                "sha256": file_sha256(args.global_release_manifest),
            },
            "sequence": {
                "path": args.sequence_release_manifest.as_posix(),
                "sha256": file_sha256(args.sequence_release_manifest),
            },
        },
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "permutation_replicates": int(config["permutation_replicates"]),
        "primary_panel_identity_count": len(panels[0]),
        "condition_count": len(summaries),
        "output_files": {
            name: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for name, path in output_paths.items()
        },
    }
    atomic_write_json(args.output / "analysis_manifest.json", manifest)
    print(json.dumps(story, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
