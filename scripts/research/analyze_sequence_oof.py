#!/usr/bin/env python
"""Bootstrap sequence hidden-order recovery from frozen OOF evidence."""

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
    paired_song_group_bootstrap_difference,
    strict_pair_accuracy,
    top_code_targets,
)
from TaikoChartEstimator.research.release import (
    SEQUENCE_ENSEMBLE_ROW_COUNT,
    validate_global_release_manifest,
    validate_sequence_release_manifest,
)

EXPECTED_SEQUENCE_KEYS = {
    ("cap8", "seq_ordinary_huber"),
    ("cap8", "seq_topcoded_huber"),
    ("cap8", "seq_ordinal_logit"),
    ("cap7", "seq_ordinary_huber"),
    ("cap7", "seq_ordinal_logit"),
    ("cap9", "seq_ordinary_huber"),
    ("cap9", "seq_ordinal_logit"),
    ("native", "seq_ordinal_logit"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/configs/primary.json"),
    )
    parser.add_argument(
        "--sequence-release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_sequence_oof/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--global-release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/reports"),
    )
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile_summary(
    values: Sequence[float],
    *,
    requested_replicates: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if len(finite) == 0:
        return {
            "bootstrap_mean": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "valid_bootstrap_replicates": 0,
            "invalid_bootstrap_replicates": requested_replicates,
        }
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return {
        "bootstrap_mean": float(finite.mean()),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "valid_bootstrap_replicates": int(len(finite)),
        "invalid_bootstrap_replicates": requested_replicates - int(len(finite)),
    }


def title_bootstrap_rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    hidden = [row for row in rows if float(row["evaluation_original_label"]) >= cap]
    if not hidden:
        raise ValueError("hidden-tail bootstrap received an empty panel")
    by_title: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in hidden:
        by_title[str(row["normalized_title"])].append(row)
    titles = sorted(by_title)
    if len(titles) < 2:
        raise ValueError("hidden-tail bootstrap requires at least two titles")
    original = np.asarray(
        [float(row["evaluation_original_label"]) for row in hidden],
        dtype=np.float64,
    )
    latent = np.asarray(
        [float(row["latent_score"]) for row in hidden], dtype=np.float64
    )
    point_spearman = float(spearmanr(original, latent).statistic)
    point_kendall = float(kendalltau(original, latent, variant="b").statistic)
    point_pair = strict_pair_accuracy(original, latent)
    generator = np.random.default_rng(seed)
    spearman_values: list[float] = []
    kendall_values: list[float] = []
    pair_values: list[float] = []
    for _ in range(replicates):
        sampled_rows: list[Mapping[str, Any]] = []
        for index in generator.integers(0, len(titles), size=len(titles)):
            sampled_rows.extend(by_title[titles[int(index)]])
        sampled_original = np.asarray(
            [float(row["evaluation_original_label"]) for row in sampled_rows]
        )
        sampled_latent = np.asarray(
            [float(row["latent_score"]) for row in sampled_rows]
        )
        if (
            len(set(sampled_original.tolist())) < 2
            or len(set(sampled_latent.tolist())) < 2
        ):
            continue
        rho = float(spearmanr(sampled_original, sampled_latent).statistic)
        tau = float(kendalltau(sampled_original, sampled_latent, variant="b").statistic)
        pair = strict_pair_accuracy(sampled_original, sampled_latent)
        if math.isfinite(rho):
            spearman_values.append(rho)
        if math.isfinite(tau):
            kendall_values.append(tau)
        if pair is not None and math.isfinite(pair):
            pair_values.append(pair)
    return {
        "hidden_tail_count": len(hidden),
        "title_clusters": len(titles),
        "bootstrap_method": "normalized_title_cluster_resampling",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_level": 0.95,
        "spearman_rho": point_spearman,
        "spearman": percentile_summary(
            spearman_values, requested_replicates=replicates
        ),
        "kendall_tau_b": point_kendall,
        "kendall": percentile_summary(kendall_values, requested_replicates=replicates),
        "strict_pair_accuracy": point_pair,
        "strict_pair": percentile_summary(pair_values, requested_replicates=replicates),
    }


def hidden_panel(rows: Sequence[Mapping[str, Any]], *, cap: int):
    original = [float(row["evaluation_original_label"]) for row in rows]
    top_coded = top_code_targets(
        original,
        [3] * len(rows),
        cap=cap,
        selected_course_ids={3},
    )
    return build_hidden_tail_panel(
        top_coded,
        [str(row["normalized_title"]) for row in rows],
    )


def paired_contrast(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    first_by_chart = {str(row["chart_id"]): row for row in first}
    second_by_chart = {str(row["chart_id"]): row for row in second}
    if len(first_by_chart) != len(first):
        raise ValueError("first sequence contrast has duplicate chart IDs")
    if len(second_by_chart) != len(second):
        raise ValueError("second sequence contrast has duplicate chart IDs")
    if set(first_by_chart) != set(second_by_chart):
        raise ValueError("paired sequence contrast has unequal chart coverage")
    chart_ids = sorted(first_by_chart)
    ordered_first = [first_by_chart[chart_id] for chart_id in chart_ids]
    ordered_second = [second_by_chart[chart_id] for chart_id in chart_ids]
    for first_row, second_row in zip(ordered_first, ordered_second):
        if (
            float(first_row["evaluation_original_label"])
            != float(second_row["evaluation_original_label"])
            or str(first_row["normalized_title"])
            != str(second_row["normalized_title"])
        ):
            raise ValueError("paired sequence contrast rows disagree on target or title")
    panel = hidden_panel(ordered_first, cap=cap)
    return paired_song_group_bootstrap_difference(
        panel,
        [float(row["latent_score"]) for row in ordered_first],
        [float(row["latent_score"]) for row in ordered_second],
        replicates=replicates,
        seed=seed,
    )


def paired_representation_contrast(
    sequence_rows: Sequence[Mapping[str, Any]],
    global_rows: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Compare sequence and global scores on an identical OOF chart panel."""

    sequence_by_chart = {str(row["chart_id"]): row for row in sequence_rows}
    global_by_chart = {str(row["chart_id"]): row for row in global_rows}
    if len(sequence_by_chart) != len(sequence_rows):
        raise ValueError("sequence representation contrast has duplicate chart IDs")
    if len(global_by_chart) != len(global_rows):
        raise ValueError("global representation contrast has duplicate chart IDs")
    if set(sequence_by_chart) != set(global_by_chart):
        raise ValueError("representation contrast has unequal chart coverage")
    chart_ids = sorted(sequence_by_chart)
    ordered_sequence = [sequence_by_chart[chart_id] for chart_id in chart_ids]
    ordered_global = [global_by_chart[chart_id] for chart_id in chart_ids]
    for sequence_row, global_row in zip(ordered_sequence, ordered_global):
        if (
            float(sequence_row["evaluation_original_label"])
            != float(global_row["original_star"])
            or str(sequence_row["normalized_title"])
            != str(global_row["normalized_title"])
        ):
            raise ValueError("representation contrast rows disagree on target or title")
    panel = hidden_panel(ordered_sequence, cap=cap)
    return paired_song_group_bootstrap_difference(
        panel,
        [float(row["latent_score"]) for row in ordered_sequence],
        [float(row["latent_score"]) for row in ordered_global],
        replicates=replicates,
        seed=seed,
    )



def main() -> None:
    args = parse_args()
    config = load_json_object(args.config)
    if (
        config.get("experiment_id") != "icassp2027_sequence_oof"
        or config.get("primary_evidence") is not True
        or int(config.get("outer_folds", -1)) != 5
        or config.get("scenarios")
        != {
            "cap7": {"cap": 7},
            "cap8": {"cap": 8},
            "cap9": {"cap": 9},
            "native": {"cap": 10},
        }
        or {
            (str(row.get("scenario")), str(row.get("condition")))
            for row in config.get("matrix", [])
            if isinstance(row, Mapping)
        }
        != EXPECTED_SEQUENCE_KEYS
    ):
        raise ValueError("sequence analysis config differs from the frozen matrix")
    replicates = int(config["bootstrap_replicates"])
    seed = int(config["bootstrap_seed"])
    if replicates != 10_000 or seed != 2027:
        raise ValueError("sequence analysis bootstrap contract drifted")
    sequence_release = validate_sequence_release_manifest(
        args.sequence_release_manifest
    )
    global_release = validate_global_release_manifest(args.global_release_manifest)
    ensembles = load_jsonl(sequence_release.ensemble_scores)
    if len(ensembles) != SEQUENCE_ENSEMBLE_ROW_COUNT:
        raise ValueError("sequence release has incomplete ensemble coverage")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ensembles:
        grouped[(str(row["scenario"]), str(row["condition"]))].append(row)
    if set(grouped) != EXPECTED_SEQUENCE_KEYS or any(
        len(rows) != 1_023 for rows in grouped.values()
    ):
        raise ValueError("sequence release matrix has incomplete chart coverage")
    ensemble_keys = [
        (str(row["scenario"]), str(row["condition"]), str(row["chart_id"]))
        for row in ensembles
    ]
    if len(ensemble_keys) != len(set(ensemble_keys)):
        raise ValueError("sequence release ensemble grain is not unique")

    hidden_metrics: list[dict[str, Any]] = []
    for (scenario, condition), rows in sorted(grouped.items()):
        if scenario == "native":
            continue
        cap = int(config["scenarios"][scenario]["cap"])
        hidden_metrics.append(
            {
                "schema_version": "sequence_hidden_tail_bootstrap",
                "scenario": scenario,
                "condition": condition,
                "cap": cap,
                **title_bootstrap_rank_metrics(
                    rows,
                    cap=cap,
                    replicates=replicates,
                    seed=seed,
                ),
            }
        )

    contrast_specs = (
        ("cap8", "seq_topcoded_huber", "seq_ordinary_huber"),
        ("cap8", "seq_ordinal_logit", "seq_ordinary_huber"),
        ("cap7", "seq_ordinal_logit", "seq_ordinary_huber"),
        ("cap9", "seq_ordinal_logit", "seq_ordinary_huber"),
    )
    contrasts: list[dict[str, Any]] = []
    for scenario, first_condition, second_condition in contrast_specs:
        cap = int(config["scenarios"][scenario]["cap"])
        contrasts.append(
            {
                "schema_version": "sequence_hidden_tail_contrast",
                "scenario": scenario,
                "cap": cap,
                "first_condition": first_condition,
                "second_condition": second_condition,
                "contrast_direction": "first_minus_second",
                **paired_contrast(
                    grouped[(scenario, first_condition)],
                    grouped[(scenario, second_condition)],
                    cap=cap,
                    replicates=replicates,
                    seed=seed,
                ),
            }
        )

    global_scores = load_jsonl(global_release.model_scores)
    global_ordinal_by_scenario = {
        scenario: [
            row
            for row in global_scores
            if row.get("scenario") == scenario
            and row.get("condition") == "ordinal_logit"
        ]
        for scenario in ("cap7", "cap8", "cap9")
    }
    representation_contrasts: list[dict[str, Any]] = []
    for scenario in ("cap7", "cap8", "cap9"):
        cap = int(config["scenarios"][scenario]["cap"])
        representation_contrasts.append(
            {
                "schema_version": "sequence_representation_contrast",
                "scenario": scenario,
                "cap": cap,
                "first_condition": "seq_ordinal_logit",
                "second_condition": "ordinal_logit",
                "contrast_direction": "sequence_minus_global",
                **paired_representation_contrast(
                    grouped[(scenario, "seq_ordinal_logit")],
                    global_ordinal_by_scenario[scenario],
                    cap=cap,
                    replicates=replicates,
                    seed=seed,
                ),
            }
        )

    output_paths = {
        "hidden_tail_metrics": args.output / "hidden_tail_metrics.jsonl",
        "hidden_tail_contrasts": args.output / "hidden_tail_contrasts.jsonl",
        "representation_contrasts": args.output / "representation_contrasts.jsonl",
    }
    atomic_write_jsonl(output_paths["hidden_tail_metrics"], hidden_metrics)
    atomic_write_jsonl(output_paths["hidden_tail_contrasts"], contrasts)
    atomic_write_jsonl(
        output_paths["representation_contrasts"],
        representation_contrasts,
    )
    atomic_write_json(
        args.output / "analysis_manifest.json",
        {
            "schema_version": "sequence_analysis_manifest",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "hidden_metric_rows": len(hidden_metrics),
            "contrast_rows": len(contrasts),
            "representation_contrast_rows": len(representation_contrasts),
            "output_files": {
                name: {
                    "path": path.as_posix(),
                    "sha256": file_sha256(path),
                }
                for name, path in sorted(output_paths.items())
            },
            "input_files": {
                "config": {
                    "path": args.config.as_posix(),
                    "sha256": file_sha256(args.config),
                },
                "ensemble_scores": {
                    "path": str(
                        sequence_release.manifest["files"]["ensemble_scores"][
                            "path"
                        ]
                    ),
                    "sha256": file_sha256(sequence_release.ensemble_scores),
                },
                "global_scores": {
                    "path": str(
                        global_release.manifest["files"]["model_scores"]["path"]
                    ),
                    "sha256": file_sha256(global_release.model_scores),
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
        },
    )
    print(
        json.dumps(
            {
                "hidden_metric_rows": len(hidden_metrics),
                "contrast_rows": len(contrasts),
                "representation_contrast_rows": len(representation_contrasts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
