#!/usr/bin/env python
"""Build canonical sequence OOF tables from matching append-only evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from scripts.research.run_sequence_oof_matrix import (
    EXPECTED_FOLD_RUNS,
    matching_success,
    planned_fold_runs,
)
from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    source_files_snapshot,
)
from TaikoChartEstimator.research.oof import (
    build_hidden_tail_panel,
    hidden_tail_rank_metrics,
    top_code_targets,
)
from TaikoChartEstimator.research.sequence import SEQUENCE_TRAINING_SOURCE_PATHS
from TaikoChartEstimator.research.sequence_data import load_sequence_chart_records

EXPERIMENT_ID = "icassp2027_sequence_oof"
CANONICAL_OUTPUT_FILES = (
    "fold_runs.jsonl",
    "model_scores.jsonl",
    "ensemble_scores.jsonl",
    "sequence_metrics.jsonl",
    "latest_successful_runs.json",
)
EXPECTED_SEED_OOF_ROWS = 16_368
EXPECTED_ENSEMBLE_ROWS = 8_184
EXPECTED_METRIC_ROWS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/configs/primary.json"),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/SPEC.md"),
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/canonical/fold_assignments.jsonl"
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("experiments/icassp2027_core/data/dataset_manifest.json"),
    )
    parser.add_argument(
        "--dataset-snapshot",
        type=Path,
        default=Path("experiments/icassp2027_core/remote/minimal_dataset_snapshot"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/raw/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/canonical"),
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


def validate_canonical_manifest(
    root: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Require the complete frozen OOF inventory and byte-current outputs."""

    expected_counts = {
        "fold_run_count": EXPECTED_FOLD_RUNS,
        "seed_oof_row_count": EXPECTED_SEED_OOF_ROWS,
        "ensemble_row_count": EXPECTED_ENSEMBLE_ROWS,
        "metric_row_count": EXPECTED_METRIC_ROWS,
    }
    if (
        manifest.get("schema_version") != "sequence_oof_canonical_manifest"
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or any(manifest.get(key) != value for key, value in expected_counts.items())
    ):
        raise ValueError("sequence canonical manifest violates the frozen counts")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(CANONICAL_OUTPUT_FILES):
        raise ValueError("sequence canonical manifest has an unexpected file set")
    for name in CANONICAL_OUTPUT_FILES:
        path = root / name
        expected = files.get(name)
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or len(expected) != 64
            or file_sha256(path) != expected
        ):
            raise ValueError(f"sequence canonical output is stale: {name}")


def validate_primary_cuda_manifest(manifest: Mapping[str, Any]) -> tuple[str, int]:
    parameters = manifest.get("parameters", {})
    outputs = manifest.get("outputs", {})
    device = str(parameters.get("device", ""))
    peak_memory = outputs.get("peak_cuda_memory_allocated_bytes")
    if not device.startswith("cuda"):
        raise ValueError("primary sequence evidence must run on CUDA")
    if not isinstance(peak_memory, int) or peak_memory <= 0:
        raise ValueError("primary sequence evidence must record positive CUDA memory")
    return device, peak_memory


def validate_fold_supporting_evidence(
    attempt: Path,
    manifest: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate selection, refit, and standardization semantics for one fold."""

    scenario = str(planned["scenario"])
    condition = str(planned["condition"])
    seed = int(planned["seed"])
    outer_fold = int(planned["outer_fold"])
    cap = int(config["scenarios"][scenario]["cap"])
    parameters = manifest.get("parameters")
    outputs = manifest.get("outputs")
    if (
        manifest.get("run_schema_version") != "raw_run_v2"
        or manifest.get("status") != "success"
        or not isinstance(parameters, Mapping)
        or not isinstance(outputs, Mapping)
    ):
        raise ValueError("successful fold run has an invalid manifest schema")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(
        f"{condition}/seed-{seed}/attempt-"
    ):
        raise ValueError("successful fold run has an invalid run ID")
    try:
        completed_at = datetime.fromisoformat(
            str(manifest.get("completed_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("successful fold run has an invalid completion time") from exc
    if completed_at.tzinfo is None:
        raise ValueError("successful fold completion time has no UTC offset")
    partitions = parameters.get("partition_sizes")
    if not isinstance(partitions, Mapping) or any(
        not isinstance(partitions.get(name), int)
        or isinstance(partitions.get(name), bool)
        or int(partitions[name]) <= 0
        for name in ("train", "validation", "test", "refit")
    ):
        raise ValueError("successful fold run has invalid partition sizes")
    if (
        int(partitions["train"])
        + int(partitions["validation"])
        + int(partitions["test"])
        != 1_023
        or int(partitions["refit"])
        != int(partitions["train"]) + int(partitions["validation"])
    ):
        raise ValueError("successful fold run does not use the full OOF panel")
    if (
        parameters.get("primary_evidence") is not True
        or int(parameters.get("cap", -1)) != cap
        or parameters.get("architecture") != config["architecture"]
        or parameters.get("training") != config["training"]
        or parameters.get("selection_label") != "observed_label_mae_only"
        or not str(parameters.get("device", "")).startswith("cuda")
        or int(parameters.get("num_workers", -1)) < 0
    ):
        raise ValueError("successful fold run violates the frozen parameter contract")

    checkpoint = load_json_object(attempt / "checkpoint_selection.json")
    best_epoch = checkpoint.get("best_epoch")
    best_mae = checkpoint.get("best_validation_observed_label_mae")
    if (
        checkpoint.get("schema_version") != "sequence_checkpoint_selection"
        or checkpoint.get("selection_metric")
        != "validation_observed_label_mae"
        or checkpoint.get("original_label_used_for_selection") is not False
        or not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 1 <= best_epoch <= int(config["training"]["epochs"])
        or checkpoint.get("refit_epochs") != best_epoch
        or not isinstance(best_mae, (int, float))
        or isinstance(best_mae, bool)
        or not math.isfinite(float(best_mae))
        or float(best_mae) < 0.0
        or outputs.get("best_epoch") != best_epoch
        or outputs.get("best_validation_observed_label_mae") != best_mae
    ):
        raise ValueError("successful fold run has invalid checkpoint selection")

    selection_history = load_jsonl(attempt / "selection_history.jsonl")
    refit_history = load_jsonl(attempt / "refit_history.jsonl")
    if not best_epoch <= len(selection_history) <= int(config["training"]["epochs"]):
        raise ValueError("selection history length is incompatible with best epoch")
    if len(refit_history) != best_epoch:
        raise ValueError("refit history does not reproduce the selected epoch count")
    for history, phase in (
        (selection_history, "selection"),
        (refit_history, "refit"),
    ):
        if [row.get("epoch") for row in history] != list(
            range(1, len(history) + 1)
        ):
            raise ValueError(f"{phase} history epochs are not contiguous")
        for row in history:
            required_finite = (
                "learning_rate",
                "epoch_wall_seconds",
                "training_observation_loss",
            )
            if row.get("phase") != phase or any(
                not isinstance(row.get(name), (int, float))
                or isinstance(row.get(name), bool)
                or not math.isfinite(float(row[name]))
                for name in required_finite
            ):
                raise ValueError(f"{phase} history contains invalid training evidence")
            if float(row["epoch_wall_seconds"]) <= 0.0:
                raise ValueError(f"{phase} history has a non-positive epoch time")
            if (
                float(row["learning_rate"]) < 0.0
                or float(row["training_observation_loss"]) < 0.0
            ):
                raise ValueError(f"{phase} history has a negative optimizer value")
            if phase == "selection" and (
                not isinstance(row.get("validation_observed_label_mae"), (int, float))
                or isinstance(row.get("validation_observed_label_mae"), bool)
                or not math.isfinite(float(row["validation_observed_label_mae"]))
                or not isinstance(row.get("checkpoint_improved"), bool)
                or not isinstance(row.get("stale_epochs"), int)
                or isinstance(row.get("stale_epochs"), bool)
            ):
                raise ValueError("selection history has invalid observed-label evidence")

    reproduced_best = math.inf
    reproduced_best_epoch = -1
    reproduced_stale = 0
    minimum_delta = float(config["training"]["early_stopping_min_delta"])
    patience = int(config["training"]["early_stopping_patience"])
    for row in selection_history:
        validation_mae = float(row["validation_observed_label_mae"])
        improved = validation_mae < reproduced_best - minimum_delta
        if improved:
            reproduced_best = validation_mae
            reproduced_best_epoch = int(row["epoch"])
            reproduced_stale = 0
        else:
            reproduced_stale += 1
        if (
            row["checkpoint_improved"] is not improved
            or int(row["stale_epochs"]) != reproduced_stale
        ):
            raise ValueError("selection history does not reproduce checkpoint logic")
    if (
        reproduced_best_epoch != best_epoch
        or reproduced_best != float(best_mae)
        or (
            len(selection_history) < int(config["training"]["epochs"])
            and reproduced_stale < patience
        )
    ):
        raise ValueError("checkpoint selection is not reproduced by its history")

    standardization = load_json_object(attempt / "score_standardization.json")
    if set(standardization) != {
        "source_partition",
        "mean",
        "standard_deviation",
    } or standardization.get("source_partition") != "outer_refit":
        raise ValueError("score standardization did not use the outer-refit panel")
    for name in ("mean", "standard_deviation"):
        value = standardization.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("score standardization contains a non-finite value")
    if float(standardization["standard_deviation"]) <= 0.0:
        raise ValueError("score standardization has a non-positive scale")

    model_record = load_json_object(attempt / "model_record.json")
    thresholds = model_record.get("ordered_thresholds")
    if (
        set(model_record)
        != {
            "condition",
            "maximum_observed_label",
            "architecture",
            "ordered_thresholds",
            "selection_parameter_count",
        }
        or model_record.get("condition") != condition
        or int(model_record.get("maximum_observed_label", -1)) != cap
        or model_record.get("architecture") != config["architecture"]
        or not isinstance(model_record.get("selection_parameter_count"), int)
        or isinstance(model_record.get("selection_parameter_count"), bool)
        or int(model_record["selection_parameter_count"]) <= 0
    ):
        raise ValueError("model record violates the frozen fold contract")
    if condition == "seq_ordinal_logit":
        if (
            not isinstance(thresholds, list)
            or len(thresholds) != cap - 1
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in thresholds
            )
            or any(
                float(first) >= float(second)
                for first, second in zip(thresholds, thresholds[1:])
            )
        ):
            raise ValueError("ordinal model record has invalid ordered thresholds")
    elif thresholds is not None:
        raise ValueError("non-ordinal model unexpectedly records ordered thresholds")

    fold_wall_seconds = outputs.get("fold_wall_seconds")
    if (
        not isinstance(fold_wall_seconds, (int, float))
        or isinstance(fold_wall_seconds, bool)
        or not math.isfinite(float(fold_wall_seconds))
        or float(fold_wall_seconds) <= 0.0
    ):
        raise ValueError("successful fold run has invalid wall-clock evidence")
    return {
        "run_id": run_id,
        "completed_at_utc": completed_at.isoformat(),
        "best_epoch": best_epoch,
        "fold_wall_seconds": float(fold_wall_seconds),
    }


def source_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "config_sha256": file_sha256(args.config),
        "spec_sha256": file_sha256(args.spec),
        "fold_assignments_sha256": file_sha256(args.fold_assignments),
        "dataset_manifest_sha256": file_sha256(args.dataset_manifest),
        "dataset_snapshot_manifest_sha256": file_sha256(
            args.dataset_snapshot / "RESEARCH_DATASET_SNAPSHOT.json"
        ),
        "training_source_sha256": source_files_snapshot(SEQUENCE_TRAINING_SOURCE_PATHS)[
            "aggregate_sha256"
        ],
    }


def seed_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(row["scenario"]), str(row["condition"]), int(row["seed"])


def condition_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["scenario"]), str(row["condition"])


def validate_fold_predictions(
    rows: Sequence[Mapping[str, Any]],
    planned: Mapping[str, Any],
    *,
    record_by_chart: Mapping[str, Any],
    expected_source_hashes: Mapping[str, str],
    expected_timing: Mapping[str, Any] | None = None,
) -> None:
    expected_charts = {
        record.chart_id
        for record in record_by_chart.values()
        if record.outer_test_fold == int(planned["outer_fold"])
    }
    observed_charts = [str(row.get("chart_id", "")) for row in rows]
    if len(observed_charts) != len(set(observed_charts)):
        raise ValueError(f"duplicate fold prediction for {seed_key(planned)}")
    if set(observed_charts) != expected_charts:
        raise ValueError(
            "fold predictions do not equal the frozen outer test partition: "
            f"planned={planned}"
        )
    for row in rows:
        if (
            row.get("experiment_id") != EXPERIMENT_ID
            or row.get("scenario") != planned["scenario"]
            or row.get("condition") != planned["condition"]
            or int(row.get("seed", -1)) != int(planned["seed"])
            or int(row.get("outer_fold", -1)) != int(planned["outer_fold"])
        ):
            raise ValueError("prediction row disagrees with its planned fold key")
        record = record_by_chart[str(row["chart_id"])]
        if str(row.get("song_id", "")) != record.song_id:
            raise ValueError("prediction row disagrees with frozen song ID")
        if str(row.get("normalized_title", "")) != record.normalized_title:
            raise ValueError("prediction row disagrees with frozen normalized title")
        if int(row.get("outer_fold", -1)) != record.outer_test_fold:
            raise ValueError("prediction row disagrees with frozen outer fold")
        if float(row["observed_label"]) != float(record.observed_label):
            raise ValueError("prediction row disagrees with frozen observed label")
        if int(row["evaluation_original_label"]) != record.original_label:
            raise ValueError("prediction row disagrees with evaluation-only label")
        for field in (
            "raw_latent_score",
            "latent_score",
            "expected_observed_label",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"canonical {field} is non-finite")
        if row.get("source_hashes") != dict(expected_source_hashes):
            raise ValueError("prediction row source hashes disagree with frozen inputs")
        timing = row.get("timing")
        if (
            not isinstance(timing, Mapping)
            or not math.isfinite(float(timing.get("fold_wall_seconds", math.nan)))
            or float(timing["fold_wall_seconds"]) <= 0.0
            or int(timing.get("best_epoch", 0)) <= 0
        ):
            raise ValueError("prediction row has invalid timing evidence")
        if expected_timing is not None and (
            float(timing["fold_wall_seconds"])
            != float(expected_timing["fold_wall_seconds"])
            or int(timing["best_epoch"]) != int(expected_timing["best_epoch"])
        ):
            raise ValueError("prediction row timing disagrees with its run manifest")


def metric_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
    cap: int,
    aggregation: str,
    seed: int | None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["chart_id"]))
    original = np.asarray(
        [float(row["evaluation_original_label"]) for row in ordered],
        dtype=np.float64,
    )
    observed = np.asarray(
        [float(row["observed_label"]) for row in ordered], dtype=np.float64
    )
    expected = np.asarray(
        [float(row["expected_observed_label"]) for row in ordered],
        dtype=np.float64,
    )
    latent = np.asarray(
        [float(row["latent_score"]) for row in ordered], dtype=np.float64
    )
    groups = [str(row["normalized_title"]) for row in ordered]
    top_coded = top_code_targets(
        original,
        np.full(len(original), 3, dtype=np.int64),
        cap=cap,
        selected_course_ids={3},
    )
    panel = build_hidden_tail_panel(top_coded, groups)
    hidden = hidden_tail_rank_metrics(panel, latent)
    catalog_spearman = float(spearmanr(original, latent).statistic)
    if not math.isfinite(catalog_spearman):
        raise ValueError("catalog Spearman is non-finite")
    return {
        "schema_version": "sequence_oof_metrics",
        "scenario": scenario,
        "condition": condition,
        "cap": cap,
        "aggregation": aggregation,
        "seed": seed,
        "chart_count": len(ordered),
        "title_group_count": len(set(groups)),
        "observed_label_mae": float(
            np.mean(np.abs(np.clip(expected, 1.0, float(cap)) - observed))
        ),
        "catalog_spearman_against_original": catalog_spearman,
        **hidden,
    }


def summarize_optional_metric(
    values: Sequence[float | None],
) -> tuple[float | None, float | None]:
    """Summarize a metric while preserving a genuinely undefined condition."""

    if all(value is None for value in values):
        return None, None
    if any(value is None for value in values):
        raise ValueError("metric is undefined for only a subset of ensemble seeds")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("metric summary contains a non-finite value")
    return float(np.mean(numeric)), float(np.std(numeric, ddof=0))


def build_ensemble(
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    scenario: str,
    condition: str,
) -> list[dict[str, Any]]:
    by_seed_chart = {
        seed: {str(row["chart_id"]): row for row in rows}
        for seed, rows in seed_rows.items()
    }
    chart_sets = {frozenset(rows) for rows in by_seed_chart.values()}
    if len(chart_sets) != 1:
        raise ValueError(
            f"ensemble seeds have different coverage for {scenario}/{condition}"
        )
    charts = sorted(next(iter(chart_sets)))
    output: list[dict[str, Any]] = []
    for chart_id in charts:
        members = [by_seed_chart[seed][chart_id] for seed in sorted(by_seed_chart)]
        invariant_fields = (
            "song_id",
            "normalized_title",
            "observed_label",
            "evaluation_original_label",
            "outer_fold",
        )
        for field in invariant_fields:
            values = {json.dumps(row[field], sort_keys=True) for row in members}
            if len(values) != 1:
                raise ValueError(f"ensemble members disagree on {field} for {chart_id}")
        output.append(
            {
                "schema_version": "sequence_oof_ensemble_prediction",
                "experiment_id": EXPERIMENT_ID,
                "scenario": scenario,
                "condition": condition,
                "chart_id": chart_id,
                "song_id": members[0]["song_id"],
                "normalized_title": members[0]["normalized_title"],
                "outer_fold": members[0]["outer_fold"],
                "seeds": sorted(by_seed_chart),
                "observed_label": members[0]["observed_label"],
                "evaluation_original_label": members[0]["evaluation_original_label"],
                "latent_score": float(
                    np.mean([float(row["latent_score"]) for row in members])
                ),
                "expected_observed_label": float(
                    np.mean([float(row["expected_observed_label"]) for row in members])
                ),
                "latent_score_seed_sd": float(
                    np.std([float(row["latent_score"]) for row in members], ddof=0)
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    config = load_json_object(args.config)
    planned = planned_fold_runs(config)
    if len(planned) != EXPECTED_FOLD_RUNS:
        raise RuntimeError("planned sequence matrix changed")
    hashes = source_hashes(args)
    records_by_scenario = {
        scenario: load_sequence_chart_records(
            args.fold_assignments,
            cap=int(value["cap"]),
        )
        for scenario, value in config["scenarios"].items()
    }
    attempts: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []
    rows_by_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for planned_row in planned:
        attempt = matching_success(
            args.runs,
            planned_row,
            source_hashes=hashes,
        )
        if attempt is None:
            raise RuntimeError(f"missing matching successful fold-run: {planned_row}")
        predictions = load_jsonl(attempt / "test_predictions.jsonl")
        record_by_chart = {
            record.chart_id: record
            for record in records_by_scenario[str(planned_row["scenario"])]
        }
        manifest = load_json_object(attempt / "run_manifest.json")
        support = validate_fold_supporting_evidence(
            attempt,
            manifest,
            planned_row,
            config=config,
        )
        validate_fold_predictions(
            predictions,
            planned_row,
            record_by_chart=record_by_chart,
            expected_source_hashes=hashes,
            expected_timing=support,
        )
        device, peak_cuda_memory = validate_primary_cuda_manifest(manifest)
        attempts.append(
            {
                "scenario": planned_row["scenario"],
                "condition": planned_row["condition"],
                "seed": planned_row["seed"],
                "outer_fold": planned_row["outer_fold"],
                "attempt": str(attempt),
                "run_id": support["run_id"],
                "completed_at_utc": support["completed_at_utc"],
                "device": device,
                "fold_wall_seconds": support["fold_wall_seconds"],
                "peak_cuda_memory_allocated_bytes": peak_cuda_memory,
            }
        )
        key = seed_key(planned_row)
        rows_by_seed[key].extend(dict(row) for row in predictions)

    run_ids = [str(row["run_id"]) for row in attempts]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("canonical fold inventory contains duplicate run IDs")

    metrics: list[dict[str, Any]] = []
    for (scenario, condition, seed), rows in sorted(rows_by_seed.items()):
        records = records_by_scenario[scenario]
        expected_charts = {record.chart_id for record in records}
        chart_ids = [str(row["chart_id"]) for row in rows]
        if len(chart_ids) != len(expected_charts) or set(chart_ids) != expected_charts:
            raise RuntimeError(
                f"incomplete seed OOF coverage for {scenario}/{condition}/{seed}"
            )
        if len(chart_ids) != len(set(chart_ids)):
            raise RuntimeError(
                f"duplicate seed OOF chart for {scenario}/{condition}/{seed}"
            )
        sorted_rows = sorted(rows, key=lambda row: str(row["chart_id"]))
        all_seed_rows.extend(sorted_rows)
        metrics.append(
            metric_row(
                sorted_rows,
                scenario=scenario,
                condition=condition,
                cap=int(config["scenarios"][scenario]["cap"]),
                aggregation="seed",
                seed=seed,
            )
        )

    grouped_seed_rows: dict[tuple[str, str], dict[int, Sequence[Mapping[str, Any]]]] = (
        defaultdict(dict)
    )
    for (scenario, condition, seed), rows in rows_by_seed.items():
        grouped_seed_rows[(scenario, condition)][seed] = rows
    ensembles: list[dict[str, Any]] = []
    for (scenario, condition), seed_mapping in sorted(grouped_seed_rows.items()):
        ensemble = build_ensemble(
            seed_mapping,
            scenario=scenario,
            condition=condition,
        )
        ensembles.extend(ensemble)
        ensemble_metrics = metric_row(
            ensemble,
            scenario=scenario,
            condition=condition,
            cap=int(config["scenarios"][scenario]["cap"]),
            aggregation="ensemble",
            seed=None,
        )
        seed_metric_rows = [
            row
            for row in metrics
            if row["scenario"] == scenario
            and row["condition"] == condition
            and row["aggregation"] == "seed"
        ]
        ensemble_metrics["seed_count"] = len(seed_metric_rows)
        hidden_mean, hidden_sd = summarize_optional_metric(
            [row["spearman_rho"] for row in seed_metric_rows]
        )
        ensemble_metrics["hidden_tail_spearman_seed_mean"] = hidden_mean
        ensemble_metrics["hidden_tail_spearman_seed_sd"] = hidden_sd
        ensemble_metrics["catalog_spearman_seed_mean"] = float(
            np.mean(
                [
                    row["catalog_spearman_against_original"]
                    for row in seed_metric_rows
                ]
            )
        )
        ensemble_metrics["catalog_spearman_seed_sd"] = float(
            np.std(
                [
                    row["catalog_spearman_against_original"]
                    for row in seed_metric_rows
                ],
                ddof=0,
            )
        )
        metrics.append(ensemble_metrics)

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output / "fold_runs.jsonl", attempts)
    atomic_write_jsonl(args.output / "model_scores.jsonl", all_seed_rows)
    atomic_write_jsonl(args.output / "ensemble_scores.jsonl", ensembles)
    atomic_write_jsonl(args.output / "sequence_metrics.jsonl", metrics)
    latest = {
        (
            f"{row['scenario']}--{row['condition']}--seed-{row['seed']}--"
            f"fold-{row['outer_fold']}"
        ): row["attempt"]
        for row in attempts
    }
    atomic_write_json(args.output / "latest_successful_runs.json", latest)
    atomic_write_json(
        args.output / "MANIFEST.json",
        {
            "schema_version": "sequence_oof_canonical_manifest",
            "experiment_id": EXPERIMENT_ID,
            "source_hashes": hashes,
            "fold_run_count": len(attempts),
            "seed_oof_row_count": len(all_seed_rows),
            "ensemble_row_count": len(ensembles),
            "metric_row_count": len(metrics),
            "files": {
                name: file_sha256(args.output / name)
                for name in CANONICAL_OUTPUT_FILES
            },
        },
    )
    validate_canonical_manifest(
        args.output,
        load_json_object(args.output / "MANIFEST.json"),
    )
    print(
        json.dumps(
            {
                "fold_runs": len(attempts),
                "seed_oof_rows": len(all_seed_rows),
                "ensemble_rows": len(ensembles),
                "metric_rows": len(metrics),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
