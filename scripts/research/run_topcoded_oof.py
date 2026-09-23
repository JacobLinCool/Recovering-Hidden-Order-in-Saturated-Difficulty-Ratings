#!/usr/bin/env python
"""Run one append-only top-coded Oni out-of-fold experiment attempt."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from TaikoChartEstimator.research.censored_linear import (
    GlobalLinearHuberRegressor,
)
from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    create_attempt_directory,
    file_sha256,
    finish_run,
    start_run,
)
from TaikoChartEstimator.research.features import (
    BASE_FEATURE_NAMES,
    feature_matrix,
    load_feature_records,
)
from TaikoChartEstimator.research.oof import (
    SongGroupFold,
    build_hidden_tail_panel,
    build_song_group_folds,
    hidden_tail_rank_metrics,
    top_code_targets,
)
from TaikoChartEstimator.research.ordinal import (
    CumulativeLinkOrdinalRegressor,
)
from TaikoChartEstimator.research.ordinal_reduction import (
    OrdinalLogitReduction,
)

EXPERIMENT_ID = "icassp2027_topcoded_oof_v1"
CONDITIONS = (
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
    "ordinal_logit",
    "ordinal_probit",
    "ordinal_logit_reduction",
)
SCENARIOS = ("native", "cap7", "cap8", "cap9")
STAR_SCALE_CONDITIONS = {
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
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
        "--output-root",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/raw/runs"
        ),
    )
    return parser.parse_args()


def _oni_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records if record["difficulty"] == "oni"]
    if not selected:
        raise ValueError("feature table contains no Oni charts")
    chart_ids = [str(record["chart_id"]) for record in selected]
    if len(chart_ids) != len(set(chart_ids)):
        raise ValueError("Oni chart IDs are not unique")
    if any(int(record["difficulty_id"]) != 3 for record in selected):
        raise ValueError("Oni panel contains a non-Oni difficulty ID")
    return sorted(selected, key=lambda record: str(record["chart_id"]))


def _candidate_parameters(
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
    raise ValueError(f"unsupported condition: {condition}")


def _build_model(
    condition: str,
    parameters: Mapping[str, Any],
    config: Mapping[str, Any],
):
    if condition in {"ordinary_huber", "topcoded_huber"}:
        return GlobalLinearHuberRegressor(
            alpha=float(parameters["alpha"]),
            censored=condition == "topcoded_huber",
            huber_delta=float(config["huber_delta"]),
            max_iter=int(config["optimizer_max_iter"]),
            tolerance=float(config["optimizer_tolerance"]),
        )
    if condition == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(parameters["alpha"])),
        )
    if condition == "gbdt":
        return HistGradientBoostingRegressor(
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            early_stopping=False,
            random_state=int(config["seed"]),
        )
    if condition in {"ordinal_logit", "ordinal_probit"}:
        return CumulativeLinkOrdinalRegressor(
            alpha=float(parameters["alpha"]),
            link="logit" if condition == "ordinal_logit" else "probit",
            max_iter=int(config["optimizer_max_iter"]),
            tolerance=float(config["optimizer_tolerance"]),
            min_threshold_gap=float(config["minimum_threshold_gap"]),
        )
    if condition == "ordinal_logit_reduction":
        return OrdinalLogitReduction(
            C=float(parameters["C"]),
            max_iter=int(config["optimizer_max_iter"]),
            tolerance=float(config["optimizer_tolerance"]),
            random_state=int(config["seed"]),
        )
    raise ValueError(f"unsupported condition: {condition}")


def _fit_model(
    condition: str,
    parameters: Mapping[str, Any],
    config: Mapping[str, Any],
    matrix: np.ndarray,
    observed_targets: np.ndarray,
    *,
    cap: float,
):
    model = _build_model(condition, parameters, config)
    if condition in {"ordinary_huber", "topcoded_huber"}:
        right = observed_targets >= cap
        model.fit(
            matrix,
            observed_targets,
            is_left_censored=np.zeros(len(observed_targets), dtype=bool),
            is_right_censored=right,
        )
    else:
        model.fit(matrix, observed_targets)
    return model


def _predict_model(
    condition: str,
    model: Any,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if condition in {"ordinal_logit", "ordinal_probit"}:
        latent = np.asarray(model.predict_latent(matrix), dtype=np.float64)
        expected = np.asarray(
            model.predict_expected_label(matrix),
            dtype=np.float64,
        )
    elif condition == "ordinal_logit_reduction":
        expected = np.asarray(
            model.predict_expected_label(matrix),
            dtype=np.float64,
        )
        latent = np.asarray(model.predict_latent(matrix), dtype=np.float64)
    else:
        expected = np.asarray(model.predict(matrix), dtype=np.float64)
        latent = expected.copy()
    if (
        latent.shape != (len(matrix),)
        or expected.shape != (len(matrix),)
        or not np.all(np.isfinite(latent))
        or not np.all(np.isfinite(expected))
    ):
        raise RuntimeError(
            f"{condition} produced invalid prediction arrays: "
            f"latent={latent.shape}, expected={expected.shape}"
        )
    return latent, expected


def _model_diagnostic(
    condition: str,
    model: Any,
) -> dict[str, Any]:
    if condition in {"ordinary_huber", "topcoded_huber"}:
        return model.coefficient_record(BASE_FEATURE_NAMES)
    if condition in {"ordinal_logit", "ordinal_probit"}:
        return model.coefficient_record(BASE_FEATURE_NAMES)
    if condition == "ordinal_logit_reduction":
        return model.coefficient_record(BASE_FEATURE_NAMES)
    if condition == "ridge":
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        return {
            "model": "ridge",
            "feature_names": list(BASE_FEATURE_NAMES),
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "standardized_coefficients": ridge.coef_.tolist(),
            "standardized_intercept": float(ridge.intercept_),
            "n_iter": (
                None
                if ridge.n_iter_ is None
                else np.asarray(ridge.n_iter_).tolist()
            ),
        }
    if condition == "gbdt":
        return {
            "model": "hist_gradient_boosting_regressor",
            "feature_names": list(BASE_FEATURE_NAMES),
            "n_iter": int(model.n_iter_),
            "learning_rate": float(model.learning_rate),
            "max_leaf_nodes": int(model.max_leaf_nodes),
        }
    raise ValueError(f"unsupported condition: {condition}")


def _records_for_groups(
    records: Sequence[Mapping[str, Any]],
    groups: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected_groups = set(groups)
    selected = [
        record
        for record in records
        if str(record["normalized_title"]) in selected_groups
    ]
    if not selected:
        raise ValueError("fold panel is empty")
    return selected


def _observed_targets(
    records: Sequence[Mapping[str, Any]],
    *,
    cap: float,
) -> np.ndarray:
    coded = top_code_targets(
        [float(record["star"]) for record in records],
        [int(record["difficulty_id"]) for record in records],
        cap=cap,
        selected_course_ids={3},
    )
    return np.asarray(coded.observed_targets, dtype=np.float64)


def _validation_mae(
    expected: np.ndarray,
    observed_targets: np.ndarray,
    *,
    cap: float,
) -> float:
    clipped = np.clip(expected, 1.0, cap)
    value = float(np.abs(clipped - observed_targets).mean())
    if not math.isfinite(value):
        raise RuntimeError("validation MAE is non-finite")
    return value


def _fold_assignment_rows(
    records: Sequence[Mapping[str, Any]],
    folds: Sequence[SongGroupFold],
) -> list[dict[str, Any]]:
    outer_by_group: dict[str, int] = {}
    inner_validation_by_group: dict[str, list[int]] = {}
    for fold in folds:
        for group in fold.test_groups:
            if group in outer_by_group:
                raise RuntimeError(f"group {group!r} has multiple outer folds")
            outer_by_group[group] = int(fold.outer_fold)
        for group in fold.validation_groups:
            inner_validation_by_group.setdefault(group, []).append(
                int(fold.outer_fold)
            )
    rows: list[dict[str, Any]] = []
    for record in records:
        group = str(record["normalized_title"])
        if group not in outer_by_group:
            raise RuntimeError(f"group {group!r} has no outer test fold")
        rows.append(
            {
                "chart_id": str(record["chart_id"]),
                "song_id": str(record["song_id"]),
                "normalized_title": group,
                "original_star": int(record["star"]),
                "outer_test_fold": outer_by_group[group],
                "inner_validation_for_outer_folds": sorted(
                    inner_validation_by_group.get(group, [])
                ),
            }
        )
    return sorted(rows, key=lambda row: row["chart_id"])


def _run_fold(
    *,
    fold: SongGroupFold,
    condition: str,
    scenario: str,
    cap: float,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    attempt: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    inner_train = _records_for_groups(records, fold.train_groups)
    validation = _records_for_groups(records, fold.validation_groups)
    outer_train = [*inner_train, *validation]
    test = _records_for_groups(records, fold.test_groups)

    train_groups = {str(row["normalized_title"]) for row in inner_train}
    validation_groups = {str(row["normalized_title"]) for row in validation}
    test_groups = {str(row["normalized_title"]) for row in test}
    if (
        train_groups & validation_groups
        or train_groups & test_groups
        or validation_groups & test_groups
    ):
        raise RuntimeError(f"title leakage in outer fold {fold.outer_fold}")

    x_train = feature_matrix(inner_train, include_course=False)
    y_train = _observed_targets(inner_train, cap=cap)
    x_validation = feature_matrix(validation, include_course=False)
    y_validation = _observed_targets(validation, cap=cap)

    search_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for parameters in _candidate_parameters(condition, config):
        model = _fit_model(
            condition,
            parameters,
            config,
            x_train,
            y_train,
            cap=cap,
        )
        _, expected = _predict_model(condition, model, x_validation)
        mae = _validation_mae(expected, y_validation, cap=cap)
        serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        row = {
            "scenario": scenario,
            "condition": condition,
            "outer_fold": int(fold.outer_fold),
            "parameters": dict(parameters),
            "validation_observed_mae": mae,
            "selected": False,
        }
        search_rows.append(row)
        candidates.append((mae, serialized, dict(parameters)))
    _, _, selected_parameters = min(candidates)
    for row in search_rows:
        row["selected"] = row["parameters"] == selected_parameters

    x_outer_train = feature_matrix(outer_train, include_course=False)
    y_outer_train = _observed_targets(outer_train, cap=cap)
    model = _fit_model(
        condition,
        selected_parameters,
        config,
        x_outer_train,
        y_outer_train,
        cap=cap,
    )
    x_test = feature_matrix(test, include_course=False)
    latent, expected = _predict_model(condition, model, x_test)
    observed_test = _observed_targets(test, cap=cap)
    clipped = np.clip(expected, 1.0, cap)

    prediction_rows: list[dict[str, Any]] = []
    for record, latent_value, expected_value, clipped_value, observed_value in zip(
        test,
        latent,
        expected,
        clipped,
        observed_test,
    ):
        original = float(record["star"])
        prediction_rows.append(
            {
                "scenario": scenario,
                "condition": condition,
                "seed": int(config["seed"]),
                "outer_fold": int(fold.outer_fold),
                "chart_id": str(record["chart_id"]),
                "song_id": str(record["song_id"]),
                "title": str(record["title"]),
                "normalized_title": str(record["normalized_title"]),
                "difficulty": "oni",
                "difficulty_id": 3,
                "original_star": original,
                "observed_star": float(observed_value),
                "active_cap": cap,
                "is_top_coded": bool(original >= cap),
                "is_strictly_hidden": bool(original > cap),
                "latent_score": float(latent_value),
                "expected_observed_label": float(expected_value),
                "clipped_observed_prediction": float(clipped_value),
                "primary_split": str(record["primary_split"]),
            }
        )

    diagnostics = {
        "scenario": scenario,
        "condition": condition,
        "outer_fold": int(fold.outer_fold),
        "inner_train_charts": len(inner_train),
        "inner_validation_charts": len(validation),
        "outer_train_charts": len(outer_train),
        "outer_test_charts": len(test),
        "inner_train_groups": len(train_groups),
        "inner_validation_groups": len(validation_groups),
        "outer_test_groups": len(test_groups),
        "selected_parameters": selected_parameters,
        "validation_observed_mae": min(row["validation_observed_mae"] for row in search_rows),
        "test_observed_mae": float(np.abs(clipped - observed_test).mean()),
    }
    model_record = {
        "scenario": scenario,
        "condition": condition,
        "outer_fold": int(fold.outer_fold),
        "selected_parameters": selected_parameters,
        "diagnostic": _model_diagnostic(condition, model),
    }
    model_directory = attempt / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_directory / f"fold-{fold.outer_fold}.joblib")
    return prediction_rows, search_rows, diagnostics, model_record


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("config experiment_id does not match runner")
    if args.condition not in config["conditions"]:
        raise ValueError(f"condition {args.condition!r} is not frozen in config")
    if args.scenario not in config["scenarios"]:
        raise ValueError(f"scenario {args.scenario!r} is not frozen in config")
    cap = float(config["scenarios"][args.scenario]["cap"])
    primary_evidence = bool(config["primary_evidence"])
    condition_key = f"{args.scenario}--{args.condition}"
    run_condition = (
        condition_key if primary_evidence else f"{condition_key}__smoke"
    )

    records = _oni_records(load_feature_records(args.features))
    folds = build_song_group_folds(
        [str(record["normalized_title"]) for record in records],
        n_folds=int(config["outer_folds"]),
        seed=int(config["seed"]),
        validation_fraction=float(config["inner_validation_fraction"]),
    )
    assignments = _fold_assignment_rows(records, folds)
    attempt = create_attempt_directory(
        args.output_root,
        run_condition,
        int(config["seed"]),
    )
    run_manifest = start_run(
        attempt,
        experiment_id=EXPERIMENT_ID,
        condition=run_condition,
        seed=int(config["seed"]),
        inputs={
            "features": str(args.features),
            "features_sha256": file_sha256(args.features),
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_manifest_sha256": file_sha256(args.dataset_manifest),
            "config": str(args.config),
            "config_sha256": file_sha256(args.config),
            "spec": str(args.spec),
            "spec_sha256": file_sha256(args.spec),
        },
        parameters={
            **config,
            "scenario": args.scenario,
            "condition": args.condition,
            "active_cap": cap,
        },
    )
    try:
        predictions: list[dict[str, Any]] = []
        search: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        model_records: list[dict[str, Any]] = []
        for fold in folds:
            fold_predictions, fold_search, fold_diagnostics, model_record = (
                _run_fold(
                    fold=fold,
                    condition=args.condition,
                    scenario=args.scenario,
                    cap=cap,
                    records=records,
                    config=config,
                    attempt=attempt,
                )
            )
            predictions.extend(fold_predictions)
            search.extend(fold_search)
            diagnostics.append(fold_diagnostics)
            model_records.append(model_record)

        predictions = sorted(predictions, key=lambda row: row["chart_id"])
        chart_ids = [row["chart_id"] for row in predictions]
        expected_chart_ids = {str(record["chart_id"]) for record in records}
        if (
            len(predictions) != len(records)
            or len(chart_ids) != len(set(chart_ids))
            or set(chart_ids) != expected_chart_ids
        ):
            raise RuntimeError("OOF prediction coverage is not exactly one per chart")

        original_targets = np.asarray(
            [float(row["original_star"]) for row in predictions],
            dtype=np.float64,
        )
        observed_targets = np.asarray(
            [float(row["observed_star"]) for row in predictions],
            dtype=np.float64,
        )
        expected = np.asarray(
            [float(row["expected_observed_label"]) for row in predictions],
            dtype=np.float64,
        )
        latent = np.asarray(
            [float(row["latent_score"]) for row in predictions],
            dtype=np.float64,
        )
        coded = top_code_targets(
            original_targets,
            np.full(len(predictions), 3, dtype=np.int64),
            cap=cap,
            selected_course_ids={3},
        )
        hidden_panel = build_hidden_tail_panel(
            coded,
            [str(row["normalized_title"]) for row in predictions],
        )
        aggregate_metrics: dict[str, Any] = {
            "scenario": args.scenario,
            "condition": args.condition,
            "chart_count": len(predictions),
            "catalog_song_group_count": len(
                {str(row["normalized_title"]) for row in predictions}
            ),
            "active_cap": cap,
            "oof_observed_mae": _validation_mae(
                expected,
                observed_targets,
                cap=cap,
            ),
            "oof_original_mae": float(
                np.abs(np.clip(expected, 1.0, 10.0) - original_targets).mean()
            ),
            **hidden_tail_rank_metrics(hidden_panel, latent),
        }
        if args.condition in STAR_SCALE_CONDITIONS:
            top_mask = original_targets >= cap
            shortfall = np.maximum(cap - latent[top_mask], 0.0)
            aggregate_metrics.update(
                {
                    "top_category_lower_bound_violation_rate": float(
                        (shortfall > 0.0).mean()
                    ),
                    "top_category_mean_shortfall": float(shortfall.mean()),
                    "lower_bound_diagnostic_undefined_reason": None,
                }
            )
        else:
            aggregate_metrics.update(
                {
                    "top_category_lower_bound_violation_rate": None,
                    "top_category_mean_shortfall": None,
                    "lower_bound_diagnostic_undefined_reason": (
                        "latent_ordinal_score_is_not_in_star_units"
                    ),
                }
            )

        atomic_write_jsonl(attempt / "fold_assignments.jsonl", assignments)
        atomic_write_jsonl(attempt / "validation_search.jsonl", search)
        atomic_write_jsonl(attempt / "oof_predictions.jsonl", predictions)
        atomic_write_jsonl(attempt / "fold_diagnostics.jsonl", diagnostics)
        atomic_write_json(attempt / "model_records.json", {"folds": model_records})
        atomic_write_json(attempt / "aggregate_metrics.json", aggregate_metrics)
        finish_run(
            attempt,
            run_manifest,
            status="success",
            outputs={
                "fold_assignments": "fold_assignments.jsonl",
                "validation_search": "validation_search.jsonl",
                "oof_predictions": "oof_predictions.jsonl",
                "fold_diagnostics": "fold_diagnostics.jsonl",
                "model_records": "model_records.json",
                "aggregate_metrics": "aggregate_metrics.json",
                "model_directory": "models",
            },
        )
        print(
            json.dumps(
                {
                    "attempt": str(attempt),
                    "scenario": args.scenario,
                    "condition": args.condition,
                    "metrics": aggregate_metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception:
        error = traceback.format_exc()
        finish_run(
            attempt,
            run_manifest,
            status="failed",
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
