#!/usr/bin/env python
"""Resume the frozen 80-fold GPU matrix with a hard cost stop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    source_files_snapshot,
)
from TaikoChartEstimator.research.sequence import SEQUENCE_TRAINING_SOURCE_PATHS

EXPERIMENT_ID = "icassp2027_sequence_oof"
EXPECTED_FOLD_RUNS = 80
REQUIRED_OUTPUTS = (
    "selection_model.safetensors",
    "best_model.safetensors",
    "selection_history.jsonl",
    "refit_history.jsonl",
    "test_predictions.jsonl",
    "fold_metrics.json",
    "checkpoint_selection.json",
    "score_standardization.json",
    "model_record.json",
)


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
        "--cost-ledger",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/raw/cost_ledger.jsonl"),
    )
    parser.add_argument(
        "--result-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_sequence_oof/raw/REMOTE_RESULT_MANIFEST.json"
        ),
    )
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--budget-usd", type=float, default=25.0)
    parser.add_argument(
        "--billing-started-at-utc",
        required=True,
        help="RunPod billing start as an ISO-8601 UTC timestamp.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("billing-started-at-utc must include a UTC offset")
    converted = parsed.astimezone(timezone.utc)
    if converted > datetime.now(timezone.utc):
        raise ValueError("billing start cannot be in the future")
    return converted


def planned_fold_runs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for row in config.get("matrix", []):
        for seed in row.get("seeds", []):
            for outer_fold in range(int(config["outer_folds"])):
                planned.append(
                    {
                        "scenario": str(row["scenario"]),
                        "condition": str(row["condition"]),
                        "seed": int(seed),
                        "outer_fold": outer_fold,
                    }
                )
    keys = {
        (row["scenario"], row["condition"], row["seed"], row["outer_fold"])
        for row in planned
    }
    if len(planned) != EXPECTED_FOLD_RUNS or len(keys) != EXPECTED_FOLD_RUNS:
        raise ValueError(
            f"frozen matrix must contain {EXPECTED_FOLD_RUNS} unique fold-runs"
        )
    return planned


def attempt_parent(root: Path, row: Mapping[str, Any]) -> Path:
    key = f"{row['scenario']}--{row['condition']}--fold-{row['outer_fold']}"
    return root / key / f"seed-{row['seed']}"


def matching_success(
    root: Path,
    row: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
) -> Path | None:
    for attempt in sorted(attempt_parent(root, row).glob("attempt-*"), reverse=True):
        status_path = attempt / "status.json"
        manifest_path = attempt / "run_manifest.json"
        if not status_path.is_file() or not manifest_path.is_file():
            continue
        status = load_json_object(status_path)
        manifest = load_json_object(manifest_path)
        if status.get("status") != "success" or manifest.get("status") != "success":
            continue
        if (
            manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("scenario") != row["scenario"]
            or manifest.get("condition") != row["condition"]
            or int(manifest.get("seed", -1)) != row["seed"]
            or int(manifest.get("outer_fold", -1)) != row["outer_fold"]
        ):
            continue
        if not manifest.get("parameters", {}).get("primary_evidence", False):
            continue
        inputs = manifest.get("inputs", {})
        if any(inputs.get(key) != value for key, value in source_hashes.items()):
            continue
        if any(not (attempt / name).is_file() for name in REQUIRED_OUTPUTS):
            continue
        return attempt
    return None


def successful_smoke_exists(
    root: Path,
    *,
    source_hashes: Mapping[str, str],
    expected_architecture: Mapping[str, Any],
    expected_panel_size: int = 1_023,
) -> bool:
    pattern = root.glob("*--smoke/seed-*/attempt-*/run_manifest.json")
    for manifest_path in pattern:
        attempt = manifest_path.parent
        status_path = attempt / "status.json"
        if not status_path.is_file():
            continue
        manifest = load_json_object(manifest_path)
        status = load_json_object(status_path)
        if status.get("status") != "success" or manifest.get("status") != "success":
            continue
        if manifest.get("parameters", {}).get("primary_evidence", True):
            continue
        inputs = manifest.get("inputs", {})
        if any(inputs.get(key) != value for key, value in source_hashes.items()):
            continue
        parameters = manifest.get("parameters", {})
        partitions = parameters.get("partition_sizes", {})
        train_size = int(partitions.get("train", -1))
        validation_size = int(partitions.get("validation", -1))
        test_size = int(partitions.get("test", -1))
        refit_size = int(partitions.get("refit", -1))
        outputs = manifest.get("outputs", {})
        if (
            parameters.get("architecture") != dict(expected_architecture)
            or int(parameters.get("training", {}).get("epochs", -1)) != 1
            or not str(parameters.get("device", "")).startswith("cuda")
            or min(train_size, validation_size, test_size) <= 0
            or train_size + validation_size + test_size != expected_panel_size
            or refit_size != train_size + validation_size
            or not isinstance(
                outputs.get("peak_cuda_memory_allocated_bytes"), int
            )
            or int(outputs["peak_cuda_memory_allocated_bytes"]) <= 0
            or any(not (attempt / name).is_file() for name in REQUIRED_OUTPUTS)
        ):
            continue
        return True
    return False


def elapsed_billing_hours(started_at: datetime) -> float:
    return max(
        0.0,
        (datetime.now(timezone.utc) - started_at).total_seconds() / 3600.0,
    )


def ledger_event(
    *,
    event: str,
    hourly_rate: float,
    budget: float,
    billing_started_at: datetime,
    completed: int,
    remaining: int,
    projected_total_cost: float | None,
    run: Mapping[str, Any] | None = None,
    attempt: Path | None = None,
) -> dict[str, Any]:
    hours = elapsed_billing_hours(billing_started_at)
    return {
        "schema_version": "runpod_cost_ledger",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "hourly_rate_usd": hourly_rate,
        "budget_usd": budget,
        "billing_started_at_utc": billing_started_at.isoformat(),
        "billed_hours_estimate": hours,
        "cost_estimate_usd": hours * hourly_rate,
        "projected_total_cost_usd": projected_total_cost,
        "completed_fold_runs": completed,
        "remaining_fold_runs": remaining,
        "run": dict(run) if run is not None else None,
        "attempt": str(attempt) if attempt is not None else None,
    }


def append_ledger(path: Path, row: Mapping[str, Any]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    atomic_write_jsonl(path, [*existing, dict(row)])


def write_result_manifest(
    *,
    runs: Path,
    cost_ledger: Path,
    output: Path,
    root: Path = Path("."),
) -> dict[str, Any]:
    """Hash the complete remotely produced evidence set for local verification."""

    resolved_root = root.resolve()
    resolved_runs = runs.resolve()
    resolved_output = output.resolve()
    resolved_ledger = cost_ledger.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "result manifest output must stay within repository root"
        ) from exc
    if resolved_output == resolved_ledger or resolved_output.is_relative_to(
        resolved_runs
    ):
        raise ValueError("result manifest cannot be part of its own evidence set")
    candidates = sorted(
        (path for path in runs.rglob("*") if path.is_file()),
        key=lambda path: path.resolve().relative_to(resolved_root).as_posix(),
    )
    if cost_ledger.is_file():
        candidates.append(cost_ledger)
    if not candidates:
        raise RuntimeError("cannot manifest an empty remote result set")

    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"result evidence cannot be a symlink: {candidate}")
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"result evidence lies outside repository root: {candidate}"
            ) from exc
        if relative in seen:
            raise ValueError(f"duplicate result evidence path: {relative}")
        seen.add(relative)
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    files.sort(key=lambda row: row["path"])
    manifest = {
        "schema_version": "sequence_remote_result_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    atomic_write_json(output, manifest)
    return manifest


def execute_pending_matrix(
    *,
    args: argparse.Namespace,
    billing_started_at: datetime,
    source_hashes: Mapping[str, str],
    pending: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    durations: list[float] = []
    append_ledger(
        args.cost_ledger,
        ledger_event(
            event="matrix_start",
            hourly_rate=args.hourly_rate_usd,
            budget=args.budget_usd,
            billing_started_at=billing_started_at,
            completed=len(completed_rows),
            remaining=len(pending),
            projected_total_cost=None,
        ),
    )
    for index, row in enumerate(pending):
        spent = elapsed_billing_hours(billing_started_at) * args.hourly_rate_usd
        remaining_count = len(pending) - index
        projected = None
        if durations:
            projected_hours = (
                elapsed_billing_hours(billing_started_at)
                + (sum(durations) / len(durations)) * remaining_count / 3600.0
            )
            projected = projected_hours * args.hourly_rate_usd
        if spent >= args.budget_usd or (
            projected is not None and projected > args.budget_usd
        ):
            append_ledger(
                args.cost_ledger,
                ledger_event(
                    event="budget_stop",
                    hourly_rate=args.hourly_rate_usd,
                    budget=args.budget_usd,
                    billing_started_at=billing_started_at,
                    completed=len(completed_rows),
                    remaining=remaining_count,
                    projected_total_cost=projected,
                    run=row,
                ),
            )
            raise RuntimeError(
                "projected or accrued matrix cost exceeds the US$25 hard budget"
            )
        command = [
            sys.executable,
            "scripts/research/run_sequence_oof_fold.py",
            "--scenario",
            row["scenario"],
            "--condition",
            row["condition"],
            "--seed",
            str(row["seed"]),
            "--outer-fold",
            str(row["outer_fold"]),
            "--config",
            str(args.config),
            "--spec",
            str(args.spec),
            "--fold-assignments",
            str(args.fold_assignments),
            "--dataset-manifest",
            str(args.dataset_manifest),
            "--dataset-snapshot",
            str(args.dataset_snapshot),
            "--output-root",
            str(args.runs),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.cache_dir is not None:
            command.extend(("--cache-dir", args.cache_dir))
        started = time.monotonic()
        subprocess.run(command, check=True)
        durations.append(time.monotonic() - started)
        attempt = matching_success(
            args.runs,
            row,
            source_hashes=source_hashes,
        )
        if attempt is None:
            raise RuntimeError(
                "fold command returned success without canonical raw evidence"
            )
        completed_rows.append({**row, "attempt": str(attempt)})
        remaining_after = len(pending) - index - 1
        projected_hours = (
            elapsed_billing_hours(billing_started_at)
            + (sum(durations) / len(durations)) * remaining_after / 3600.0
        )
        append_ledger(
            args.cost_ledger,
            ledger_event(
                event="fold_success",
                hourly_rate=args.hourly_rate_usd,
                budget=args.budget_usd,
                billing_started_at=billing_started_at,
                completed=len(completed_rows),
                remaining=remaining_after,
                projected_total_cost=projected_hours * args.hourly_rate_usd,
                run=row,
                attempt=attempt,
            ),
        )
    append_ledger(
        args.cost_ledger,
        ledger_event(
            event="matrix_complete",
            hourly_rate=args.hourly_rate_usd,
            budget=args.budget_usd,
            billing_started_at=billing_started_at,
            completed=len(completed_rows),
            remaining=0,
            projected_total_cost=(
                elapsed_billing_hours(billing_started_at) * args.hourly_rate_usd
            ),
        ),
    )
    return completed_rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.hourly_rate_usd < 0.5:
        raise ValueError("hourly GPU rate must be positive and below US$0.50")
    if not 0.0 < args.budget_usd <= 25.0:
        raise ValueError("budget must be positive and no greater than US$25")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    billing_started_at = parse_utc_timestamp(args.billing_started_at_utc)
    config = load_json_object(args.config)
    if not config.get("primary_evidence", False):
        raise ValueError("matrix runner requires a primary-evidence config")
    planned = planned_fold_runs(config)
    snapshot_manifest = args.dataset_snapshot / "RESEARCH_DATASET_SNAPSHOT.json"
    source_hashes = {
        "config_sha256": file_sha256(args.config),
        "spec_sha256": file_sha256(args.spec),
        "fold_assignments_sha256": file_sha256(args.fold_assignments),
        "dataset_manifest_sha256": file_sha256(args.dataset_manifest),
        "dataset_snapshot_manifest_sha256": file_sha256(snapshot_manifest),
    }
    source_hashes["training_source_sha256"] = source_files_snapshot(
        SEQUENCE_TRAINING_SOURCE_PATHS
    )["aggregate_sha256"]
    if not args.dry_run and not successful_smoke_exists(
        args.runs,
        source_hashes=source_hashes,
        expected_architecture=config["architecture"],
    ):
        raise RuntimeError(
            "no matching successful smoke attempt; run one fold with --smoke first"
        )
    completed_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in planned:
        attempt = matching_success(
            args.runs,
            row,
            source_hashes=source_hashes,
        )
        if attempt is None:
            pending.append(row)
        else:
            completed_rows.append({**row, "attempt": str(attempt)})
    if args.dry_run:
        print(
            json.dumps(
                {
                    "expected_fold_runs": len(planned),
                    "matching_successes": len(completed_rows),
                    "pending": pending,
                },
                indent=2,
            )
        )
        return

    try:
        completed_rows = execute_pending_matrix(
            args=args,
            billing_started_at=billing_started_at,
            source_hashes=source_hashes,
            pending=pending,
            completed_rows=completed_rows,
        )
    except BaseException as error:
        try:
            write_result_manifest(
                runs=args.runs,
                cost_ledger=args.cost_ledger,
                output=args.result_manifest,
            )
        except Exception as manifest_error:
            error.add_note(f"result manifest also failed: {manifest_error}")
        raise
    result_manifest = write_result_manifest(
        runs=args.runs,
        cost_ledger=args.cost_ledger,
        output=args.result_manifest,
    )
    print(
        json.dumps(
            {
                "completed_fold_runs": len(completed_rows),
                "expected_fold_runs": EXPECTED_FOLD_RUNS,
                "estimated_cost_usd": (
                    elapsed_billing_hours(billing_started_at) * args.hourly_rate_usd
                ),
                "budget_usd": args.budget_usd,
                "result_manifest": str(args.result_manifest),
                "result_file_count": result_manifest["file_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
