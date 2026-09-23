#!/usr/bin/env python
"""Train one append-only sequence OOF fold under the frozen protocol."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from scipy.stats import kendalltau, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from TaikoChartEstimator.research.data import (
    SNAPSHOT_MANIFEST_NAME,
    load_manifest,
    load_source_dataset,
    validate_source_against_manifest,
)
from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    create_attempt_directory,
    file_sha256,
    finish_run,
    source_files_snapshot,
    start_run,
)
from TaikoChartEstimator.research.oof import strict_pair_accuracy
from TaikoChartEstimator.research.sequence import (
    SEQUENCE_CONDITIONS,
    SEQUENCE_TRAINING_SOURCE_PATHS,
    BeatSynchronousSequenceEstimator,
    SequenceEstimatorConfig,
    standardize_from_training_scores,
)
from TaikoChartEstimator.research.sequence_data import (
    ObservedLabelCollator,
    OniSequenceDataset,
    SequenceChartRecord,
    load_sequence_chart_records,
    partition_record_indices,
)

EXPERIMENT_ID = "icassp2027_sequence_oof"
EXPECTED_PANEL_SIZE = 1023


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--condition", required=True, choices=sorted(SEQUENCE_CONDITIONS)
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
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
        "--output-root",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/raw/runs"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def phase_seed(seed: int, outer_fold: int, phase: str) -> int:
    offsets = {"selection": 0, "refit": 50_000}
    if phase not in offsets:
        raise ValueError(f"unknown training phase {phase!r}")
    return int(seed) * 100 + int(outer_fold) + offsets[phase]


def validate_planned_key(
    config: Mapping[str, Any],
    *,
    scenario: str,
    condition: str,
    seed: int,
) -> int:
    scenarios = config.get("scenarios", {})
    if scenario not in scenarios:
        raise ValueError(
            f"unknown scenario {scenario!r}; expected one of {sorted(scenarios)}"
        )
    matches = [
        row
        for row in config.get("matrix", [])
        if row.get("scenario") == scenario
        and row.get("condition") == condition
        and seed in row.get("seeds", [])
    ]
    if len(matches) != 1:
        raise ValueError(
            "requested scenario/condition/seed is not exactly one frozen matrix key"
        )
    return int(scenarios[scenario]["cap"])


def resolved_parameters(
    config: Mapping[str, Any], *, smoke: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = dict(config["training"])
    architecture = dict(config["architecture"])
    if smoke:
        training.update(
            {
                "epochs": 1,
                "early_stopping_patience": 1,
            }
        )
    return training, architecture


def estimator_config(architecture: Mapping[str, Any]) -> SequenceEstimatorConfig:
    return SequenceEstimatorConfig(
        d_model=int(architecture["d_model"]),
        encoder_layers=int(architecture["encoder_layers"]),
        attention_heads=int(architecture["attention_heads"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        head_hidden_dim=int(architecture["head_hidden_dim"]),
        max_tokens_per_instance=int(architecture["max_tokens_per_instance"]),
        encoder_pooling=str(architecture["encoder_pooling"]),
    )


def make_loader(
    dataset: OniSequenceDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    if not indices:
        raise ValueError("cannot construct a loader for an empty partition")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=ObservedLabelCollator(dataset.records),
    )


def move_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "instances": batch["instances"].to(device, non_blocking=True),
        "instance_masks": batch["instance_masks"].to(device, non_blocking=True),
        "instance_counts": batch["instance_counts"].to(device, non_blocking=True),
        "observed_label": batch["observed_label"].to(device, non_blocking=True),
    }


def train_epoch(
    model: BeatSynchronousSequenceEstimator,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    *,
    huber_delta: float,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(
            batch["instances"],
            batch["instance_masks"],
            batch["instance_counts"],
        )
        loss = model.observation_loss(
            output,
            batch["observed_label"],
            huber_delta=huber_delta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        count = int(batch["observed_label"].numel())
        total_loss += float(loss.detach().cpu()) * count
        total_examples += count
    if total_examples == 0:
        raise RuntimeError("training loader produced no examples")
    return {"training_observation_loss": total_loss / total_examples}


@torch.inference_mode()
def predict_observed(
    model: BeatSynchronousSequenceEstimator,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(
            batch["instances"],
            batch["instance_masks"],
            batch["instance_counts"],
        )
        raw_scores = output.latent_score.detach().cpu().tolist()
        expected_labels = output.expected_observed_label.detach().cpu().tolist()
        observed_labels = batch["observed_label"].detach().cpu().tolist()
        for index, chart_id in enumerate(raw_batch["chart_ids"]):
            rows.append(
                {
                    "chart_id": chart_id,
                    "song_id": raw_batch["song_ids"][index],
                    "normalized_title": raw_batch["normalized_titles"][index],
                    "observed_label": float(observed_labels[index]),
                    "raw_latent_score": float(raw_scores[index]),
                    "expected_observed_label": float(expected_labels[index]),
                }
            )
    if not rows:
        raise RuntimeError("evaluation loader produced no examples")
    if any(
        not math.isfinite(float(row[field]))
        for row in rows
        for field in (
            "observed_label",
            "raw_latent_score",
            "expected_observed_label",
        )
    ):
        raise RuntimeError("model produced a non-finite prediction")
    return rows


def observed_mae(
    rows: Sequence[Mapping[str, Any]], *, maximum_observed_label: int
) -> float:
    if not rows:
        raise ValueError("rows must not be empty")
    if maximum_observed_label < 2:
        raise ValueError("maximum_observed_label must be at least two")
    return float(
        np.mean(
            [
                abs(
                    float(
                        np.clip(
                            float(row["expected_observed_label"]),
                            1.0,
                            float(maximum_observed_label),
                        )
                    )
                    - float(row["observed_label"])
                )
                for row in rows
            ]
        )
    )


def new_model(
    *, condition: str, cap: int, architecture: Mapping[str, Any], device: torch.device
) -> BeatSynchronousSequenceEstimator:
    return BeatSynchronousSequenceEstimator(
        condition,
        cap,
        estimator_config(architecture),
    ).to(device)


def select_epoch(
    *,
    condition: str,
    cap: int,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    seed: int,
    checkpoint: Path,
    history_path: Path,
) -> tuple[int, float, list[dict[str, Any]]]:
    set_seed(seed)
    model = new_model(
        condition=condition,
        cap=cap,
        architecture=architecture,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    maximum_epochs = int(training["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(maximum_epochs, 1))
    patience = int(training["early_stopping_patience"])
    minimum_delta = float(training["early_stopping_min_delta"])
    best_mae = math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        epoch_started = time.monotonic()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            huber_delta=float(training["huber_delta"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        validation_rows = predict_observed(model, validation_loader, device)
        validation_mae = observed_mae(
            validation_rows, maximum_observed_label=cap
        )
        improved = validation_mae < best_mae - minimum_delta
        if improved:
            best_mae = validation_mae
            best_epoch = epoch
            stale_epochs = 0
            save_model(model, checkpoint)
        else:
            stale_epochs += 1
        thresholds = (
            model.ordinal_head.thresholds().detach().cpu().tolist()
            if model.ordinal_head is not None
            else None
        )
        history.append(
            {
                "phase": "selection",
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation_observed_label_mae": validation_mae,
                "checkpoint_improved": improved,
                "stale_epochs": stale_epochs,
                "ordered_thresholds": thresholds,
                "epoch_wall_seconds": time.monotonic() - epoch_started,
                **train_metrics,
            }
        )
        atomic_write_jsonl(history_path, history)
        scheduler.step()
        if stale_epochs >= patience:
            break
    if best_epoch < 1 or not checkpoint.is_file():
        raise RuntimeError("observed-label selection produced no checkpoint")
    return best_epoch, best_mae, history


def refit_model(
    *,
    condition: str,
    cap: int,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    seed: int,
    checkpoint: Path,
    history_path: Path,
) -> BeatSynchronousSequenceEstimator:
    if epochs < 1:
        raise ValueError("refit epochs must be positive")
    set_seed(seed)
    model = new_model(
        condition=condition,
        cap=cap,
        architecture=architecture,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        metrics = train_epoch(
            model,
            loader,
            optimizer,
            device,
            huber_delta=float(training["huber_delta"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        history.append(
            {
                "phase": "refit",
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_wall_seconds": time.monotonic() - epoch_started,
                **metrics,
            }
        )
        atomic_write_jsonl(history_path, history)
        scheduler.step()
    save_model(model, checkpoint)
    return model


def quick_fold_metrics(
    rows: Sequence[Mapping[str, Any]], *, cap: int
) -> dict[str, Any]:
    original = np.asarray(
        [float(row["evaluation_original_label"]) for row in rows], dtype=np.float64
    )
    expected_unclipped = np.asarray(
        [float(row["expected_observed_label"]) for row in rows], dtype=np.float64
    )
    expected = np.clip(expected_unclipped, 1.0, float(cap))
    latent = np.asarray([float(row["latent_score"]) for row in rows], dtype=np.float64)
    observed = np.asarray(
        [float(row["observed_label"]) for row in rows], dtype=np.float64
    )
    hidden = original >= cap
    metrics: dict[str, Any] = {
        "test_size": int(len(rows)),
        "observed_label_mae": float(np.mean(np.abs(expected - observed))),
        "native_catalog_spearman": float(spearmanr(original, latent).statistic),
        "hidden_tail_size": int(hidden.sum()),
    }
    if hidden.sum() >= 2 and len(set(original[hidden].tolist())) >= 2:
        metrics.update(
            {
                "hidden_tail_spearman": float(
                    spearmanr(original[hidden], latent[hidden]).statistic
                ),
                "hidden_tail_kendall_tau_b": float(
                    kendalltau(original[hidden], latent[hidden], variant="b").statistic
                ),
                "hidden_tail_strict_pair_accuracy": strict_pair_accuracy(
                    original[hidden], latent[hidden]
                ),
            }
        )
    else:
        metrics.update(
            {
                "hidden_tail_spearman": None,
                "hidden_tail_kendall_tau_b": None,
                "hidden_tail_strict_pair_accuracy": None,
            }
        )
    return metrics


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    config = load_json_object(args.config)
    cap = validate_planned_key(
        config,
        scenario=args.scenario,
        condition=args.condition,
        seed=args.seed,
    )
    if not args.smoke and not config.get("primary_evidence", False):
        raise ValueError("primary run requires a primary-evidence config")
    training, architecture = resolved_parameters(config, smoke=args.smoke)
    records = load_sequence_chart_records(
        args.fold_assignments,
        cap=cap,
        expected_count=EXPECTED_PANEL_SIZE,
    )
    partitions = partition_record_indices(records, outer_fold=args.outer_fold)
    device = resolve_device(args.device)
    manifest = load_manifest(args.dataset_manifest)
    source = load_source_dataset(
        manifest["dataset_name"],
        args.cache_dir,
        dataset_snapshot=args.dataset_snapshot,
        expected_source_fingerprints=manifest["source_fingerprints"],
    )
    validate_source_against_manifest(source, manifest)
    dataset = OniSequenceDataset(
        records,
        manifest,
        source,
        window_measures=architecture["window_measures"],
        hop_measures=int(architecture["hop_measures"]),
        max_instances_per_chart=int(architecture["max_instances_per_chart"]),
        max_tokens_per_instance=int(architecture["max_tokens_per_instance"]),
        cache_dir=args.cache_dir,
    )

    snapshot_manifest = args.dataset_snapshot / SNAPSHOT_MANIFEST_NAME
    source_hashes = {
        "config_sha256": file_sha256(args.config),
        "spec_sha256": file_sha256(args.spec),
        "fold_assignments_sha256": file_sha256(args.fold_assignments),
        "dataset_manifest_sha256": file_sha256(args.dataset_manifest),
        "dataset_snapshot_manifest_sha256": file_sha256(snapshot_manifest),
        "training_source_sha256": source_files_snapshot(SEQUENCE_TRAINING_SOURCE_PATHS)[
            "aggregate_sha256"
        ],
    }
    key = f"{args.scenario}--{args.condition}--fold-{args.outer_fold}"
    if args.smoke:
        key = f"{key}--smoke"
    attempt = create_attempt_directory(args.output_root, key, args.seed)
    run_started = time.monotonic()
    run_manifest = start_run(
        attempt,
        experiment_id=EXPERIMENT_ID,
        condition=args.condition,
        scenario=args.scenario,
        outer_fold=args.outer_fold,
        seed=args.seed,
        inputs={
            "config": str(args.config),
            "spec": str(args.spec),
            "fold_assignments": str(args.fold_assignments),
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_snapshot": str(args.dataset_snapshot),
            **source_hashes,
        },
        parameters={
            "primary_evidence": not args.smoke,
            "cap": cap,
            "training": training,
            "architecture": architecture,
            "device": str(device),
            "num_workers": args.num_workers,
            "partition_sizes": {
                name: len(indices) for name, indices in partitions.items()
            },
            "selection_label": "observed_label_mae_only",
        },
    )
    try:
        pin_memory = device.type == "cuda"
        batch_size = int(training["batch_size"])
        selection_seed = phase_seed(args.seed, args.outer_fold, "selection")
        refit_seed = phase_seed(args.seed, args.outer_fold, "refit")
        train_loader = make_loader(
            dataset,
            partitions["train"],
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=selection_seed,
            pin_memory=pin_memory,
        )
        validation_loader = make_loader(
            dataset,
            partitions["validation"],
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=selection_seed,
            pin_memory=pin_memory,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        best_epoch, best_validation_mae, _ = select_epoch(
            condition=args.condition,
            cap=cap,
            architecture=architecture,
            training=training,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            seed=selection_seed,
            checkpoint=attempt / "selection_model.safetensors",
            history_path=attempt / "selection_history.jsonl",
        )
        del train_loader, validation_loader
        gc.collect()
        refit_loader = make_loader(
            dataset,
            partitions["refit"],
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            seed=refit_seed,
            pin_memory=pin_memory,
        )
        model = refit_model(
            condition=args.condition,
            cap=cap,
            architecture=architecture,
            training=training,
            loader=refit_loader,
            epochs=best_epoch,
            device=device,
            seed=refit_seed,
            checkpoint=attempt / "best_model.safetensors",
            history_path=attempt / "refit_history.jsonl",
        )
        del refit_loader
        gc.collect()
        load_model(model, attempt / "best_model.safetensors", device=str(device))
        refit_eval_loader = make_loader(
            dataset,
            partitions["refit"],
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=refit_seed,
            pin_memory=pin_memory,
        )
        test_loader = make_loader(
            dataset,
            partitions["test"],
            batch_size=batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=refit_seed,
            pin_memory=pin_memory,
        )
        refit_rows = predict_observed(model, refit_eval_loader, device)
        del refit_eval_loader
        test_rows = predict_observed(model, test_loader, device)
        del test_loader
        gc.collect()
        training_raw = torch.tensor(
            [row["raw_latent_score"] for row in refit_rows], dtype=torch.float64
        )
        test_raw = torch.tensor(
            [row["raw_latent_score"] for row in test_rows], dtype=torch.float64
        )
        standardized, training_mean, training_sd = standardize_from_training_scores(
            training_raw, test_raw
        )
        record_by_chart: dict[str, SequenceChartRecord] = {
            record.chart_id: record for record in records
        }
        elapsed = time.monotonic() - run_started
        timing = {
            "fold_wall_seconds": elapsed,
            "best_epoch": best_epoch,
        }
        output_rows: list[dict[str, Any]] = []
        for row, latent_score in zip(test_rows, standardized.tolist(), strict=True):
            record = record_by_chart[str(row["chart_id"])]
            output_rows.append(
                {
                    "schema_version": "sequence_oof_prediction",
                    "experiment_id": EXPERIMENT_ID,
                    "scenario": args.scenario,
                    "condition": args.condition,
                    "outer_fold": args.outer_fold,
                    "seed": args.seed,
                    **row,
                    "latent_score": float(latent_score),
                    "evaluation_original_label": record.original_label,
                    "source_hashes": source_hashes,
                    "timing": timing,
                }
            )
        metrics = quick_fold_metrics(output_rows, cap=cap)
        selection_record = {
            "schema_version": "sequence_checkpoint_selection",
            "selection_metric": "validation_observed_label_mae",
            "best_epoch": best_epoch,
            "best_validation_observed_label_mae": best_validation_mae,
            "original_label_used_for_selection": False,
            "refit_epochs": best_epoch,
        }
        threshold_values = (
            model.ordinal_head.thresholds().detach().cpu().tolist()
            if model.ordinal_head is not None
            else None
        )
        atomic_write_jsonl(attempt / "test_predictions.jsonl", output_rows)
        atomic_write_json(attempt / "fold_metrics.json", metrics)
        atomic_write_json(attempt / "checkpoint_selection.json", selection_record)
        atomic_write_json(
            attempt / "score_standardization.json",
            {
                "source_partition": "outer_refit",
                "mean": training_mean,
                "standard_deviation": training_sd,
            },
        )
        atomic_write_json(
            attempt / "model_record.json",
            {
                "condition": args.condition,
                "maximum_observed_label": cap,
                "architecture": architecture,
                "ordered_thresholds": threshold_values,
                "selection_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
        )
        peak_cuda_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        finish_run(
            attempt,
            run_manifest,
            status="success",
            outputs={
                "best_epoch": best_epoch,
                "best_validation_observed_label_mae": best_validation_mae,
                "selection_model": "selection_model.safetensors",
                "best_model": "best_model.safetensors",
                "selection_history": "selection_history.jsonl",
                "refit_history": "refit_history.jsonl",
                "test_predictions": "test_predictions.jsonl",
                "fold_metrics": "fold_metrics.json",
                "checkpoint_selection": "checkpoint_selection.json",
                "score_standardization": "score_standardization.json",
                "model_record": "model_record.json",
                "fold_wall_seconds": elapsed,
                "peak_cuda_memory_allocated_bytes": peak_cuda_memory,
            },
        )
        print(
            json.dumps(
                {
                    "attempt": str(attempt),
                    "best_epoch": best_epoch,
                    "best_validation_observed_label_mae": best_validation_mae,
                    "fold_metrics": metrics,
                    "fold_wall_seconds": elapsed,
                    "peak_cuda_memory_allocated_bytes": peak_cuda_memory,
                },
                indent=2,
            )
        )
    except Exception:
        finish_run(
            attempt,
            run_manifest,
            status="failed",
            error=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
