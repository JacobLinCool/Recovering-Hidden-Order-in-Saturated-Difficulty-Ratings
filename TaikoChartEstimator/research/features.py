"""Deterministic global symbolic features for non-neural baselines."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import (
    DIFFICULTY_TO_ID,
    NOTE_TYPE_TO_ID,
)
from ..data.v2.tokenizer import EventTokenizer
from .data import DIFFICULTIES, load_source_dataset

FEATURE_SCHEMA_VERSION = "global_symbolic_v1"
BASE_FEATURE_NAMES = (
    "chart_duration",
    "note_count",
    "density_mean",
    "density_std",
    "density_p95",
    "density_max",
    "inter_note_mean",
    "inter_note_std",
    "inter_note_min",
    "inter_note_max",
    "bpm_mean",
    "bpm_std",
    "bpm_min",
    "bpm_max",
    "abs_scroll_mean",
    "abs_scroll_max",
    "long_note_fraction",
    "large_note_fraction",
    "note_type_entropy",
)
COURSE_FEATURE_NAMES = tuple(f"course_{name}" for name in DIFFICULTIES)
LONG_NOTE_IDS = {
    NOTE_TYPE_TO_ID[name] for name in ("Roll", "RollBig", "Balloon", "BalloonAlt")
}
LARGE_NOTE_IDS = {
    NOTE_TYPE_TO_ID[name] for name in ("DonBig", "KaBig", "RollBig")
}
END_NOTE_ID = NOTE_TYPE_TO_ID["EndOf"]


def _summary(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(values.mean()),
        float(values.std(ddof=0)),
        float(values.min()),
        float(values.max()),
    )


def _entropy(ids: Sequence[int]) -> float:
    if not ids:
        return 0.0
    counts = np.asarray(list(Counter(ids).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def extract_global_features(
    chart: Mapping[str, Any],
    tokenizer: EventTokenizer | None = None,
) -> dict[str, float]:
    """Extract the frozen set of global statistics from one parsed chart."""

    tokenizer = tokenizer if tokenizer is not None else EventTokenizer()
    segments = chart.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("chart must contain at least one parsed segment")
    tokens = [
        token
        for token in tokenizer.tokenize_chart(segments)
        if token.note_type != END_NOTE_ID
    ]
    if not tokens:
        raise ValueError("chart contains no recognized playable notes")

    timestamps = np.asarray([token.timestamp for token in tokens], dtype=np.float64)
    density = np.asarray([token.local_density for token in tokens], dtype=np.float64)
    bpm = np.asarray([token.bpm for token in tokens], dtype=np.float64)
    abs_scroll = np.abs(
        np.asarray([token.scroll for token in tokens], dtype=np.float64)
    )
    inter_note = np.diff(np.sort(timestamps))
    bpm_mean, bpm_std, bpm_min, bpm_max = _summary(bpm)
    interval_mean, interval_std, interval_min, interval_max = _summary(inter_note)
    note_ids = [token.note_type for token in tokens]

    features = {
        "chart_duration": float(max(timestamps.max() - timestamps.min(), 0.0)),
        "note_count": float(len(tokens)),
        "density_mean": float(density.mean()),
        "density_std": float(density.std(ddof=0)),
        "density_p95": float(np.percentile(density, 95)),
        "density_max": float(density.max()),
        "inter_note_mean": interval_mean,
        "inter_note_std": interval_std,
        "inter_note_min": interval_min,
        "inter_note_max": interval_max,
        "bpm_mean": bpm_mean,
        "bpm_std": bpm_std,
        "bpm_min": bpm_min,
        "bpm_max": bpm_max,
        "abs_scroll_mean": float(abs_scroll.mean()),
        "abs_scroll_max": float(abs_scroll.max()),
        "long_note_fraction": float(
            np.mean([note_id in LONG_NOTE_IDS for note_id in note_ids])
        ),
        "large_note_fraction": float(
            np.mean([note_id in LARGE_NOTE_IDS for note_id in note_ids])
        ),
        "note_type_entropy": _entropy(note_ids),
    }
    if tuple(features) != BASE_FEATURE_NAMES:
        raise RuntimeError("feature extraction order drifted from schema")
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError("global feature extraction produced a non-finite value")
    return features


def build_feature_records(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Materialize one canonical global-feature record per valid chart."""

    source = load_source_dataset(manifest["dataset_name"], cache_dir)
    tokenizer = EventTokenizer()
    records: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        row = source[entry["source_split"]][entry["source_index"]]
        for difficulty in DIFFICULTIES:
            if difficulty not in entry["charts"]:
                continue
            chart = row[difficulty]
            features = extract_global_features(chart, tokenizer)
            difficulty_id = DIFFICULTY_TO_ID[difficulty]
            records.append(
                {
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "chart_id": f"{entry['song_id']}:{difficulty}",
                    "song_id": entry["song_id"],
                    "title": entry["title"],
                    "normalized_title": entry["normalized_title"],
                    "difficulty": difficulty,
                    "difficulty_id": difficulty_id,
                    "star": int(chart["level"]),
                    "primary_split": entry["primary_split"],
                    "in_source_test_no_overlap": entry[
                        "in_source_test_no_overlap"
                    ],
                    **features,
                    **{
                        name: float(index == difficulty_id)
                        for index, name in enumerate(COURSE_FEATURE_NAMES)
                    },
                }
            )
    return records


def save_feature_records(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write feature records atomically as JSON Lines."""

    if not records:
        raise ValueError("cannot save an empty feature table")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def load_feature_records(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the canonical global-feature table."""

    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("global feature table is empty")
    chart_ids: set[str] = set()
    for record in records:
        if record.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported feature schema: {record.get('feature_schema_version')!r}"
            )
        chart_id = record["chart_id"]
        if chart_id in chart_ids:
            raise ValueError(f"duplicate feature chart_id: {chart_id}")
        chart_ids.add(chart_id)
        for feature in (*BASE_FEATURE_NAMES, *COURSE_FEATURE_NAMES):
            value = record.get(feature)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{chart_id} has invalid {feature}: {value!r}")
    return records


def feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    include_course: bool,
) -> np.ndarray:
    """Convert feature records into a stable dense matrix."""

    feature_names = (
        (*BASE_FEATURE_NAMES, *COURSE_FEATURE_NAMES)
        if include_course
        else BASE_FEATURE_NAMES
    )
    return np.asarray(
        [[float(record[name]) for name in feature_names] for record in records],
        dtype=np.float64,
    )
