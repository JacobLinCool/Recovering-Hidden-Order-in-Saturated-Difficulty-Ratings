"""Dataset construction for leakage-free chart-difficulty experiments.

The source dataset contains exact chart duplicates across its published
train/test boundary.  This module therefore creates a frozen, auditable song
manifest before any model is fitted.  All charts belonging to one normalized
song title remain in the same partition.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

DATASET_NAME = "JacobLinCool/taiko-1000-parsed"
DIFFICULTIES = ("easy", "normal", "hard", "oni", "ura")
SOURCE_SPLITS = ("train", "test")
PRIMARY_SPLITS = ("train", "validation", "test")
MANIFEST_SCHEMA_VERSION = "dedup_group_v1"
SNAPSHOT_SCHEMA_VERSION = "icassp_dataset_snapshot_v1"
SNAPSHOT_MANIFEST_NAME = "RESEARCH_DATASET_SNAPSHOT.json"


class ManifestError(ValueError):
    """Raised when a dataset manifest violates the frozen data protocol."""


def normalize_title(value: str) -> str:
    """Return a Unicode-safe identity key for grouping versions of a song."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_fraction(group_key: str, seed: int) -> float:
    """Map a group key deterministically into the half-open interval [0, 1)."""

    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / 2**64


def assign_primary_split(group_key: str, seed: int) -> str:
    """Assign a title group to the frozen 80/10/10 primary split."""

    fraction = stable_fraction(group_key, seed)
    if fraction < 0.8:
        return "train"
    if fraction < 0.9:
        return "validation"
    return "test"


def _without_audio_decoding(dataset: Dataset) -> Dataset:
    if "audio" not in dataset.column_names:
        return dataset
    return dataset.cast_column("audio", Audio(decode=False))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_snapshot(
    path: str | Path,
    *,
    expected_dataset_name: str | None = None,
    expected_source_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify a portable chart-only dataset snapshot before loading it."""

    root = Path(path).resolve()
    metadata_path = root / SNAPSHOT_MANIFEST_NAME
    if not metadata_path.is_file():
        raise ManifestError(
            f"dataset snapshot is missing {SNAPSHOT_MANIFEST_NAME}: {root}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ManifestError(
            "unsupported dataset snapshot schema: "
            f"{metadata.get('schema_version')!r}"
        )
    if (
        expected_dataset_name is not None
        and metadata.get("dataset_name") != expected_dataset_name
    ):
        raise ManifestError(
            "dataset snapshot source differs from the frozen manifest: "
            f"{metadata.get('dataset_name')!r} != {expected_dataset_name!r}"
        )
    if expected_source_fingerprints is not None:
        actual = metadata.get("source_fingerprints")
        expected = dict(expected_source_fingerprints)
        if actual != expected:
            raise ManifestError(
                "dataset snapshot fingerprints differ from the frozen manifest"
            )

    file_rows = metadata.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ManifestError("dataset snapshot contains no file manifest")
    expected_files: dict[str, Mapping[str, Any]] = {}
    for row in file_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ManifestError("dataset snapshot has an invalid file entry")
        relative = row["path"]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ManifestError(f"unsafe dataset snapshot path: {relative!r}")
        if relative in expected_files:
            raise ManifestError(f"duplicate dataset snapshot path: {relative!r}")
        expected_files[relative] = row

    actual_files = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != metadata_path
    }
    if actual_files != set(expected_files):
        raise ManifestError(
            "dataset snapshot membership differs from its manifest: "
            f"missing={sorted(set(expected_files).difference(actual_files))}, "
            f"extra={sorted(actual_files.difference(expected_files))}"
        )
    for relative, row in expected_files.items():
        candidate = root / relative
        if candidate.stat().st_size != int(row.get("size_bytes", -1)):
            raise ManifestError(f"dataset snapshot size mismatch: {relative}")
        if _file_sha256(candidate) != row.get("sha256"):
            raise ManifestError(f"dataset snapshot digest mismatch: {relative}")
    return metadata


def load_source_dataset(
    dataset_name: str = DATASET_NAME,
    cache_dir: str | None = None,
    *,
    dataset_snapshot: str | Path | None = None,
    expected_source_fingerprints: Mapping[str, str] | None = None,
) -> DatasetDict:
    """Load both source splits without decoding audio."""

    if dataset_snapshot is None:
        loaded = load_dataset(dataset_name, cache_dir=cache_dir)
    else:
        validate_dataset_snapshot(
            dataset_snapshot,
            expected_dataset_name=dataset_name,
            expected_source_fingerprints=expected_source_fingerprints,
        )
        loaded = load_from_disk(str(Path(dataset_snapshot).resolve()))
    if not isinstance(loaded, DatasetDict):
        raise ManifestError(f"expected DatasetDict, received {type(loaded).__name__}")

    missing = set(SOURCE_SPLITS).difference(loaded)
    if missing:
        raise ManifestError(f"source dataset is missing splits: {sorted(missing)}")

    return DatasetDict(
        {
            split: _without_audio_decoding(loaded[split])
            for split in SOURCE_SPLITS
        }
    )


def validate_source_against_manifest(
    source: DatasetDict,
    manifest: Mapping[str, Any],
) -> None:
    """Reject source snapshots whose row order or chart labels drifted."""

    for split in SOURCE_SPLITS:
        split_entries = [
            entry
            for entry in manifest["entries"]
            if entry["source_split"] == split
        ]
        if not split_entries:
            continue
        maximum_index = max(int(entry["source_index"]) for entry in split_entries)
        if maximum_index >= len(source[split]):
            raise ManifestError(
                f"source split {split!r} has {len(source[split])} rows, "
                f"but the manifest references index {maximum_index}"
            )
        for entry in split_entries:
            source_index = int(entry["source_index"])
            actual = _chart_summary(source[split][source_index])
            if actual != entry["charts"]:
                raise ManifestError(
                    "source chart summary differs from the frozen manifest at "
                    f"{split}:{source_index}"
                )


def _title_from_row(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ManifestError("row metadata must be a mapping")
    title = metadata.get("TITLE")
    if not isinstance(title, str) or not title.strip():
        raise ManifestError("every row must have a non-empty metadata.TITLE")
    return title.strip()


def _chart_summary(row: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    charts: dict[str, dict[str, int]] = {}
    for difficulty in DIFFICULTIES:
        chart = row.get(difficulty)
        if not isinstance(chart, Mapping):
            continue
        segments = chart.get("segments")
        level = chart.get("level")
        if not segments or not isinstance(level, (int, float)) or not math.isfinite(level):
            continue
        charts[difficulty] = {
            "level": int(level),
            "segment_count": len(segments),
        }
    return charts


def _source_rows(source: DatasetDict) -> Iterable[tuple[str, int, Mapping[str, Any]]]:
    for source_split in SOURCE_SPLITS:
        for source_index in range(len(source[source_split])):
            yield source_split, source_index, source[source_split][source_index]


def _source_fingerprints(source: DatasetDict) -> dict[str, str]:
    return {
        split: str(getattr(source[split], "_fingerprint", "unknown"))
        for split in SOURCE_SPLITS
    }


def build_dataset_manifest(
    *,
    dataset_name: str = DATASET_NAME,
    cache_dir: str | None = None,
    split_seed: int = 2027,
    source: DatasetDict | None = None,
) -> dict[str, Any]:
    """Build the primary and leakage-diagnostic song manifest.

    Exact duplicate TJA rows are retained only once, in deterministic source
    order.  Normalized title groups are then assigned atomically to a primary
    split.
    """

    source = source if source is not None else load_source_dataset(dataset_name, cache_dir)
    seen_tja_hashes: set[str] = set()
    entries: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    source_train_titles: set[str] = set()

    prepared_rows: list[tuple[str, int, str, str, str, dict[str, dict[str, int]]]] = []
    for source_split, source_index, row in _source_rows(source):
        tja = row.get("tja")
        if not isinstance(tja, str) or not tja:
            raise ManifestError(
                f"{source_split}[{source_index}] has no non-empty TJA text"
            )
        title = _title_from_row(row)
        title_norm = normalize_title(title)
        if not title_norm:
            raise ManifestError(
                f"{source_split}[{source_index}] title normalizes to an empty key"
            )
        if source_split == "train":
            source_train_titles.add(title_norm)
        prepared_rows.append(
            (
                source_split,
                source_index,
                title,
                title_norm,
                sha256_text(tja),
                _chart_summary(row),
            )
        )

    for source_split, source_index, title, title_norm, tja_hash, charts in prepared_rows:
        source_ref = f"{source_split}:{source_index}"
        if tja_hash in seen_tja_hashes:
            duplicate_rows.append(
                {
                    "source_ref": source_ref,
                    "source_split": source_split,
                    "source_index": source_index,
                    "tja_sha256": tja_hash,
                    "normalized_title": title_norm,
                }
            )
            continue

        seen_tja_hashes.add(tja_hash)
        primary_split = assign_primary_split(title_norm, split_seed)
        entries.append(
            {
                "song_id": f"tja-{tja_hash[:16]}",
                "source_ref": source_ref,
                "source_split": source_split,
                "source_index": source_index,
                "title": title,
                "normalized_title": title_norm,
                "tja_sha256": tja_hash,
                "primary_split": primary_split,
                "in_source_test_no_overlap": (
                    source_split == "test" and title_norm not in source_train_titles
                ),
                "charts": charts,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "source_fingerprints": _source_fingerprints(source),
        "split_seed": split_seed,
        "entries": entries,
        "excluded_exact_duplicates": duplicate_rows,
    }
    manifest["summary"] = summarize_dataset_manifest(manifest)
    validate_dataset_manifest(manifest)
    return manifest


def summarize_dataset_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compute deterministic audit counts from manifest records."""

    entries = manifest["entries"]
    row_counts = Counter(entry["primary_split"] for entry in entries)
    chart_counts: Counter[str] = Counter()
    censored_counts: Counter[str] = Counter()
    title_group_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    diagnostic_rows = 0

    max_by_difficulty = {"easy": 5, "normal": 7, "hard": 8, "oni": 10, "ura": 10}
    for entry in entries:
        split = entry["primary_split"]
        title_group_counts[split][entry["normalized_title"]] += 1
        diagnostic_rows += int(entry["in_source_test_no_overlap"])
        for difficulty, chart in entry["charts"].items():
            chart_counts[split] += 1
            chart_counts[f"{split}:{difficulty}"] += 1
            if chart["level"] >= max_by_difficulty[difficulty]:
                censored_counts[split] += 1
                censored_counts[f"{split}:{difficulty}"] += 1

    return {
        "source_rows": len(entries) + len(manifest["excluded_exact_duplicates"]),
        "deduplicated_rows": len(entries),
        "excluded_exact_duplicate_rows": len(manifest["excluded_exact_duplicates"]),
        "unique_title_groups": len(
            {entry["normalized_title"] for entry in entries}
        ),
        "rows_by_primary_split": dict(sorted(row_counts.items())),
        "title_groups_by_primary_split": {
            split: len(title_group_counts[split]) for split in PRIMARY_SPLITS
        },
        "charts_by_primary_split": {
            split: chart_counts[split] for split in PRIMARY_SPLITS
        },
        "right_censored_charts_by_primary_split": {
            split: censored_counts[split] for split in PRIMARY_SPLITS
        },
        "charts_by_split_and_difficulty": {
            f"{split}:{difficulty}": chart_counts[f"{split}:{difficulty}"]
            for split in PRIMARY_SPLITS
            for difficulty in DIFFICULTIES
        },
        "source_test_no_overlap_rows": diagnostic_rows,
    }


def validate_dataset_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject duplicate identities, group leakage, and malformed assignments."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported schema_version: {manifest.get('schema_version')!r}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest entries must be a non-empty list")

    song_ids: set[str] = set()
    source_refs: set[str] = set()
    tja_hashes: set[str] = set()
    title_assignments: defaultdict[str, set[str]] = defaultdict(set)

    for entry in entries:
        split = entry.get("primary_split")
        if split not in PRIMARY_SPLITS:
            raise ManifestError(f"invalid primary split: {split!r}")
        for field, seen in (
            ("song_id", song_ids),
            ("source_ref", source_refs),
            ("tja_sha256", tja_hashes),
        ):
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                raise ManifestError(f"entry has invalid {field}: {value!r}")
            if value in seen:
                raise ManifestError(f"duplicate {field}: {value}")
            seen.add(value)
        title_norm = entry.get("normalized_title")
        if not isinstance(title_norm, str) or not title_norm:
            raise ManifestError("entry has empty normalized_title")
        title_assignments[title_norm].add(split)

    leaking = {
        title: sorted(splits)
        for title, splits in title_assignments.items()
        if len(splits) != 1
    }
    if leaking:
        raise ManifestError(f"title groups cross primary splits: {leaking}")

    duplicate_hashes = {
        entry.get("tja_sha256")
        for entry in manifest.get("excluded_exact_duplicates", [])
    }
    unknown_hashes = duplicate_hashes.difference(tja_hashes)
    if unknown_hashes:
        raise ManifestError(
            "excluded duplicates do not reference retained hashes: "
            f"{sorted(unknown_hashes)}"
        )


def save_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Write a validated manifest atomically."""

    validate_dataset_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a frozen manifest."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset_manifest(manifest)
    return manifest


def load_manifest_panel(
    manifest: Mapping[str, Any],
    panel: str,
    *,
    cache_dir: str | None = None,
    source: DatasetDict | None = None,
) -> tuple[Dataset, list[str]]:
    """Materialize a manifest panel and its stable song IDs."""

    if panel not in (*PRIMARY_SPLITS, "source_test_no_overlap"):
        raise ManifestError(f"unknown panel: {panel}")
    source = (
        source
        if source is not None
        else load_source_dataset(manifest["dataset_name"], cache_dir)
    )
    entries = [
        entry
        for entry in manifest["entries"]
        if (
            entry["primary_split"] == panel
            if panel in PRIMARY_SPLITS
            else entry["in_source_test_no_overlap"]
        )
    ]
    if not entries:
        raise ManifestError(f"panel {panel!r} is empty")

    pieces: list[Dataset] = []
    song_ids: list[str] = []
    for source_split in SOURCE_SPLITS:
        split_entries = [
            entry for entry in entries if entry["source_split"] == source_split
        ]
        if not split_entries:
            continue
        pieces.append(
            source[source_split].select(
                [entry["source_index"] for entry in split_entries]
            )
        )
        song_ids.extend(entry["song_id"] for entry in split_entries)

    if not pieces:
        raise ManifestError(f"panel {panel!r} has no materializable rows")
    dataset = pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces)
    return dataset, song_ids
