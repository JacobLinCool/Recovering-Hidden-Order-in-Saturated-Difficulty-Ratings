"""Fail-closed contracts for the Hidden Order release boundary."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence import file_sha256

GLOBAL_EXPERIMENT_ID = "icassp2027_topcoded_oof_v1"
GLOBAL_RELEASE_SCHEMA = "topcoded_oof_release_manifest"
GLOBAL_CONDITIONS = (
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
    "ordinal_logit",
    "ordinal_probit",
    "ordinal_logit_reduction",
)
GLOBAL_SCENARIOS = ("native", "cap7", "cap8", "cap9")
GLOBAL_CHART_COUNT = 1_023
GLOBAL_TITLE_GROUP_COUNT = 1_019
GLOBAL_OUTER_FOLDS = 5
GLOBAL_SCORE_ROW_COUNT = (
    len(GLOBAL_SCENARIOS) * len(GLOBAL_CONDITIONS) * GLOBAL_CHART_COUNT
)

SEQUENCE_EXPERIMENT_ID = "icassp2027_sequence_oof"
SEQUENCE_RELEASE_SCHEMA = "sequence_oof_release_manifest"
SEQUENCE_FOLD_RUN_COUNT = 80
SEQUENCE_SEED_SCORE_ROW_COUNT = 16_368
SEQUENCE_ENSEMBLE_ROW_COUNT = 8_184
SEQUENCE_METRIC_ROW_COUNT = 24

STRUCTURED_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".tsv"})
_SENSITIVE_KEY = re.compile(r"(?:^|_)player(?:_|$)", re.IGNORECASE)
_SENSITIVE_PATH = re.compile(
    r"(?:^|[/_.-])player(?:[/_.-]|$)|donder",
    re.IGNORECASE,
)
_PATH_SUFFIXES = (
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".tsv",
)


@dataclass(frozen=True)
class GlobalReleaseInputs:
    """Verified frozen inputs for player-free global analysis."""

    manifest_path: Path
    fold_assignments: Path
    model_scores: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class SequenceReleaseInputs:
    """Verified frozen inputs for player-free sequence analysis."""

    manifest_path: Path
    fold_runs: Path
    model_scores: Path
    ensemble_scores: Path
    sequence_metrics: Path
    manifest: Mapping[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _safe_repository_path(value: Any, *, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be repository-relative: {value!r}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"{label} file is missing: {value}")
    return path


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def validate_global_release_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path = ".",
) -> GlobalReleaseInputs:
    """Verify the exact frozen OOF inputs allowed into the release analysis."""

    path = Path(manifest_path)
    manifest = _read_json_object(path)
    expected_top_level = {
        "schema_version",
        "experiment_id",
        "source_kind",
        "contract",
        "files",
    }
    if set(manifest) != expected_top_level:
        raise ValueError(
            "global release manifest has unexpected fields: "
            f"{sorted(set(manifest).symmetric_difference(expected_top_level))}"
        )
    if manifest["schema_version"] != GLOBAL_RELEASE_SCHEMA:
        raise ValueError("global release manifest has the wrong schema")
    if manifest["experiment_id"] != GLOBAL_EXPERIMENT_ID:
        raise ValueError("global release manifest has the wrong experiment")
    if manifest["source_kind"] != "frozen_title_grouped_out_of_fold_predictions":
        raise ValueError("global release manifest has the wrong source boundary")

    contract = manifest["contract"]
    expected_contract = {
        "chart_count": GLOBAL_CHART_COUNT,
        "title_group_count": GLOBAL_TITLE_GROUP_COUNT,
        "outer_folds": GLOBAL_OUTER_FOLDS,
        "conditions": list(GLOBAL_CONDITIONS),
        "scenarios": list(GLOBAL_SCENARIOS),
    }
    if contract != expected_contract:
        raise ValueError("global release manifest contract drifted")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        "fold_assignments",
        "model_scores",
    }:
        raise ValueError("global release manifest must name exactly two OOF inputs")
    expected_rows = {
        "fold_assignments": GLOBAL_CHART_COUNT,
        "model_scores": GLOBAL_SCORE_ROW_COUNT,
    }
    root = Path(repository_root).resolve()
    resolved: dict[str, Path] = {}
    for name, expected_row_count in expected_rows.items():
        record = files[name]
        if not isinstance(record, dict) or set(record) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise ValueError(f"global release {name} record has unexpected fields")
        candidate = _safe_repository_path(record["path"], root=root, label=name)
        if record["row_count"] != expected_row_count:
            raise ValueError(f"global release {name} declares the wrong row count")
        if _line_count(candidate) != expected_row_count:
            raise ValueError(f"global release {name} has the wrong row count")
        if record["sha256"] != file_sha256(candidate):
            raise ValueError(f"global release {name} hash does not reproduce")
        resolved[name] = candidate

    return GlobalReleaseInputs(
        manifest_path=path,
        fold_assignments=resolved["fold_assignments"],
        model_scores=resolved["model_scores"],
        manifest=manifest,
    )


def validate_sequence_release_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path = ".",
) -> SequenceReleaseInputs:
    """Verify the exact frozen sequence OOF inputs allowed into the release."""

    path = Path(manifest_path)
    manifest = _read_json_object(path)
    expected_top_level = {
        "schema_version",
        "experiment_id",
        "source_kind",
        "contract",
        "files",
    }
    if set(manifest) != expected_top_level:
        raise ValueError(
            "sequence release manifest has unexpected fields: "
            f"{sorted(set(manifest).symmetric_difference(expected_top_level))}"
        )
    if manifest["schema_version"] != SEQUENCE_RELEASE_SCHEMA:
        raise ValueError("sequence release manifest has the wrong schema")
    if manifest["experiment_id"] != SEQUENCE_EXPERIMENT_ID:
        raise ValueError("sequence release manifest has the wrong experiment")
    if manifest["source_kind"] != "frozen_title_grouped_sequence_oof_predictions":
        raise ValueError("sequence release manifest has the wrong source boundary")

    expected_rows = {
        "fold_runs": SEQUENCE_FOLD_RUN_COUNT,
        "model_scores": SEQUENCE_SEED_SCORE_ROW_COUNT,
        "ensemble_scores": SEQUENCE_ENSEMBLE_ROW_COUNT,
        "sequence_metrics": SEQUENCE_METRIC_ROW_COUNT,
    }
    expected_contract = {
        "fold_run_count": SEQUENCE_FOLD_RUN_COUNT,
        "seed_oof_row_count": SEQUENCE_SEED_SCORE_ROW_COUNT,
        "ensemble_row_count": SEQUENCE_ENSEMBLE_ROW_COUNT,
        "metric_row_count": SEQUENCE_METRIC_ROW_COUNT,
    }
    if manifest["contract"] != expected_contract:
        raise ValueError("sequence release manifest contract drifted")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(expected_rows):
        raise ValueError("sequence release manifest must name exactly four OOF inputs")
    root = Path(repository_root).resolve()
    resolved: dict[str, Path] = {}
    for name, expected_row_count in expected_rows.items():
        record = files[name]
        if not isinstance(record, dict) or set(record) != {
            "path",
            "row_count",
            "sha256",
        }:
            raise ValueError(f"sequence release {name} record has unexpected fields")
        candidate = _safe_repository_path(record["path"], root=root, label=name)
        if record["row_count"] != expected_row_count:
            raise ValueError(f"sequence release {name} declares the wrong row count")
        if _line_count(candidate) != expected_row_count:
            raise ValueError(f"sequence release {name} has the wrong row count")
        if record["sha256"] != file_sha256(candidate):
            raise ValueError(f"sequence release {name} hash does not reproduce")
        resolved[name] = candidate

    return SequenceReleaseInputs(
        manifest_path=path,
        fold_runs=resolved["fold_runs"],
        model_scores=resolved["model_scores"],
        ensemble_scores=resolved["ensemble_scores"],
        sequence_metrics=resolved["sequence_metrics"],
        manifest=manifest,
    )


def _looks_like_path(value: str) -> bool:
    lowered = value.lower()
    return "/" in value or "\\" in value or lowered.endswith(_PATH_SUFFIXES)


def _structured_values(path: Path, text: str) -> list[Any]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix == ".json":
        return [json.loads(text)]
    delimiter = "," if path.suffix == ".csv" else "\t"
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def _sensitive_structured_location(value: Any, *, location: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY.search(key_text):
                return f"{location}.{key_text}"
            if _looks_like_path(key_text) and _SENSITIVE_PATH.search(key_text):
                return f"{location}.{key_text}"
            found = _sensitive_structured_location(
                child,
                location=f"{location}.{key_text}",
            )
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _sensitive_structured_location(
                child,
                location=f"{location}[{index}]",
            )
            if found is not None:
                return found
    elif isinstance(value, str) and _looks_like_path(value) and _SENSITIVE_PATH.search(value):
        return location
    return None


def public_release_findings(paths: Sequence[Path]) -> list[dict[str, str]]:
    """Find forbidden structured keys or source paths in public release files.

    Natural-language privacy statements are intentionally allowed. Only
    structured field names and path-like strings cross this gate.
    """

    findings: list[dict[str, str]] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        if path.suffix in STRUCTURED_SUFFIXES:
            for value in _structured_values(path, text):
                location = _sensitive_structured_location(value)
                if location is not None:
                    findings.append(
                        {
                            "file": path.as_posix(),
                            "reason": f"sensitive_structured_value:{location}",
                        }
                    )
                    break
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _looks_like_path(line) and _SENSITIVE_PATH.search(line):
                findings.append(
                    {
                        "file": path.as_posix(),
                        "reason": f"sensitive_path:line-{line_number}",
                    }
                )
                break
    return findings
