"""Append-only raw-run evidence helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files_snapshot(
    paths: tuple[str, ...],
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Hash an explicit, ordered research-source contract."""

    if not paths:
        raise ValueError("paths must not be empty")
    root_path = Path(root).resolve()
    if len(paths) != len(set(paths)):
        raise ValueError("source snapshot paths must be unique")
    inventory: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for relative in sorted(paths):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"source path must be repository-relative: {relative}")
        path = root_path / candidate
        if not path.is_file():
            raise FileNotFoundError(f"required source file is missing: {relative}")
        digest = file_sha256(path)
        inventory[candidate.as_posix()] = digest
        aggregate.update(candidate.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": "explicit_research_source",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": inventory,
    }


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def atomic_write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def atomic_write_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(target)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "datasets"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "research_source": research_source_snapshot(),
    }


def research_source_snapshot(root: str | Path = ".") -> dict[str, Any]:
    """Hash the exact research source tree used by a run.

    A Git commit alone is insufficient when experiments intentionally run from
    a dirty research worktree. The per-file inventory makes that state
    independently auditable without embedding data or credentials.
    """

    root_path = Path(root).resolve()
    patterns = (
        "TaikoChartEstimator/**/*.py",
        "scripts/research/**/*.py",
        "experiments/*/configs/*.json",
        "pyproject.toml",
        "uv.lock",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in root_path.glob(pattern)
            if path.is_file()
        },
        key=lambda path: path.relative_to(root_path).as_posix(),
    )
    if not paths:
        raise RuntimeError(f"no research source files found under {root_path}")

    inventory: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root_path).as_posix()
        digest = file_sha256(path)
        inventory[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": "research_source_v1",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": inventory,
    }


def create_attempt_directory(
    root: str | Path,
    condition: str,
    seed: int,
) -> Path:
    """Create a unique directory; never reuse or overwrite an earlier attempt."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = (
        Path(root)
        / condition
        / f"seed-{seed}"
        / f"attempt-{stamp}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def start_run(
    attempt: Path,
    *,
    experiment_id: str,
    condition: str,
    seed: int,
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    scenario: str | None = None,
    outer_fold: int | None = None,
) -> dict[str, Any]:
    run_manifest = {
        "run_schema_version": "raw_run_v2",
        "experiment_id": experiment_id,
        "run_id": f"{condition}/seed-{seed}/{attempt.name}",
        "attempt_id": attempt.name,
        "condition": condition,
        "seed": seed,
        "started_at_utc": utc_now(),
        "status": "running",
        "execution_command": [sys.executable, *sys.argv],
        "inputs": dict(inputs),
        "parameters": dict(parameters),
        "environment": environment_snapshot(),
    }
    if scenario is not None:
        if not scenario.strip():
            raise ValueError("scenario must be non-empty when provided")
        run_manifest["scenario"] = scenario
    if outer_fold is not None:
        if (
            isinstance(outer_fold, bool)
            or not isinstance(outer_fold, Integral)
            or outer_fold < 0
        ):
            raise ValueError("outer_fold must be a non-negative integer")
        run_manifest["outer_fold"] = int(outer_fold)
    atomic_write_json(attempt / "run_manifest.json", run_manifest)
    atomic_write_json(
        attempt / "status.json",
        {"status": "running", "updated_at_utc": utc_now()},
    )
    return run_manifest


def finish_run(
    attempt: Path,
    run_manifest: dict[str, Any],
    *,
    status: str,
    outputs: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if status not in {"success", "failed"}:
        raise ValueError(f"invalid terminal status: {status}")
    terminal = {
        "status": status,
        "updated_at_utc": utc_now(),
    }
    if error is not None:
        terminal["error"] = error
    atomic_write_json(attempt / "status.json", terminal)
    run_manifest = dict(run_manifest)
    run_manifest.update(
        {
            "status": status,
            "completed_at_utc": terminal["updated_at_utc"],
            "outputs": dict(outputs or {}),
        }
    )
    if error is not None:
        run_manifest["error"] = error
    atomic_write_json(attempt / "run_manifest.json", run_manifest)
