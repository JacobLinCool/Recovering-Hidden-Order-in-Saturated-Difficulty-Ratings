#!/usr/bin/env python
"""Build a verified, chart-only dataset snapshot for an offline GPU worker."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from datasets import DatasetDict

from TaikoChartEstimator.research.data import (
    DIFFICULTIES,
    SNAPSHOT_MANIFEST_NAME,
    SNAPSHOT_SCHEMA_VERSION,
    load_manifest,
    load_source_dataset,
    validate_dataset_snapshot,
    validate_source_against_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/icassp2027_core/data/dataset_manifest.json"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/icassp2027_core/remote/minimal_dataset_snapshot"
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "experiments/icassp2027_core/remote/"
            "taiko_icassp2027_dataset_snapshot.tar.gz"
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != SNAPSHOT_MANIFEST_NAME
    ]


def build_snapshot(
    manifest_path: Path,
    output: Path,
    *,
    cache_dir: str | None = None,
    source: DatasetDict | None = None,
) -> dict[str, Any]:
    """Create an immutable DatasetDict containing only model input columns."""

    manifest = load_manifest(manifest_path)
    source = (
        source
        if source is not None
        else load_source_dataset(manifest["dataset_name"], cache_dir)
    )
    validate_source_against_manifest(source, manifest)
    actual_fingerprints = {
        split: str(source[split]._fingerprint) for split in ("train", "test")
    }
    if actual_fingerprints != manifest["source_fingerprints"]:
        raise ValueError(
            "source dataset fingerprints differ from the frozen manifest"
        )

    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dataset snapshot: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        minimal = DatasetDict(
            {
                split: source[split].select_columns(list(DIFFICULTIES))
                for split in ("train", "test")
            }
        )
        minimal.save_to_disk(str(temporary))
        metadata = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "dataset_name": manifest["dataset_name"],
            "source_fingerprints": manifest["source_fingerprints"],
            "source_split_rows": {
                split: len(source[split]) for split in ("train", "test")
            },
            "columns": list(DIFFICULTIES),
            "files": snapshot_file_rows(temporary),
        }
        (temporary / SNAPSHOT_MANIFEST_NAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        validate_dataset_snapshot(
            temporary,
            expected_dataset_name=manifest["dataset_name"],
            expected_source_fingerprints=manifest["source_fingerprints"],
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def normalized_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_archive(snapshot: Path, archive: Path) -> dict[str, Any]:
    """Package a verified snapshot as a deterministic gzip tar archive."""

    snapshot = snapshot.resolve()
    metadata = validate_dataset_snapshot(snapshot)
    paths = sorted(path for path in snapshot.rglob("*") if path.is_file())
    archive = archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(f"{archive.suffix}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                for path in paths:
                    content = path.read_bytes()
                    relative = path.relative_to(snapshot).as_posix()
                    tar.addfile(
                        normalized_tar_info(relative, len(content)),
                        io.BytesIO(content),
                    )
    temporary.replace(archive)
    digest = file_sha256(archive)
    checksum = archive.with_suffix(f"{archive.suffix}.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {
        "schema_version": metadata["schema_version"],
        "archive": str(archive),
        "archive_sha256": digest,
        "checksum": str(checksum),
        "size_bytes": archive.stat().st_size,
        "snapshot_files": len(paths),
    }


def main() -> None:
    args = parse_args()
    build_snapshot(
        args.manifest,
        args.output,
        cache_dir=args.cache_dir,
    )
    result = build_archive(args.output, args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
