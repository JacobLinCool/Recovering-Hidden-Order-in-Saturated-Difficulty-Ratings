#!/usr/bin/env python
"""Snapshot public Dan-i Dojo course metadata with explicit provenance."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from TaikoChartEstimator.research.data import normalize_title
from TaikoChartEstimator.research.dojo import (
    DOJO_SCHEMA_VERSION,
    REGULAR_DAN_ORDER,
    DojoDataError,
    validate_course_placements,
)
from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    file_sha256,
)

API_ROOT = "https://taiko.wiki/api/v1"
SITE_ROOT = "https://taiko.wiki"
TAIKO_WIKI_SOURCE_COMMIT = "84ecdd69438a054fdf6a3ba7a33194089327f58a"
TAIKO_WIKI_LICENSE_URL = (
    "https://github.com/taikowiki/taikowiki/blob/"
    f"{TAIKO_WIKI_SOURCE_COMMIT}/LICENSE"
)
OFFICIAL_CONTEXT_URLS = {
    "20": [
        "https://taiko-ch.net/blog/?m=20200623",
        "https://taiko-ch.net/blog/?p=5102",
        "https://taiko-ch.net/blog/?m=20201119",
    ],
    "21": [
        "https://taiko-ch.net/blog/?p=5623",
        "https://taiko-ch.net/blog/?m=202108",
    ],
    "22": [
        "https://taiko-ch.net/blog/?m=202205",
        "https://taiko-ch.net/blog/?m=202208",
    ],
    "23": [
        "https://taiko-ch.net/blog/?m=20230914",
    ],
    "24": [
        "https://taiko-ch.net/blog/?m=20240523",
        "https://taiko-ch.net/blog/?m=202409",
    ],
    "25": [
        "https://taiko-ch.net/blog/?p=14433",
        "https://taiko-ch.net/blog/?m=202509",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/icassp2027_dojo/config.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/icassp2027_dojo/data"),
    )
    parser.add_argument(
        "--retrieved-at-utc",
        help="Optional ISO timestamp for a reproducible fixture or replay.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DojoDataError(f"{path} must contain a JSON object")
    return value


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TaikoChartEstimator-research/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise DojoDataError(f"HTTP {response.status} for {url}")
        content_type = response.headers.get_content_type()
        if content_type != "application/json":
            raise DojoDataError(f"expected JSON from {url}, received {content_type}")
        return response.read()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _decode_json(payload: bytes, *, source: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DojoDataError(f"invalid JSON from {source}: {error}") from error


def _course_level(song: dict[str, Any], difficulty: str) -> int:
    courses = song.get("courses")
    if not isinstance(courses, dict):
        raise DojoDataError(f"song {song.get('songNo')} has no course mapping")
    course = courses.get(difficulty)
    if not isinstance(course, dict):
        raise DojoDataError(
            f"song {song.get('songNo')} has no {difficulty!r} course"
        )
    level = course.get("level")
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        raise DojoDataError(f"song {song.get('songNo')} has an invalid level")
    integer = int(level)
    if float(level) != integer or not 1 <= integer <= 10:
        raise DojoDataError(f"song {song.get('songNo')} has an invalid star level")
    return integer


def _validate_song_dani_reference(
    song: dict[str, Any],
    *,
    version: str,
    dan: str,
    position: int,
    difficulty: str,
) -> None:
    course = song["courses"][difficulty]
    references = course.get("dani")
    expected = {
        "version": version,
        "dan": dan,
        "order": position,
    }
    if not isinstance(references, list) or not any(
        all(reference.get(key) == value for key, value in expected.items())
        for reference in references
        if isinstance(reference, dict)
    ):
        raise DojoDataError(
            "course table and song-level Dan-i reference disagree: "
            f"version={version}, dan={dan}, position={position}, "
            f"song={song.get('songNo')}"
        )


def build_placements(
    version_documents: dict[str, dict[str, Any]],
    songs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    songs_by_number: dict[str, dict[str, Any]] = {}
    for song in songs:
        song_number = str(song.get("songNo", ""))
        if not song_number or song_number in songs_by_number:
            raise DojoDataError(f"invalid or duplicate song number: {song_number!r}")
        songs_by_number[song_number] = song

    rows: list[dict[str, Any]] = []
    for version in sorted(version_documents, key=int):
        document = version_documents[version]
        if document.get("version") != version or not isinstance(document.get("data"), list):
            raise DojoDataError(f"invalid Dan-i version document: {version}")
        courses = [
            course
            for course in document["data"]
            if isinstance(course, dict) and course.get("dan") in REGULAR_DAN_ORDER
        ]
        if {str(course["dan"]) for course in courses} != set(REGULAR_DAN_ORDER):
            raise DojoDataError(f"version {version} does not contain all regular ranks")
        for course in courses:
            dan = str(course["dan"])
            course_songs = course.get("songs")
            if not isinstance(course_songs, list) or len(course_songs) != 3:
                raise DojoDataError(f"version {version} rank {dan} needs three songs")
            for position, placement in enumerate(course_songs, start=1):
                if not isinstance(placement, dict):
                    raise DojoDataError("course song entry must be an object")
                song_number = str(placement.get("songNo", ""))
                difficulty = str(placement.get("difficulty", ""))
                song = songs_by_number.get(song_number)
                if song is None:
                    raise DojoDataError(f"unknown song number in Dan-i data: {song_number}")
                title = str(song.get("title", "")).strip()
                if not title:
                    raise DojoDataError(f"song {song_number} has no title")
                _validate_song_dani_reference(
                    song,
                    version=version,
                    dan=dan,
                    position=position,
                    difficulty=difficulty,
                )
                rows.append(
                    {
                        "schema_version": DOJO_SCHEMA_VERSION,
                        "year": 2000 + int(version),
                        "version": version,
                        "dan": dan,
                        "tier_order": REGULAR_DAN_ORDER[dan],
                        "position": position,
                        "song_no": song_number,
                        "title": title,
                        "normalized_title": normalize_title(title),
                        "difficulty": difficulty,
                        "official_star": _course_level(song, difficulty),
                        "source_kind": (
                            "community_transcription_of_official_game_course"
                        ),
                        "source_api_url": f"{API_ROOT}/dani/version/{version}",
                        "source_song_api_url": f"{API_ROOT}/song/no/{song_number}",
                        "source_course_url": f"{SITE_ROOT}/dani/{version}?lang=ja",
                        "official_context_urls": OFFICIAL_CONTEXT_URLS[version],
                    }
                )
    rows.sort(key=lambda row: (row["year"], row["tier_order"], row["position"]))
    validate_course_placements(rows)
    return rows


def main() -> None:
    args = parse_args()
    config = _read_json(args.config)
    versions = [str(version) for version in config["versions"]]
    if set(versions) != set(OFFICIAL_CONTEXT_URLS):
        raise DojoDataError("configured versions lack official context URLs")
    retrieved_at = args.retrieved_at_utc or datetime.now(timezone.utc).isoformat()
    try:
        datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise DojoDataError("retrieved-at timestamp must be ISO 8601") from error

    raw_root = args.output / "raw"
    source_urls = {
        "songs": f"{API_ROOT}/song/all",
        **{f"version_{version}": f"{API_ROOT}/dani/version/{version}" for version in versions},
    }
    raw_paths: dict[str, Path] = {"songs": raw_root / "songs.json"}
    raw_paths.update(
        {f"version_{version}": raw_root / f"dani_{version}.json" for version in versions}
    )
    payloads: dict[str, bytes] = {}
    for name, url in source_urls.items():
        payload = _fetch(url)
        payloads[name] = payload
        _write_bytes(raw_paths[name], payload)

    songs = _decode_json(payloads["songs"], source=source_urls["songs"])
    if not isinstance(songs, list):
        raise DojoDataError("song API must return an array")
    version_documents = {
        version: _decode_json(
            payloads[f"version_{version}"],
            source=source_urls[f"version_{version}"],
        )
        for version in versions
    }
    if any(not isinstance(value, dict) for value in version_documents.values()):
        raise DojoDataError("Dan-i API must return objects")
    placements = build_placements(version_documents, songs)
    expected_total = int(config["expected_total_placements"])
    if len(placements) != expected_total:
        raise DojoDataError(
            f"course snapshot has {len(placements)} placements, expected {expected_total}"
        )

    placement_path = args.output / "course_placements.jsonl"
    atomic_write_jsonl(placement_path, placements)
    manifest = {
        "schema_version": "icassp2027_dojo_source_manifest_v1",
        "retrieved_at_utc": retrieved_at,
        "source_description": (
            "Taiko Wiki public API transcription of official Dan-i Dojo course "
            "facts; annual official Taiko no Tatsujin notices establish edition "
            "and mode context but do not verify every transcribed row"
        ),
        "source_code_commit": TAIKO_WIKI_SOURCE_COMMIT,
        "source_code_license": "MIT",
        "source_code_license_url": TAIKO_WIKI_LICENSE_URL,
        "course_fact_license": "not asserted; factual metadata used with attribution",
        "official_mode_url": (
            "https://taiko.namco-ch.net/taiko/en/special/dani_dojo/index.php"
        ),
        "official_context_urls": OFFICIAL_CONTEXT_URLS,
        "source_files": {
            name: {
                "url": source_urls[name],
                "path": raw_paths[name].as_posix(),
                "sha256": file_sha256(raw_paths[name]),
                "size_bytes": raw_paths[name].stat().st_size,
            }
            for name in sorted(raw_paths)
        },
        "output": {
            "path": placement_path.as_posix(),
            "sha256": file_sha256(placement_path),
            "placement_count": len(placements),
            "year_count": len(versions),
            "rank_count_per_year": len(REGULAR_DAN_ORDER),
            "songs_per_rank": 3,
        },
    }
    atomic_write_json(args.output / "SOURCE_MANIFEST.json", manifest)
    atomic_write_text(
        args.output / "README.md",
        "# Dan-i Dojo course snapshot\n\n"
        "This directory records factual course placements from the public Taiko "
        "Wiki API. The site is a community-maintained transcription of official "
        "game content; it is not a Bandai Namco data release. "
        "`SOURCE_MANIFEST.json` pins response hashes, API URLs, the website "
        "source-code commit/license, and official annual notices used to "
        "corroborate the curriculum context.\n\n"
        "The paper analysis releases only the fields needed to reproduce "
        "aggregate course-placement results. It contains no player or account "
        "data.\n",
    )
    print(json.dumps(manifest["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
