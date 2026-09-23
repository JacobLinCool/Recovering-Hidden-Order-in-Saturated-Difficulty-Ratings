#!/usr/bin/env python3
"""Verify the self-contained, public evidence release for the paper.

This gate deliberately reads no private chart dataset, title-bearing canonical
table, historical GPU log, or network resource. It checks what a reviewer can
actually obtain from this repository; it does not claim to retrain the models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARTIFACT = Path("paper/manuscript/artifact/public")
PAPER = Path("paper/manuscript")
MANIFEST_SCHEMA = "icassp2027_public_artifact_v2"
CHART_COUNT = 1_023
FOLD_COUNT = 5
DOJO_PLACEMENT_COUNT = 47
REQUIRED_FILES = frozenset(
    {
        "EXPERIMENT_SPEC.md",
        "README.md",
        "REPRODUCE.md",
        "dojo_DATA.md",
        "dojo_config.json",
        "dojo_criterion_summary.jsonl",
        "dojo_leave_one_year_out.jsonl",
        "dojo_per_year_metrics.jsonl",
        "dojo_public_matches.jsonl",
        "dojo_source_manifest.json",
        "global_hidden_tail_contrasts.jsonl",
        "global_hidden_tail_metrics.jsonl",
        "global_oof_scores.jsonl",
        "global_primary_config.json",
        "paper_evidence.json",
        "privacy_report.json",
        "reproducibility_inventory.json",
        "sequence_fold_runs.jsonl",
        "sequence_hidden_tail_contrasts.jsonl",
        "sequence_hidden_tail_metrics.jsonl",
        "sequence_metrics.jsonl",
        "sequence_oof_scores.jsonl",
        "sequence_primary_config.json",
        "sequence_representation_contrasts.jsonl",
    }
)
CHART_KEY = re.compile(r"chart-[0-9a-f]{20}\Z")
PRIVATE_PATH = re.compile(r"/Users/|/home/|/root/|file://|[A-Za-z]:\\Users\\", re.I)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----")
SECRET_ASSIGNMENT = re.compile(r"(?:api[_-]?key|access[_-]?token|secret)[\"' ]*[:=]", re.I)
FORBIDDEN_FIELDS = frozenset(
    {"title", "normalized_title", "chart_id", "song_id", "player", "player_id", "user_id", "account_id"}
)


class ReleaseError(ValueError):
    """A public release contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid JSON: {path}: {exc}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                require(bool(line.strip()), f"blank JSONL line: {path}:{number}")
                value = json.loads(line)
                require(isinstance(value, dict), f"non-object JSONL row: {path}:{number}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid JSONL: {path}: {exc}") from exc
    return rows


def check_private_content(value: Any, location: str = "$") -> None:
    """Reject identifying structured fields while allowing privacy prose."""

    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"non-string field at {location}")
            require(key.lower() not in FORBIDDEN_FIELDS, f"private field: {location}.{key}")
            check_private_content(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_private_content(child, f"{location}[{index}]")
    elif isinstance(value, str):
        require(not PRIVATE_PATH.search(value), f"private path at {location}")
        require(not PRIVATE_KEY.search(value), f"private key at {location}")
        require(not SECRET_ASSIGNMENT.search(value), f"secret assignment at {location}")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite number at {location}")


def verify_manifest(directory: Path) -> int:
    manifest = read_json(directory / "MANIFEST.json")
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "wrong public manifest schema")
    entries = manifest.get("files")
    require(isinstance(entries, dict), "manifest files must be an object")
    require(REQUIRED_FILES <= set(entries), "public manifest omits a required evidence file")
    require(
        all(isinstance(name, str) and name == Path(name).name and name != "MANIFEST.json" for name in entries),
        "public manifest contains an unsafe file name",
    )
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    require(actual == set(entries) | {"MANIFEST.json"}, "public directory has missing or extra files")
    require(
        all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
        "public directory contains a directory or symlink",
    )
    for name in sorted(entries):
        path = directory / name
        entry = entries[name]
        require(isinstance(entry, dict), f"invalid manifest entry: {name}")
        require(entry.get("size_bytes") == path.stat().st_size, f"size mismatch: {name}")
        require(entry.get("sha256") == sha256(path), f"SHA-256 mismatch: {name}")
        text = path.read_text(encoding="utf-8")
        require(not PRIVATE_PATH.search(text), f"private path in {name}")
        require(not PRIVATE_KEY.search(text), f"private key in {name}")
        require(not SECRET_ASSIGNMENT.search(text), f"secret assignment in {name}")
        if path.suffix == ".json":
            check_private_content(read_json(path), name)
        elif path.suffix == ".jsonl":
            for number, row in enumerate(read_jsonl(path), 1):
                check_private_content(row, f"{name}:{number}")
    privacy = read_json(directory / "privacy_report.json")
    require(privacy.get("status") == "pass", "privacy report did not pass")
    require(privacy.get("forbidden_findings") == 0, "privacy report has findings")
    return len(entries) + 1


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def verify_scores(directory: Path) -> tuple[dict[str, tuple[float, int]], dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, int]]:
    global_config = read_json(directory / "global_primary_config.json")
    sequence_config = read_json(directory / "sequence_primary_config.json")
    conditions = global_config.get("conditions")
    scenarios = global_config.get("scenarios")
    matrix = sequence_config.get("matrix")
    require(isinstance(conditions, list) and len(conditions) == 7 and len(set(conditions)) == 7, "global conditions changed")
    require(isinstance(scenarios, dict) and {name: spec["cap"] for name, spec in scenarios.items()} == {"native": 10, "cap7": 7, "cap8": 8, "cap9": 9}, "global scenarios changed")
    require(global_config.get("outer_folds") == FOLD_COUNT == sequence_config.get("outer_folds"), "outer folds changed")
    require(isinstance(matrix, list) and len(matrix) == 8, "sequence matrix changed")
    planned = {(item["scenario"], item["condition"]): tuple(item["seeds"]) for item in matrix}
    require(len(planned) == len(matrix), "duplicate sequence matrix condition")
    require(all(name in scenarios and seeds and len(seeds) == len(set(seeds)) for (name, _), seeds in planned.items()), "invalid sequence matrix entry")

    reference: dict[str, tuple[float, int]] = {}
    global_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    global_groups: Counter[tuple[str, str]] = Counter()
    for row in read_jsonl(directory / "global_oof_scores.jsonl"):
        require(row.get("schema_version") == "public_global_oof_score", "global score schema changed")
        scenario, condition, key = row["scenario"], row["condition"], row["chart_key"]
        require(scenario in scenarios and condition in conditions, "unexpected global score condition")
        require(isinstance(key, str) and CHART_KEY.fullmatch(key) is not None, "invalid global chart key")
        identity = scenario, condition, key
        require(identity not in global_index, "duplicate global OOF score")
        global_index[identity] = row
        global_groups[scenario, condition] += 1
        label = finite_number(row["evaluation_original_label"], "global original label")
        observed = finite_number(row["observed_label"], "global observed label")
        fold = row["outer_fold"]
        require(label.is_integer() and 1 <= label <= 10, "invalid global original label")
        require(isinstance(fold, int) and not isinstance(fold, bool) and 0 <= fold < FOLD_COUNT, "invalid global fold")
        require(observed == min(label, scenarios[scenario]["cap"]), "incorrect global top coding")
        finite_number(row["latent_score"], "global latent score")
        finite_number(row["expected_observed_label"], "global expected label")
        if key in reference:
            require(reference[key] == (label, fold), "global chart label or fold changed across conditions")
        else:
            reference[key] = label, fold
    require(len(reference) == CHART_COUNT, "global chart count changed")
    require(set(global_groups) == {(s, c) for s in scenarios for c in conditions}, "missing global scenario/condition")
    require(all(count == CHART_COUNT for count in global_groups.values()), "incomplete global OOF coverage")

    sequence_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    sequence_groups: Counter[tuple[str, str]] = Counter()
    for row in read_jsonl(directory / "sequence_oof_scores.jsonl"):
        require(row.get("schema_version") == "public_sequence_oof_score", "sequence score schema changed")
        scenario, condition, key = row["scenario"], row["condition"], row["chart_key"]
        require((scenario, condition) in planned, "unexpected sequence score condition")
        require(isinstance(key, str) and CHART_KEY.fullmatch(key) is not None, "invalid sequence chart key")
        identity = scenario, condition, key
        require(identity not in sequence_index, "duplicate sequence OOF score")
        sequence_index[identity] = row
        sequence_groups[scenario, condition] += 1
        label = finite_number(row["evaluation_original_label"], "sequence original label")
        observed = finite_number(row["observed_label"], "sequence observed label")
        require(key in reference and reference[key] == (label, row["outer_fold"]), "sequence/global chart alignment changed")
        require(observed == min(label, scenarios[scenario]["cap"]), "incorrect sequence top coding")
        require(row["seeds"] == list(planned[scenario, condition]), "sequence OOF seed set changed")
        finite_number(row["latent_score"], "sequence latent score")
        finite_number(row["expected_observed_label"], "sequence expected label")
        require(finite_number(row["latent_score_seed_sd"], "sequence seed SD") >= 0, "negative sequence seed SD")
    require(set(sequence_groups) == set(planned), "missing sequence condition")
    require(all(count == CHART_COUNT for count in sequence_groups.values()), "incomplete sequence OOF coverage")

    runs = read_jsonl(directory / "sequence_fold_runs.jsonl")
    expected_runs = {(scenario, condition, seed, fold) for (scenario, condition), seeds in planned.items() for seed in seeds for fold in range(FOLD_COUNT)}
    actual_runs = [(row["scenario"], row["condition"], row["seed"], row["outer_fold"]) for row in runs]
    require(len(actual_runs) == len(expected_runs) == 80 and set(actual_runs) == expected_runs, "sequence fold run coverage changed")
    require(all(row.get("status") == "success" for row in runs), "sequence fold has non-success status")
    return reference, global_index, sequence_index, {"global_scores": len(global_index), "sequence_scores": len(sequence_index), "sequence_fold_runs": len(runs)}


def verify_dojo(directory: Path, reference: dict[str, tuple[float, int]], global_index: dict[tuple[str, str, str], dict[str, Any]], sequence_index: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, int]:
    config = read_json(directory / "dojo_config.json")
    families = {"global": (config["primary_global_condition"], global_index), "sequence": (config["replication_sequence_condition"], sequence_index)}
    rows = read_jsonl(directory / "dojo_public_matches.jsonl")
    identities: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    years: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        require(row.get("schema_version") == "public_dojo_match_v1", "Dojo match schema changed")
        family = row["family"]
        require(family in families, "unexpected Dojo family")
        condition, scores = families[family]
        require(row["condition"] == condition, "Dojo condition differs from config")
        key = row["chart_key"]
        require(key in reference and reference[key][0] == 10, "Dojo chart is outside native star-10 panel")
        require(row["official_star"] == 10 and row["outer_fold"] == reference[key][1], "Dojo label or fold mismatch")
        score = scores.get(("native", condition, key))
        require(score is not None, "Dojo score missing from native OOF")
        require(math.isclose(finite_number(row["latent_score"], "Dojo score"), finite_number(score["latent_score"], "native OOF score"), rel_tol=1e-10, abs_tol=1e-10), "Dojo score differs from native OOF")
        identity = (row["year"], row["dan"], row["position"], key)
        require(identity not in identities[family], "duplicate Dojo placement")
        identities[family].add(identity)
        years[family].add(row["year"])
    expected_years = set(range(2020, 2026))
    require(set(identities) == set(families), "missing Dojo model family")
    require(all(len(items) == DOJO_PLACEMENT_COUNT for items in identities.values()), "Dojo placement count changed")
    require(identities["global"] == identities["sequence"], "Dojo families use different placements")
    require(all(items == expected_years for items in years.values()), "Dojo years changed")
    summary = read_jsonl(directory / "dojo_criterion_summary.jsonl")
    require(len(summary) == 2 and {row["family"] for row in summary} == set(families), "Dojo summary coverage changed")
    require(all(row["placement_count"] == DOJO_PLACEMENT_COUNT and row["year_count"] == 6 for row in summary), "Dojo summary counts changed")
    for name, year_field in (("dojo_per_year_metrics.jsonl", "year"), ("dojo_leave_one_year_out.jsonl", "held_out_year")):
        year_rows = read_jsonl(directory / name)
        require(len(year_rows) == 12, f"{name} row count changed")
        require({(row["family"], row[year_field]) for row in year_rows} == {(family, year) for family in families for year in expected_years}, f"{name} year coverage changed")
    return {"dojo_placements_per_family": DOJO_PLACEMENT_COUNT}


def verify_paper_assets(root: Path, directory: Path) -> None:
    paper = root / PAPER
    required = (
        "main.tex", "main.pdf", "references.bib", "spconf.sty", "IEEEbib.bst",
        "figures/hidden_order_protocol.tex",
        "generated/hidden_order_numbers.tex", "generated/hidden_order_table.tex",
        "generated/ceiling_compression_numbers.tex", "generated/hidden_order_protocol.pdf",
        "generated/ceiling_compression.pdf", "generated/dojo_criterion.pdf",
        "generated/paper_evidence.json",
    )
    for name in required:
        path = paper / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, f"missing paper asset: {name}")
    for name in ("main.pdf", "generated/hidden_order_protocol.pdf", "generated/ceiling_compression.pdf", "generated/dojo_criterion.pdf"):
        with (paper / name).open("rb") as stream:
            require(stream.read(5) == b"%PDF-", f"invalid PDF header: {name}")
    evidence = read_json(directory / "paper_evidence.json")
    require((paper / "generated/paper_evidence.json").read_bytes() == (directory / "paper_evidence.json").read_bytes(), "paper and public evidence JSON differ")
    require(evidence.get("schema_version") == "icassp2027_paper_evidence_v2", "paper evidence schema changed")
    require(evidence["global"]["chart_count"] == CHART_COUNT, "paper chart count changed")
    require(evidence["dojo"]["placement_count"] == DOJO_PLACEMENT_COUNT, "paper Dojo count changed")
    generated = evidence.get("generated_files", {})
    for name in ("hidden_order_numbers.tex", "hidden_order_table.tex"):
        require(generated.get(name) == sha256(paper / "generated" / name), f"paper generated hash mismatch: {name}")
    tex = (paper / "main.tex").read_text(encoding="utf-8")
    for name in ("generated/hidden_order_numbers.tex", "generated/ceiling_compression_numbers.tex", "generated/hidden_order_table.tex"):
        require(f"\\input{{{name}}}" in tex, f"manuscript does not input {name}")
    for name in ("generated/hidden_order_protocol.pdf", "generated/ceiling_compression.pdf", "generated/dojo_criterion.pdf"):
        require(f"\\includegraphics[width=\\columnwidth]{{{name}}}" in tex, f"manuscript does not include {name}")


def verify_release(root: Path) -> dict[str, int]:
    directory = root / ARTIFACT
    require(directory.is_dir(), f"missing public artifact: {directory}")
    file_count = verify_manifest(directory)
    reference, global_scores, sequence_scores, counts = verify_scores(directory)
    counts.update(verify_dojo(directory, reference, global_scores, sequence_scores))
    verify_paper_assets(root, directory)
    return {"public_files": file_count, "charts": len(reference), **counts}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        summary = verify_release(args.root)
    except (ReleaseError, KeyError, TypeError, OSError, UnicodeError) as exc:
        print(f"public release verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "pass", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
