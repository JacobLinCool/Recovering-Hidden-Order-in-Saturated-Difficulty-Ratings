#!/usr/bin/env python
"""Generate the release-safe ICASSP artifact from canonical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    file_sha256,
    source_files_snapshot,
)
from TaikoChartEstimator.research.release import (
    public_release_findings,
    validate_global_release_manifest,
    validate_sequence_release_manifest,
)

PUBLIC_SCHEMA = "icassp2027_public_artifact_v2"
PUBLIC_ARTIFACT_FILES = {
    "EXPERIMENT_SPEC.md",
    "README.md",
    "REPRODUCE.md",
    "global_hidden_tail_contrasts.jsonl",
    "global_hidden_tail_metrics.jsonl",
    "global_oof_scores.jsonl",
    "paper_evidence.json",
    "global_primary_config.json",
    "privacy_report.json",
    "reproducibility_inventory.json",
    "sequence_fold_runs.jsonl",
    "sequence_hidden_tail_contrasts.jsonl",
    "sequence_hidden_tail_metrics.jsonl",
    "sequence_metrics.jsonl",
    "sequence_oof_scores.jsonl",
    "sequence_primary_config.json",
    "sequence_representation_contrasts.jsonl",
    "dojo_DATA.md",
    "dojo_config.json",
    "dojo_criterion_summary.jsonl",
    "dojo_leave_one_year_out.jsonl",
    "dojo_per_year_metrics.jsonl",
    "dojo_public_matches.jsonl",
    "dojo_source_manifest.json",
}
REPRODUCIBILITY_SOURCE_PATHS = tuple(
    sorted(
        {
            "TaikoChartEstimator/research/data.py",
            "TaikoChartEstimator/research/evidence.py",
            "TaikoChartEstimator/research/release.py",
            "TaikoChartEstimator/research/dojo.py",
            "TaikoChartEstimator/research/oof.py",
            "scripts/research/analyze_sequence_oof.py",
            "scripts/research/analyze_topcoded_oof.py",
            "scripts/research/build_public_artifact.py",
            "scripts/research/analyze_dojo_criterion.py",
            "scripts/research/make_hidden_order_paper_artifacts.py",
            "scripts/research/make_hidden_order_protocol_figure.py",
            "scripts/research/make_dojo_criterion_figure.py",
            "scripts/verify_public_release.py",
            "scripts/research/analyze_ceiling_compression.py",
            "paper/icassp2027/figures/hidden_order_protocol.tex",
            "pyproject.toml",
            "uv.lock",
        }
    )
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)[\"' ]*[:=]"),
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/root/"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-canonical",
        type=Path,
        default=Path("experiments/icassp2027_topcoded_oof_v1/canonical"),
    )
    parser.add_argument(
        "--global-release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--global-reports",
        type=Path,
        default=Path("experiments/icassp2027_topcoded_oof_v1/reports"),
    )
    parser.add_argument(
        "--global-config",
        type=Path,
        default=Path(
            "experiments/icassp2027_topcoded_oof_v1/configs/primary.json"
        ),
    )
    parser.add_argument(
        "--sequence-release-manifest",
        type=Path,
        default=Path(
            "experiments/icassp2027_sequence_oof/OOF_RELEASE_MANIFEST.json"
        ),
    )
    parser.add_argument(
        "--sequence-reports",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/reports"),
    )
    parser.add_argument(
        "--sequence-config",
        type=Path,
        default=Path("experiments/icassp2027_sequence_oof/configs/primary.json"),
    )
    parser.add_argument(
        "--dojo-data",
        type=Path,
        default=Path("experiments/icassp2027_dojo/data"),
    )
    parser.add_argument(
        "--dojo-reports",
        type=Path,
        default=Path("experiments/icassp2027_dojo/reports"),
    )
    parser.add_argument(
        "--dojo-config",
        type=Path,
        default=Path("experiments/icassp2027_dojo/config.json"),
    )
    parser.add_argument(
        "--experiment-spec",
        type=Path,
        default=Path("experiments/icassp2027_dojo/PUBLIC_SPEC.md"),
    )
    parser.add_argument(
        "--paper-evidence",
        type=Path,
        default=Path("paper/icassp2027/generated/paper_evidence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/icassp2027/artifact/public"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def publish_paper_evidence(source: Path, output: Path) -> dict[str, Any]:
    """Publish the canonical paper evidence and require byte identity."""

    value = read_json_object(source)
    if value.get("schema_version") != "icassp2027_paper_evidence_v2":
        raise ValueError("paper evidence has an unexpected schema version")
    atomic_write_json(output, value)
    if file_sha256(output) != file_sha256(source):
        raise ValueError("paper evidence source is not canonical JSON")
    return value



def chart_key(chart_id: str) -> str:
    digest = hashlib.sha256(f"chart\0{chart_id}".encode("utf-8")).hexdigest()
    return f"chart-{digest[:20]}"


def global_public_scores(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "schema_version": "public_global_oof_score",
                "scenario": str(row["scenario"]),
                "condition": str(row["condition"]),
                "outer_fold": int(row["outer_fold"]),
                "chart_key": chart_key(str(row["chart_id"])),
                "observed_label": float(row["observed_star"]),
                "evaluation_original_label": float(row["original_star"]),
                "latent_score": float(row["latent_score"]),
                "expected_observed_label": float(row["expected_observed_label"]),
            }
        )
    output.sort(
        key=lambda row: (
            row["scenario"],
            row["condition"],
            row["outer_fold"],
            row["chart_key"],
        )
    )
    return output


def sequence_public_scores(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "schema_version": "public_sequence_oof_score",
                "scenario": str(row["scenario"]),
                "condition": str(row["condition"]),
                "outer_fold": int(row["outer_fold"]),
                "chart_key": chart_key(str(row["chart_id"])),
                "seeds": [int(seed) for seed in row["seeds"]],
                "observed_label": float(row["observed_label"]),
                "evaluation_original_label": float(row["evaluation_original_label"]),
                "latent_score": float(row["latent_score"]),
                "expected_observed_label": float(row["expected_observed_label"]),
                "latent_score_seed_sd": float(row["latent_score_seed_sd"]),
            }
        )
    output.sort(
        key=lambda row: (
            row["scenario"],
            row["condition"],
            row["outer_fold"],
            row["chart_key"],
        )
    )
    return output


def sequence_public_runs(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain complete fold provenance without publishing workspace paths."""

    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "schema_version": "public_sequence_fold_run",
                "scenario": str(row["scenario"]),
                "condition": str(row["condition"]),
                "outer_fold": int(row["outer_fold"]),
                "seed": int(row["seed"]),
                "run_id": str(row["run_id"]),
                "status": "success",
                "device": str(row["device"]),
                "completed_at_utc": str(row["completed_at_utc"]),
                "fold_wall_seconds": float(row["fold_wall_seconds"]),
                "peak_cuda_memory_allocated_bytes": int(
                    row["peak_cuda_memory_allocated_bytes"]
                ),
            }
        )
    output.sort(
        key=lambda row: (
            row["scenario"],
            row["condition"],
            row["seed"],
            row["outer_fold"],
        )
    )
    return output


def dojo_public_matches(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Publish auditable criterion rows without chart titles or source assets."""

    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("high_tier_order") is None or int(row["official_star"]) != 10:
            continue
        output.append(
            {
                "schema_version": "public_dojo_match_v1",
                "family": str(row["family"]),
                "condition": str(row["condition"]),
                "year": int(row["year"]),
                "dan": str(row["dan"]),
                "expert_tier_order": int(row["high_tier_order"]),
                "position": int(row["position"]),
                "official_star": int(row["official_star"]),
                "chart_key": chart_key(str(row["chart_id"])),
                "outer_fold": int(row["outer_fold"]),
                "latent_score": float(row["latent_score"]),
                "match_method": str(row["match_method"]),
                "source_course_url": str(row["source_course_url"]),
            }
        )
    output.sort(
        key=lambda row: (
            row["family"],
            row["year"],
            row["expert_tier_order"],
            row["position"],
            row["chart_key"],
        )
    )
    if len(output) != 94:
        raise ValueError("public Dojo panel requires 47 placements per model family")
    return output



def privacy_scan(output: Path) -> dict[str, Any]:
    """Fail closed on secrets and sensitive keys or paths in a public artifact."""

    files = sorted(path for path in output.rglob("*") if path.is_file())
    findings = public_release_findings(files)
    for path in files:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(output).as_posix()
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    {"file": relative, "reason": f"secret_pattern:{pattern.pattern}"}
                )
    if findings:
        raise RuntimeError(
            "public-artifact boundary scan failed with "
            f"{len(findings)} findings: {findings[:10]}"
        )
    return {
        "schema_version": "public_artifact_privacy_scan",
        "status": "pass",
        "files_scanned": len(files),
        "forbidden_findings": 0,
    }


def main() -> None:
    """Build the public artifact from the explicit non-human evidence boundary."""

    args = parse_args()
    required = (
        args.global_release_manifest,
        args.global_config,
        args.global_reports / "hidden_tail_metrics.jsonl",
        args.global_reports / "hidden_tail_contrasts.jsonl",
        args.sequence_release_manifest,
        args.sequence_config,
        args.sequence_reports / "hidden_tail_metrics.jsonl",
        args.sequence_reports / "hidden_tail_contrasts.jsonl",
        args.sequence_reports / "representation_contrasts.jsonl",
        args.dojo_data / "README.md",
        args.dojo_data / "SOURCE_MANIFEST.json",
        args.dojo_data / "course_placements.jsonl",
        args.dojo_reports / "matched_placements.jsonl",
        args.dojo_reports / "criterion_summary.jsonl",
        args.dojo_reports / "per_year_metrics.jsonl",
        args.dojo_reports / "leave_one_year_out.jsonl",
        args.dojo_reports / "analysis_manifest.json",
        args.dojo_config,
        args.experiment_spec,
        args.paper_evidence,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"canonical public-artifact inputs missing: {missing}")
    global_release = validate_global_release_manifest(args.global_release_manifest)
    sequence_release = validate_sequence_release_manifest(
        args.sequence_release_manifest
    )
    dojo_manifest = read_json_object(args.dojo_reports / "analysis_manifest.json")
    if (
        dojo_manifest.get("schema_version")
        != "icassp2027_dojo_analysis_manifest_v1"
        or int(dojo_manifest.get("primary_panel_identity_count", -1)) != 47
        or int(dojo_manifest.get("bootstrap_replicates", -1)) != 10_000
        or int(dojo_manifest.get("permutation_replicates", -1)) != 10_000
    ):
        raise ValueError("Dojo analysis manifest violates the frozen contract")

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(
        args.output / "global_oof_scores.jsonl",
        global_public_scores(read_jsonl(global_release.model_scores)),
    )
    atomic_write_jsonl(
        args.output / "sequence_oof_scores.jsonl",
        sequence_public_scores(
            read_jsonl(sequence_release.ensemble_scores)
        ),
    )
    atomic_write_jsonl(
        args.output / "sequence_fold_runs.jsonl",
        sequence_public_runs(read_jsonl(sequence_release.fold_runs)),
    )
    atomic_write_jsonl(
        args.output / "dojo_public_matches.jsonl",
        dojo_public_matches(read_jsonl(args.dojo_reports / "matched_placements.jsonl")),
    )
    publish_paper_evidence(args.paper_evidence, args.output / "paper_evidence.json")

    aggregate_sources = {
        "global_hidden_tail_metrics.jsonl": args.global_reports
        / "hidden_tail_metrics.jsonl",
        "global_hidden_tail_contrasts.jsonl": args.global_reports
        / "hidden_tail_contrasts.jsonl",
        "sequence_metrics.jsonl": sequence_release.sequence_metrics,
        "sequence_hidden_tail_metrics.jsonl": args.sequence_reports
        / "hidden_tail_metrics.jsonl",
        "sequence_hidden_tail_contrasts.jsonl": args.sequence_reports
        / "hidden_tail_contrasts.jsonl",
        "sequence_representation_contrasts.jsonl": args.sequence_reports
        / "representation_contrasts.jsonl",
        "dojo_criterion_summary.jsonl": args.dojo_reports
        / "criterion_summary.jsonl",
        "dojo_per_year_metrics.jsonl": args.dojo_reports
        / "per_year_metrics.jsonl",
        "dojo_leave_one_year_out.jsonl": args.dojo_reports
        / "leave_one_year_out.jsonl",
    }
    for name, source in aggregate_sources.items():
        atomic_write_jsonl(args.output / name, read_jsonl(source))
    for name, source in (
        ("EXPERIMENT_SPEC.md", args.experiment_spec),
        ("global_primary_config.json", args.global_config),
        ("sequence_primary_config.json", args.sequence_config),
        ("dojo_config.json", args.dojo_config),
        ("dojo_DATA.md", args.dojo_data / "README.md"),
    ):
        atomic_write_text(args.output / name, source.read_text(encoding="utf-8"))
    atomic_write_json(
        args.output / "dojo_source_manifest.json",
        read_json_object(args.dojo_data / "SOURCE_MANIFEST.json"),
    )

    release_docs = Path("paper/icassp2027/artifact/public")
    for name in ("README.md", "REPRODUCE.md"):
        atomic_write_text(
            args.output / name,
            (release_docs / name).read_text(encoding="utf-8"),
        )

    current_sources = source_files_snapshot(REPRODUCIBILITY_SOURCE_PATHS)
    atomic_write_json(
        args.output / "reproducibility_inventory.json",
        {
            "schema_version": "icassp2027_reproducibility_inventory_v2",
            "global_oof_release": {
                "path": args.global_release_manifest.as_posix(),
                "sha256": file_sha256(args.global_release_manifest),
                "source_kind": global_release.manifest["source_kind"],
            },
            "sequence_oof_release": {
                "path": args.sequence_release_manifest.as_posix(),
                "sha256": file_sha256(args.sequence_release_manifest),
                "source_kind": sequence_release.manifest["source_kind"],
            },
            "current_release_source": current_sources,
            "environment": {
                "pyproject_sha256": file_sha256(Path("pyproject.toml")),
                "uv_lock_sha256": file_sha256(Path("uv.lock")),
            },
            "public_boundary": {
                "raw_charts": "excluded",
                "audio": "excluded",
                "chart_titles": "excluded",
                "human_participant_data": "not_used",
                "public_course_facts": "derived rows and source hashes included",
            },
        },
    )
    initial_scan = privacy_scan(args.output)
    atomic_write_json(args.output / "privacy_report.json", initial_scan)
    files_before_manifest = sorted(
        path
        for path in args.output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    )
    relative_files = {
        path.relative_to(args.output).as_posix() for path in files_before_manifest
    }
    if relative_files != PUBLIC_ARTIFACT_FILES:
        raise ValueError(
            "public artifact has an unexpected file set: "
            f"expected={sorted(PUBLIC_ARTIFACT_FILES)}, "
            f"actual={sorted(relative_files)}"
        )
    manifest = {
        "schema_version": PUBLIC_SCHEMA,
        "privacy_scan": initial_scan,
        "files": {
            path.relative_to(args.output).as_posix(): {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files_before_manifest
        },
    }
    atomic_write_json(args.output / "MANIFEST.json", manifest)
    final_scan = privacy_scan(args.output)
    if final_scan["status"] != "pass":
        raise RuntimeError("final public-artifact privacy scan did not pass")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(files_before_manifest) + 1,
                "privacy": final_scan,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
