#!/usr/bin/env python
"""Generate ICASSP paper numbers and Table 1 from canonical evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from TaikoChartEstimator.research.evidence import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from TaikoChartEstimator.research.dojo import primary_ten_star_panel
from TaikoChartEstimator.research.release import (
    validate_sequence_release_manifest,
)

GLOBAL_LABELS = {
    "ordinary_huber": "Huber (ordinary)",
    "topcoded_huber": "Huber (one-sided)",
    "ridge": "Ridge",
    "gbdt": "GBDT",
    "ordinal_logit": "Cumulative logit",
    "ordinal_probit": "Cumulative probit",
    "ordinal_logit_reduction": "Ordinal reduction",
}
SEQUENCE_LABELS = {
    "seq_ordinary_huber": "Transformer + ordinary Huber",
    "seq_topcoded_huber": "Transformer + one-sided Huber",
    "seq_ordinal_logit": "Transformer + cumulative logit",
}
SCENARIOS = ("native", "cap7", "cap8", "cap9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global-reports",
        type=Path,
        default=Path("experiments/icassp2027_topcoded_oof_v1/reports"),
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
        "--dojo-reports",
        type=Path,
        default=Path("experiments/icassp2027_dojo/reports"),
    )
    parser.add_argument(
        "--dojo-data",
        type=Path,
        default=Path("experiments/icassp2027_dojo/data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/icassp2027/generated"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_row(
    rows: Sequence[Mapping[str, Any]],
    **criteria: Any,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def require_valid_global_contrast_bootstraps(
    rows: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject incomplete uncertainty evidence for reported global contrasts."""

    for scenario, row in rows.items():
        if (
            int(row.get("bootstrap_replicates", -1)) != 10_000
            or int(row.get("spearman_valid_replicates", -1)) != 10_000
            or int(row.get("spearman_undefined_replicates", -1)) != 0
            or int(row.get("strict_pair_accuracy_valid_replicates", -1))
            != 10_000
            or int(row.get("strict_pair_accuracy_undefined_replicates", -1))
            != 0
        ):
            raise ValueError(
                f"global {scenario} contrast requires 10,000 valid paired "
                "title bootstraps for both reported rank metrics"
            )


def require_valid_nested_difference_bootstraps(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
) -> None:
    """Reject incomplete paired title bootstraps from sequence analyses."""

    for row in rows:
        for metric in (
            "spearman_difference",
            "strict_pair_accuracy_difference",
        ):
            summary = row.get(metric)
            if (
                not isinstance(summary, Mapping)
                or int(summary.get("valid_replicates", -1)) != 10_000
                or int(summary.get("undefined_replicates", -1)) != 0
            ):
                raise ValueError(
                    f"{family} {row.get('scenario')} {metric} requires "
                    "10,000 valid paired title bootstraps"
                )


def global_score(
    native: Sequence[Mapping[str, Any]],
    hidden: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
) -> float:
    if scenario == "native":
        row = unique_row(native, scenario=scenario, condition=condition)
    else:
        row = unique_row(hidden, scenario=scenario, condition=condition)
    return float(row["spearman_rho"])


def require_headline_ranking_reversal(
    native: Sequence[Mapping[str, Any]],
    hidden: Sequence[Mapping[str, Any]],
) -> None:
    """Fail if the manuscript's frozen global ranking headline is no longer true."""

    native_rows = [
        row
        for row in native
        if row.get("scenario") == "native"
        and row.get("condition") in GLOBAL_LABELS
    ]
    cap8_rows = [
        row
        for row in hidden
        if row.get("scenario") == "cap8"
        and row.get("condition") in GLOBAL_LABELS
    ]
    if len(native_rows) != len(GLOBAL_LABELS) or len(cap8_rows) != len(
        GLOBAL_LABELS
    ):
        raise ValueError("headline reversal requires all seven global conditions")
    native_by_condition = {
        str(row["condition"]): float(row["spearman_rho"])
        for row in native_rows
    }
    cap8_by_condition = {
        str(row["condition"]): float(row["spearman_rho"])
        for row in cap8_rows
    }
    native_best = max(native_by_condition.values())
    if (
        native_by_condition["gbdt"] != native_best
        or sum(value == native_best for value in native_by_condition.values()) != 1
    ):
        raise ValueError("GBDT is not first by native catalog Spearman")
    cap8_worst = min(cap8_by_condition.values())
    if (
        cap8_by_condition["gbdt"] != cap8_worst
        or sum(value == cap8_worst for value in cap8_by_condition.values()) != 1
    ):
        raise ValueError("GBDT is not last by cap-8 hidden-tail Spearman")
    cap8_best = max(cap8_by_condition.values())
    if (
        cap8_by_condition["ordinal_logit_reduction"] != cap8_best
        or sum(value == cap8_best for value in cap8_by_condition.values()) != 1
    ):
        raise ValueError("ordinal reduction is not first at cap 8")
    pair_counts = {int(row["comparable_pair_count"]) for row in cap8_rows}
    hidden_counts = {int(row["hidden_tail_count"]) for row in cap8_rows}
    if len(pair_counts) != 1 or len(hidden_counts) != 1:
        raise ValueError("cap-8 global conditions use different evaluation panels")


def require_manifest_output(
    manifest: Mapping[str, Any],
    *,
    name: str,
    expected_path: Path,
) -> None:
    outputs = manifest.get("output_files")
    if not isinstance(outputs, Mapping):
        raise ValueError("sequence analysis manifest has no output inventory")
    record = outputs.get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"sequence analysis manifest omits {name}")
    recorded_path = Path(str(record.get("path", "")))
    recorded_hash = record.get("sha256")
    if (
        recorded_path != expected_path
        or not expected_path.is_file()
        or not isinstance(recorded_hash, str)
        or len(recorded_hash) != 64
        or file_sha256(expected_path) != recorded_hash
    ):
        raise ValueError(f"sequence analysis output is stale or mismatched: {name}")


def sequence_score(
    metrics: Sequence[Mapping[str, Any]],
    hidden: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
) -> tuple[float, float | None]:
    metric = unique_row(
        metrics,
        scenario=scenario,
        condition=condition,
        aggregation="ensemble",
        seed=None,
    )
    if scenario == "native":
        point = float(metric["catalog_spearman_against_original"])
        seed_sd = float(metric["catalog_spearman_seed_sd"])
    else:
        point = float(
            unique_row(hidden, scenario=scenario, condition=condition)["spearman_rho"]
        )
        seed_sd = float(metric["hidden_tail_spearman_seed_sd"])
    seed_count = int(metric["seed_count"])
    return point, seed_sd if seed_count > 1 else None


def tex_number(value: float, digits: int = 3, *, signed: bool = False) -> str:
    specifier = f"+.{digits}f" if signed else f".{digits}f"
    return format(value, specifier)


def representation_interpretation(ci_low: float, ci_high: float) -> str:
    if ci_low > 0.0:
        return "The sequence model outperforms global cumulative logit."
    if ci_high < 0.0:
        return "The sequence model does not outperform global cumulative logit."
    return "The interval includes zero, leaving the difference unresolved."


def format_score(value: float, seed_sd: float | None = None) -> str:
    formatted = tex_number(value)
    if seed_sd is None:
        return formatted
    return formatted + r" {[" + tex_number(seed_sd) + "]}"


def render_table(table_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Full scale & Cap 7 & Cap 8 & Cap 9 \\",
        r"\midrule",
    ]
    for index, row in enumerate(table_rows):
        if index == len(GLOBAL_LABELS):
            lines.append(r"\midrule")
        cells = [str(row["label"])]
        scores = row["scores"]
        for scenario in SCENARIOS:
            score = scores.get(scenario)
            if score is None:
                cells.append(r"---")
            elif isinstance(score, Mapping):
                cells.append(format_score(float(score["point"]), score["seed_sd"]))
            else:
                cells.append(format_score(float(score)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_table(
    global_native: Sequence[Mapping[str, Any]],
    global_hidden: Sequence[Mapping[str, Any]],
    sequence_metrics: Sequence[Mapping[str, Any]],
    sequence_hidden: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    table_rows: list[dict[str, Any]] = []
    for condition, label in GLOBAL_LABELS.items():
        scores = {
            scenario: global_score(
                global_native,
                global_hidden,
                scenario=scenario,
                condition=condition,
            )
            for scenario in SCENARIOS
        }
        table_rows.append(
            {
                "family": "global",
                "condition": condition,
                "label": label,
                "scores": scores,
            }
        )
    sequence_availability = {
        "seq_ordinary_huber": ("cap7", "cap8", "cap9"),
        "seq_topcoded_huber": ("cap8",),
        "seq_ordinal_logit": ("native", "cap7", "cap8", "cap9"),
    }
    for condition, label in SEQUENCE_LABELS.items():
        scores: dict[str, Any] = {}
        for scenario in sequence_availability[condition]:
            point, seed_sd = sequence_score(
                sequence_metrics,
                sequence_hidden,
                scenario=scenario,
                condition=condition,
            )
            scores[scenario] = {"point": point, "seed_sd": seed_sd}
        table_rows.append(
            {
                "family": "sequence",
                "condition": condition,
                "label": label,
                "scores": scores,
            }
        )

    atomic_write_text(output, render_table(table_rows))
    return table_rows


def render_numbers(evidence: Mapping[str, Any]) -> str:
    global_values = evidence["global"]
    dojo = evidence["dojo"]
    sequence = evidence["sequence"]
    values = {
        "ChartCount": str(global_values["chart_count"]),
        "TitleCount": str(global_values["title_count"]),
        "CapEightCount": str(global_values["cap8_hidden_count"]),
        "TopcodedGain": tex_number(global_values["cap8_gain"], signed=True),
        "TopcodedGainLow": tex_number(global_values["cap8_gain_ci_low"], signed=True),
        "TopcodedGainHigh": tex_number(global_values["cap8_gain_ci_high"], signed=True),
        "TopcodedCapSevenGain": tex_number(
            global_values["cap7_gain"], signed=True
        ),
        "TopcodedCapSevenGainLow": tex_number(
            global_values["cap7_gain_ci_low"], signed=True
        ),
        "TopcodedCapSevenGainHigh": tex_number(
            global_values["cap7_gain_ci_high"], signed=True
        ),
        "TopcodedCapNineGain": tex_number(
            global_values["cap9_gain"], signed=True
        ),
        "TopcodedCapNineGainLow": tex_number(
            global_values["cap9_gain_ci_low"], signed=True
        ),
        "TopcodedCapNineGainHigh": tex_number(
            global_values["cap9_gain_ci_high"], signed=True
        ),
        "GbdtNativeRho": tex_number(global_values["gbdt_native_rho"]),
        "GbdtCapEightRho": tex_number(global_values["gbdt_cap8_rho"]),
        "GbdtCapEightTau": tex_number(global_values["gbdt_cap8_tau"]),
        "GbdtCapEightPair": tex_number(global_values["gbdt_cap8_pair"]),
        "OrdinalNativeRho": tex_number(global_values["ordinal_native_rho"]),
        "OrdinalCapEightRho": tex_number(global_values["ordinal_cap8_rho"]),
        "OrdinalCapEightTau": tex_number(global_values["ordinal_cap8_tau"]),
        "OrdinalCapEightPair": tex_number(global_values["ordinal_cap8_pair"]),
        "CapEightPairCount": f"{int(global_values['cap8_pair_count']):,}",
        "DojoPlacementCount": str(dojo["placement_count"]),
        "DojoYearCount": str(dojo["year_count"]),
        "DojoTierCount": str(dojo["tier_count"]),
        "DojoPairCount": str(dojo["pair_count"]),
        "DojoRho": tex_number(dojo["spearman_rho"]),
        "DojoRhoLow": tex_number(dojo["spearman_ci_low"]),
        "DojoRhoHigh": tex_number(dojo["spearman_ci_high"]),
        "DojoPair": tex_number(dojo["pair_accuracy"]),
        "DojoPairLow": tex_number(dojo["pair_ci_low"]),
        "DojoPairHigh": tex_number(dojo["pair_ci_high"]),
        "DojoControlPlacementCount": str(dojo["control_placement_count"]),
        "DojoBelowCeilingPairCount": str(dojo["below_ceiling_pair_count"]),
        "DojoBelowCeilingPair": tex_number(dojo["below_ceiling_pair_accuracy"]),
        "SeqCapEightOrdinaryRho": tex_number(sequence["cap8_ordinary_rho"]),
        "SeqCapEightTopcodedRho": tex_number(sequence["cap8_topcoded_rho"]),
        "SeqCapEightOrdinalRho": tex_number(sequence["cap8_ordinal_rho"]),
        "SeqCapEightTopcodedGain": tex_number(
            sequence["cap8_topcoded_gain"], signed=True
        ),
        "SeqCapEightTopcodedGainLow": tex_number(
            sequence["cap8_topcoded_gain_ci_low"], signed=True
        ),
        "SeqCapEightTopcodedGainHigh": tex_number(
            sequence["cap8_topcoded_gain_ci_high"], signed=True
        ),
        "SeqRepresentationGain": tex_number(
            sequence["representation_gain"], signed=True
        ),
        "SeqRepresentationGainLow": tex_number(
            sequence["representation_gain_ci_low"], signed=True
        ),
        "SeqRepresentationGainHigh": tex_number(
            sequence["representation_gain_ci_high"], signed=True
        ),
        "GlobalOrdinalLogitCapEightRho": tex_number(
            sequence["global_ordinal_logit_cap8_rho"]
        ),
        "SeqRepresentationInterpretation": representation_interpretation(
            sequence["representation_gain_ci_low"],
            sequence["representation_gain_ci_high"],
        ),
        "SeqDojoRho": tex_number(dojo["sequence_spearman_rho"]),
        "SeqDojoRhoLow": tex_number(dojo["sequence_spearman_ci_low"]),
        "SeqDojoRhoHigh": tex_number(dojo["sequence_spearman_ci_high"]),
        "SeqDojoPair": tex_number(dojo["sequence_pair_accuracy"]),
        "SeqDojoPairLow": tex_number(dojo["sequence_pair_ci_low"]),
        "SeqDojoPairHigh": tex_number(dojo["sequence_pair_ci_high"]),
    }
    lines = [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def write_numbers(evidence: Mapping[str, Any], output: Path) -> None:
    atomic_write_text(output, render_numbers(evidence))


def _validated_dojo_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    condition: str,
) -> Mapping[str, Any]:
    row = unique_row(rows, family=family, condition=condition)
    if row.get("schema_version") != "icassp2027_dojo_criterion_summary_v1":
        raise ValueError("unexpected Dojo criterion schema")
    if (
        row.get("primary_panel") != "high_dan_official_10_oni"
        or int(row.get("placement_count", -1)) != 47
        or int(row.get("unique_chart_count", -1)) != 47
        or int(row.get("year_count", -1)) != 6
        or int(row.get("tier_count", -1)) != 5
        or int(row.get("pair_count", -1)) != 152
    ):
        raise ValueError(f"{family} Dojo primary panel drifted from the frozen spec")
    for metric in ("macro_year_spearman", "within_year_pair_accuracy"):
        summary = row.get(metric)
        if (
            not isinstance(summary, Mapping)
            or int(summary.get("valid_bootstrap_replicates", -1)) != 10_000
            or int(summary.get("undefined_bootstrap_replicates", -1)) != 0
            or int(summary.get("valid_permutation_replicates", -1)) != 10_000
            or int(summary.get("undefined_permutation_replicates", -1)) != 0
        ):
            raise ValueError(f"{family} Dojo {metric} uncertainty is incomplete")
    return row


def main() -> None:
    """Build paper evidence with the public Dan-i Dojo design criterion."""

    args = parse_args()
    sequence_release = validate_sequence_release_manifest(
        args.sequence_release_manifest
    )
    required = {
        "global_native": args.global_reports / "native_catalog_metrics.jsonl",
        "global_hidden": args.global_reports / "hidden_tail_metrics.jsonl",
        "global_contrasts": args.global_reports / "hidden_tail_contrasts.jsonl",
        "sequence_metrics": Path(
            str(sequence_release.manifest["files"]["sequence_metrics"]["path"])
        ),
        "sequence_manifest": args.sequence_release_manifest,
        "sequence_hidden": args.sequence_reports / "hidden_tail_metrics.jsonl",
        "sequence_contrasts": args.sequence_reports / "hidden_tail_contrasts.jsonl",
        "sequence_representation_contrasts": (
            args.sequence_reports / "representation_contrasts.jsonl"
        ),
        "dojo_summary": args.dojo_reports / "criterion_summary.jsonl",
        "dojo_matches": args.dojo_reports / "matched_placements.jsonl",
        "dojo_per_year": args.dojo_reports / "per_year_metrics.jsonl",
        "dojo_leave_one_year_out": args.dojo_reports / "leave_one_year_out.jsonl",
        "dojo_story": args.dojo_reports / "primary_story_summary.json",
        "dojo_analysis_manifest": args.dojo_reports / "analysis_manifest.json",
        "dojo_source_manifest": args.dojo_data / "SOURCE_MANIFEST.json",
        "dojo_placements": args.dojo_data / "course_placements.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"canonical paper inputs missing: {missing}")

    global_native = read_jsonl(required["global_native"])
    global_hidden = read_jsonl(required["global_hidden"])
    global_contrasts = read_jsonl(required["global_contrasts"])
    sequence_metrics = read_jsonl(required["sequence_metrics"])
    sequence_hidden = read_jsonl(required["sequence_hidden"])
    sequence_contrasts = read_jsonl(required["sequence_contrasts"])
    sequence_representation_contrasts = read_jsonl(
        required["sequence_representation_contrasts"]
    )
    dojo_summaries = read_jsonl(required["dojo_summary"])
    dojo_matches = read_jsonl(required["dojo_matches"])
    dojo_story = read_json(required["dojo_story"])
    dojo_analysis_manifest = read_json(required["dojo_analysis_manifest"])
    dojo_source_manifest = read_json(required["dojo_source_manifest"])

    sequence_manifest = sequence_release.manifest
    require_headline_ranking_reversal(global_native, global_hidden)
    if int(dojo_analysis_manifest.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Dojo analysis requires 10,000 bootstrap replicates")
    if int(dojo_analysis_manifest.get("permutation_replicates", -1)) != 10_000:
        raise ValueError("Dojo analysis requires 10,000 blocked permutations")
    for output_name, required_name in (
        ("criterion_summary", "dojo_summary"),
        ("matched_placements", "dojo_matches"),
        ("per_year_metrics", "dojo_per_year"),
        ("leave_one_year_out", "dojo_leave_one_year_out"),
        ("primary_story_summary", "dojo_story"),
    ):
        require_manifest_output(
            dojo_analysis_manifest,
            name=output_name,
            expected_path=required[required_name],
        )
    placement_record = dojo_source_manifest.get("output", {})
    if (
        dojo_source_manifest.get("schema_version")
        != "icassp2027_dojo_source_manifest_v1"
        or int(placement_record.get("placement_count", -1)) != 342
        or placement_record.get("sha256") != file_sha256(required["dojo_placements"])
    ):
        raise ValueError("Dojo source manifest does not verify the frozen snapshot")

    cap_rows: dict[str, Mapping[str, Any]] = {}
    for scenario in ("cap7", "cap8", "cap9"):
        cap_rows[scenario] = unique_row(
            global_contrasts,
            scenario=scenario,
            first_condition="topcoded_huber",
            second_condition="ordinary_huber",
        )
    require_valid_global_contrast_bootstraps(cap_rows)
    global_gbdt_native = unique_row(
        global_native, scenario="native", condition="gbdt"
    )
    global_gbdt_cap8 = unique_row(global_hidden, scenario="cap8", condition="gbdt")
    global_ordinal_cap8 = unique_row(
        global_hidden, scenario="cap8", condition="ordinal_logit_reduction"
    )
    global_ordinal_native = unique_row(
        global_native, scenario="native", condition="ordinal_logit_reduction"
    )

    seq_ordinary = unique_row(
        sequence_hidden, scenario="cap8", condition="seq_ordinary_huber"
    )
    seq_topcoded = unique_row(
        sequence_hidden, scenario="cap8", condition="seq_topcoded_huber"
    )
    seq_ordinal = unique_row(
        sequence_hidden, scenario="cap8", condition="seq_ordinal_logit"
    )
    seq_topcoded_contrast = unique_row(
        sequence_contrasts,
        scenario="cap8",
        first_condition="seq_topcoded_huber",
        second_condition="seq_ordinary_huber",
    )
    sequence_bootstrap_rows = {
        "hidden_tail_contrasts": sequence_contrasts,
        "representation_contrasts": sequence_representation_contrasts,
    }
    for family, rows in sequence_bootstrap_rows.items():
        require_valid_nested_difference_bootstraps(rows, family=family)
    representation = unique_row(
        sequence_representation_contrasts,
        scenario="cap8",
        first_condition="seq_ordinal_logit",
        second_condition="ordinal_logit",
    )["spearman_difference"]
    global_ordinal_logit_cap8 = unique_row(
        global_hidden, scenario="cap8", condition="ordinal_logit"
    )

    dojo_global = _validated_dojo_summary(
        dojo_summaries, family="global", condition="topcoded_huber"
    )
    dojo_sequence = _validated_dojo_summary(
        dojo_summaries, family="sequence", condition="seq_ordinal_logit"
    )
    global_primary = primary_ten_star_panel(
        [
            row
            for row in dojo_matches
            if row.get("family") == "global"
            and row.get("condition") == "topcoded_huber"
        ]
    )
    sequence_primary = primary_ten_star_panel(
        [
            row
            for row in dojo_matches
            if row.get("family") == "sequence"
            and row.get("condition") == "seq_ordinal_logit"
        ]
    )
    identity = lambda row: (
        int(row["year"]),
        str(row["dan"]),
        int(row["position"]),
        str(row["normalized_title"]),
    )
    if {identity(row) for row in global_primary} != {
        identity(row) for row in sequence_primary
    }:
        raise ValueError("global and sequence Dojo criteria use different placements")
    if len(global_primary) != 47 or len(sequence_primary) != 47:
        raise ValueError("Dojo primary match panel is incomplete")
    if int(dojo_story.get("primary_panel_identity_count", -1)) != 47:
        raise ValueError("Dojo story summary disagrees with the primary panel")

    global_rho = dojo_global["macro_year_spearman"]
    global_pair = dojo_global["within_year_pair_accuracy"]
    sequence_rho = dojo_sequence["macro_year_spearman"]
    sequence_pair = dojo_sequence["within_year_pair_accuracy"]
    controls = dojo_global["controls"]
    evidence: dict[str, Any] = {
        "schema_version": "icassp2027_paper_evidence_v2",
        "global": {
            "chart_count": int(global_gbdt_native["chart_count"]),
            "title_count": int(global_gbdt_native["title_group_count"]),
            "cap8_hidden_count": int(global_gbdt_cap8["hidden_tail_count"]),
            "cap8_gain": float(cap_rows["cap8"]["spearman_point_difference"]),
            "cap8_gain_ci_low": float(cap_rows["cap8"]["spearman_bootstrap_ci_low"]),
            "cap8_gain_ci_high": float(cap_rows["cap8"]["spearman_bootstrap_ci_high"]),
            "cap7_gain": float(cap_rows["cap7"]["spearman_point_difference"]),
            "cap7_gain_ci_low": float(cap_rows["cap7"]["spearman_bootstrap_ci_low"]),
            "cap7_gain_ci_high": float(cap_rows["cap7"]["spearman_bootstrap_ci_high"]),
            "cap9_gain": float(cap_rows["cap9"]["spearman_point_difference"]),
            "cap9_gain_ci_low": float(cap_rows["cap9"]["spearman_bootstrap_ci_low"]),
            "cap9_gain_ci_high": float(cap_rows["cap9"]["spearman_bootstrap_ci_high"]),
            "gbdt_native_rho": float(global_gbdt_native["spearman_rho"]),
            "gbdt_cap8_rho": float(global_gbdt_cap8["spearman_rho"]),
            "gbdt_cap8_tau": float(global_gbdt_cap8["kendall_tau_b"]),
            "gbdt_cap8_pair": float(global_gbdt_cap8["strict_comparable_pair_accuracy"]),
            "ordinal_native_rho": float(global_ordinal_native["spearman_rho"]),
            "ordinal_cap8_rho": float(global_ordinal_cap8["spearman_rho"]),
            "ordinal_cap8_tau": float(global_ordinal_cap8["kendall_tau_b"]),
            "ordinal_cap8_pair": float(global_ordinal_cap8["strict_comparable_pair_accuracy"]),
            "cap8_pair_count": int(global_gbdt_cap8["comparable_pair_count"]),
        },
        "dojo": {
            "placement_count": int(dojo_global["placement_count"]),
            "year_count": int(dojo_global["year_count"]),
            "tier_count": int(dojo_global["tier_count"]),
            "pair_count": int(dojo_global["pair_count"]),
            "spearman_rho": float(global_rho["point"]),
            "spearman_ci_low": float(global_rho["bootstrap_ci_low"]),
            "spearman_ci_high": float(global_rho["bootstrap_ci_high"]),
            "spearman_permutation_p": float(global_rho["permutation_p_value"]),
            "pair_accuracy": float(global_pair["point"]),
            "pair_ci_low": float(global_pair["bootstrap_ci_low"]),
            "pair_ci_high": float(global_pair["bootstrap_ci_high"]),
            "pair_permutation_p": float(global_pair["permutation_p_value"]),
            "sequence_spearman_rho": float(sequence_rho["point"]),
            "sequence_spearman_ci_low": float(sequence_rho["bootstrap_ci_low"]),
            "sequence_spearman_ci_high": float(sequence_rho["bootstrap_ci_high"]),
            "sequence_pair_accuracy": float(sequence_pair["point"]),
            "sequence_pair_ci_low": float(sequence_pair["bootstrap_ci_low"]),
            "sequence_pair_ci_high": float(sequence_pair["bootstrap_ci_high"]),
            "control_placement_count": int(controls["matched_oni_placements_all_stars"]),
            "below_ceiling_pair_count": int(controls["below_ceiling_equal_star_pair_count"]),
            "below_ceiling_pair_accuracy": float(controls["below_ceiling_equal_star_score_pair_accuracy"]),
            "source_snapshot_sha256": str(placement_record["sha256"]),
        },
        "sequence": {
            "canonical_contract": {
                key: int(sequence_manifest["contract"][key])
                for key in (
                    "fold_run_count",
                    "seed_oof_row_count",
                    "ensemble_row_count",
                    "metric_row_count",
                )
            },
            "cap8_ordinary_rho": float(seq_ordinary["spearman_rho"]),
            "cap8_topcoded_rho": float(seq_topcoded["spearman_rho"]),
            "cap8_ordinal_rho": float(seq_ordinal["spearman_rho"]),
            "cap8_topcoded_gain": float(seq_topcoded_contrast["spearman_difference"]["point_difference"]),
            "cap8_topcoded_gain_ci_low": float(seq_topcoded_contrast["spearman_difference"]["bootstrap_ci_low"]),
            "cap8_topcoded_gain_ci_high": float(seq_topcoded_contrast["spearman_difference"]["bootstrap_ci_high"]),
            "representation_gain": float(representation["point_difference"]),
            "representation_gain_ci_low": float(representation["bootstrap_ci_low"]),
            "representation_gain_ci_high": float(representation["bootstrap_ci_high"]),
            "global_ordinal_logit_cap8_rho": float(global_ordinal_logit_cap8["spearman_rho"]),
        },
        "bootstrap_contract": {
            "requested_replicates": 10_000,
            "global_hidden_contrast_rows": len(cap_rows),
            "sequence_hidden_contrast_rows": len(sequence_contrasts),
            "sequence_representation_contrast_rows": len(sequence_representation_contrasts),
            "dojo_metric_rows": 4,
            "dojo_blocked_permutation_rows": 4,
            "all_reported_intervals_have_10000_valid_replicates": True,
        },
        "source_files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in required.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    table_rows = write_table(
        global_native,
        global_hidden,
        sequence_metrics,
        sequence_hidden,
        args.output / "hidden_order_table.tex",
    )
    evidence["table_rows"] = table_rows
    write_numbers(evidence, args.output / "hidden_order_numbers.tex")
    evidence["generated_files"] = {
        "hidden_order_table.tex": file_sha256(args.output / "hidden_order_table.tex"),
        "hidden_order_numbers.tex": file_sha256(args.output / "hidden_order_numbers.tex"),
    }
    atomic_write_json(args.output / "paper_evidence.json", evidence)
    print(
        json.dumps(
            {
                "paper_evidence": str(args.output / "paper_evidence.json"),
                "table": str(args.output / "hidden_order_table.tex"),
                "numbers": str(args.output / "hidden_order_numbers.tex"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
