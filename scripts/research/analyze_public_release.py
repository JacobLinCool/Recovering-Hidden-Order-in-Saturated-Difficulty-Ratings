#!/usr/bin/env python3
"""Recompute paper point estimates from the released, title-free OOF rows.

This is a public-data check, not a retraining pipeline. Normalized titles,
row-level lower-rated Dojo controls, and per-seed sequence predictions are not
released. Consequently, title-cluster confidence intervals, those controls,
and seed variation cannot be independently recomputed by this program.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr


ARTIFACT = Path("paper/icassp2027/artifact/public")
PAPER = Path("paper/icassp2027")
SCENARIOS = {"native": 10, "cap7": 7, "cap8": 8, "cap9": 9}
GLOBAL_CONDITIONS = (
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
    "ordinal_logit",
    "ordinal_probit",
    "ordinal_logit_reduction",
)
SEQUENCE_CASES = frozenset(
    {
        ("cap7", "seq_ordinary_huber"),
        ("cap7", "seq_ordinal_logit"),
        ("cap8", "seq_ordinary_huber"),
        ("cap8", "seq_topcoded_huber"),
        ("cap8", "seq_ordinal_logit"),
        ("cap9", "seq_ordinary_huber"),
        ("cap9", "seq_ordinal_logit"),
        ("native", "seq_ordinal_logit"),
    }
)
CHART_COUNT = 1_023
DOJO_YEARS = frozenset(range(2020, 2026))
DOJO_FAMILIES = {"global": "topcoded_huber", "sequence": "seq_ordinal_logit"}
POINT_TOLERANCE = 1e-10


class PublicAnalysisError(ValueError):
    """Released rows disagree with their claimed schema or paper points."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicAnalysisError(message)


def close(actual: float, expected: Any, name: str) -> None:
    expected_number = number(expected, name)
    require(
        math.isclose(actual, expected_number, rel_tol=POINT_TOLERANCE, abs_tol=POINT_TOLERANCE),
        f"{name}: recomputed {actual:.12g}, reported {expected_number:.12g}",
    )


def number(value: Any, name: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}: expected number")
    result = float(value)
    require(math.isfinite(result), f"{name}: expected finite number")
    return result


def integer(value: Any, name: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{name}: expected integer")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicAnalysisError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                require(bool(line.strip()), f"{path}:{line_number}: blank line")
                row = json.loads(line)
                require(isinstance(row, dict), f"{path}:{line_number}: expected JSON object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicAnalysisError(f"cannot read {path}: {exc}") from exc
    return rows


def indexed_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], name: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        try:
            key = tuple(row[field] for field in keys)
        except KeyError as exc:
            raise PublicAnalysisError(f"{name}: missing {exc}") from exc
        require(key not in result, f"{name}: duplicate key {key}")
        result[key] = row
    return result


def load_oof(
    path: Path,
    *,
    schema: str,
    expected_cases: set[tuple[str, str]],
    reference: dict[str, tuple[int, int]] | None = None,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, tuple[int, int]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, str]] = set()
    seen_reference: dict[str, tuple[int, int]] = {}
    for row in read_jsonl(path):
        require(row.get("schema_version") == schema, f"{path.name}: unexpected score schema")
        scenario, condition, key = row["scenario"], row["condition"], row["chart_key"]
        case = scenario, condition
        require(case in expected_cases, f"{path.name}: unexpected condition {case}")
        require(isinstance(key, str) and re.fullmatch(r"chart-[0-9a-f]{20}", key) is not None, "invalid chart key")
        identity = scenario, condition, key
        require(identity not in identities, f"{path.name}: duplicate score {identity}")
        identities.add(identity)
        label = number(row["evaluation_original_label"], "original label")
        observed = number(row["observed_label"], "observed label")
        fold = integer(row["outer_fold"], "outer fold")
        require(label.is_integer() and 1 <= label <= 10 and 0 <= fold < 5, "invalid label or fold")
        require(observed == min(label, SCENARIOS[scenario]), f"{path.name}: incorrect top coding")
        number(row["latent_score"], "latent score")
        number(row["expected_observed_label"], "expected observed label")
        original = int(label), fold
        require(key not in seen_reference or seen_reference[key] == original, "chart label/fold changes across conditions")
        seen_reference[key] = original
        if reference is not None:
            require(reference.get(key) == original, "sequence/global chart alignment differs")
        grouped[case].append(row)
    require(set(grouped) == expected_cases, f"{path.name}: missing OOF condition")
    require(len(seen_reference) == CHART_COUNT, f"{path.name}: chart count differs")
    require(all(len(rows) == CHART_COUNT for rows in grouped.values()), f"{path.name}: incomplete OOF case")
    require(len(identities) == CHART_COUNT * len(expected_cases), f"{path.name}: wrong row count")
    if reference is not None:
        require(seen_reference == reference, "sequence/global chart sets differ")
    for rows in grouped.values():
        rows.sort(key=lambda row: row["chart_key"])
    return dict(grouped), seen_reference


@dataclass(frozen=True)
class RankPoints:
    rho: float
    tau: float
    pair_accuracy: float
    panel_count: int
    strictly_hidden_count: int
    comparable_pairs: int
    correct_pairs: int
    prediction_ties: int


def rank_points(rows: list[dict[str, Any]], cap: int | None = None) -> RankPoints:
    selected = [row for row in rows if cap is None or number(row["evaluation_original_label"], "label") >= cap]
    require(len(selected) > 1, "rank panel is empty")
    labels = np.asarray([number(row["evaluation_original_label"], "label") for row in selected], dtype=np.float64)
    scores = np.asarray([number(row["latent_score"], "score") for row in selected], dtype=np.float64)
    rho = float(spearmanr(labels, scores).statistic)
    tau = float(kendalltau(labels, scores, variant="b").statistic)
    require(math.isfinite(rho) and math.isfinite(tau), "rank point is undefined")
    left, right = np.triu_indices(len(selected), k=1)
    target_diff = labels[left] - labels[right]
    score_diff = scores[left] - scores[right]
    comparable = target_diff != 0
    pair_count = int(np.count_nonzero(comparable))
    require(pair_count > 0, "no comparable rating pairs")
    correct = int(np.count_nonzero(comparable & (((target_diff > 0) & (score_diff > 0)) | ((target_diff < 0) & (score_diff < 0)))))
    ties = int(np.count_nonzero(comparable & (score_diff == 0)))
    return RankPoints(
        rho=rho,
        tau=tau,
        pair_accuracy=correct / pair_count,
        panel_count=len(selected),
        strictly_hidden_count=int(np.count_nonzero(labels > cap)) if cap is not None else 0,
        comparable_pairs=pair_count,
        correct_pairs=correct,
        prediction_ties=ties,
    )


def check_metric_rows(
    artifact: Path,
    global_points: dict[tuple[str, str], RankPoints],
    sequence_points: dict[tuple[str, str], RankPoints],
) -> None:
    global_expected = {(scenario, condition) for scenario in SCENARIOS if scenario != "native" for condition in GLOBAL_CONDITIONS}
    global_reported = indexed_rows(read_jsonl(artifact / "global_hidden_tail_metrics.jsonl"), ("scenario", "condition"), "global hidden metrics")
    require(set(global_reported) == global_expected, "global hidden metric row coverage differs")
    for case, row in global_reported.items():
        point = global_points[case]
        require(row["hidden_tail_count"] == point.panel_count, f"{case}: hidden count differs")
        require(row["strictly_hidden_count"] == point.strictly_hidden_count, f"{case}: strict hidden count differs")
        require(row["comparable_pair_count"] == point.comparable_pairs, f"{case}: comparable pair count differs")
        require(row["strict_pair_correct_count"] == point.correct_pairs, f"{case}: correct pair count differs")
        require(row["prediction_tie_comparable_pair_count"] == point.prediction_ties, f"{case}: prediction tie count differs")
        require(row["total_pair_count"] == point.panel_count * (point.panel_count - 1) // 2, f"{case}: all pair count differs")
        close(point.rho, row["spearman_rho"], f"{case} rho")
        close(point.tau, row["kendall_tau_b"], f"{case} tau-b")
        close(point.pair_accuracy, row["strict_comparable_pair_accuracy"], f"{case} pair accuracy")

    sequence_expected = {case for case in SEQUENCE_CASES if case[0] != "native"}
    sequence_reported = indexed_rows(read_jsonl(artifact / "sequence_hidden_tail_metrics.jsonl"), ("scenario", "condition"), "sequence hidden metrics")
    require(set(sequence_reported) == sequence_expected, "sequence hidden metric row coverage differs")
    for case, row in sequence_reported.items():
        point = sequence_points[case]
        require(row["hidden_tail_count"] == point.panel_count, f"{case}: hidden count differs")
        close(point.rho, row["spearman_rho"], f"{case} rho")
        close(point.tau, row["kendall_tau_b"], f"{case} tau-b")
        close(point.pair_accuracy, row["strict_pair_accuracy"], f"{case} pair accuracy")


def check_contrasts(
    artifact: Path,
    global_points: dict[tuple[str, str], RankPoints],
    sequence_points: dict[tuple[str, str], RankPoints],
) -> None:
    cases = (
        ("global_hidden_tail_contrasts.jsonl", global_points, global_points, {(scenario, "topcoded_huber", condition) for scenario in ("cap7", "cap8", "cap9") for condition in GLOBAL_CONDITIONS if condition != "topcoded_huber"}),
        ("sequence_hidden_tail_contrasts.jsonl", sequence_points, sequence_points, {("cap8", "seq_topcoded_huber", "seq_ordinary_huber"), *((scenario, "seq_ordinal_logit", "seq_ordinary_huber") for scenario in ("cap7", "cap8", "cap9"))}),
        ("sequence_representation_contrasts.jsonl", sequence_points, global_points, {(scenario, "seq_ordinal_logit", "ordinal_logit") for scenario in ("cap7", "cap8", "cap9")}),
    )
    for filename, first_points, second_points, expected in cases:
        rows = indexed_rows(read_jsonl(artifact / filename), ("scenario", "first_condition", "second_condition"), filename)
        require(set(rows) == expected, f"{filename}: contrast row coverage differs")
        for (scenario, first, second), row in rows.items():
            first_point = first_points[scenario, first]
            second_point = second_points[scenario, second]
            if filename == "global_hidden_tail_contrasts.jsonl":
                close(first_point.rho - second_point.rho, row["spearman_point_difference"], f"{filename} {scenario}/{first}/{second} rho gain")
                close(first_point.tau - second_point.tau, row["kendall_tau_b_point_difference"], f"{filename} {scenario}/{first}/{second} tau gain")
                close(first_point.pair_accuracy - second_point.pair_accuracy, row["strict_pair_accuracy_point_difference"], f"{filename} {scenario}/{first}/{second} pair gain")
            else:
                close(first_point.rho - second_point.rho, row["spearman_difference"]["point_difference"], f"{filename} {scenario}/{first}/{second} rho gain")
                close(first_point.pair_accuracy - second_point.pair_accuracy, row["strict_pair_accuracy_difference"]["point_difference"], f"{filename} {scenario}/{first}/{second} pair gain")


def check_table_and_evidence(
    artifact: Path,
    global_points: dict[tuple[str, str], RankPoints],
    sequence_points: dict[tuple[str, str], RankPoints],
) -> dict[str, Any]:
    evidence = read_json(artifact / "paper_evidence.json")
    require(evidence.get("schema_version") == "icassp2027_paper_evidence_v2", "paper evidence schema differs")
    expected_rows = {("global", condition) for condition in GLOBAL_CONDITIONS} | {("sequence", condition) for _, condition in SEQUENCE_CASES}
    table_rows = indexed_rows(evidence["table_rows"], ("family", "condition"), "paper table")
    require(set(table_rows) == expected_rows, "paper table row coverage differs")
    for (family, condition), row in table_rows.items():
        points = global_points if family == "global" else sequence_points
        expected_scenarios = set(SCENARIOS) if family == "global" else {scenario for scenario, item in SEQUENCE_CASES if item == condition}
        require(set(row["scores"]) == expected_scenarios, f"paper table {family}/{condition}: scenario coverage differs")
        for scenario, reported in row["scores"].items():
            reported_point = reported if family == "global" else reported["point"]
            close(points[scenario, condition].rho, reported_point, f"paper table {family}/{scenario}/{condition} rho")

    global_reported = evidence["global"]
    sequence_reported = evidence["sequence"]
    require(global_reported["chart_count"] == CHART_COUNT, "paper chart count differs")
    require(global_reported["cap8_hidden_count"] == global_points["cap8", "gbdt"].panel_count == 699, "paper cap-8 panel count differs")
    require(global_reported["cap8_pair_count"] == global_points["cap8", "gbdt"].comparable_pairs == 159_318, "paper cap-8 pair count differs")
    for label, case, field in (
        ("gbdt_native_rho", ("native", "gbdt"), "rho"),
        ("gbdt_cap8_rho", ("cap8", "gbdt"), "rho"),
        ("gbdt_cap8_tau", ("cap8", "gbdt"), "tau"),
        ("gbdt_cap8_pair", ("cap8", "gbdt"), "pair_accuracy"),
        ("ordinal_native_rho", ("native", "ordinal_logit_reduction"), "rho"),
        ("ordinal_cap8_rho", ("cap8", "ordinal_logit_reduction"), "rho"),
        ("ordinal_cap8_tau", ("cap8", "ordinal_logit_reduction"), "tau"),
        ("ordinal_cap8_pair", ("cap8", "ordinal_logit_reduction"), "pair_accuracy"),
    ):
        close(getattr(global_points[case], field), global_reported[label], f"paper evidence {label}")
    for scenario in ("cap7", "cap8", "cap9"):
        gain = global_points[scenario, "topcoded_huber"].rho - global_points[scenario, "ordinary_huber"].rho
        close(gain, global_reported[f"{scenario}_gain"], f"paper evidence {scenario} gain")
    for label, case in (
        ("cap8_ordinary_rho", ("cap8", "seq_ordinary_huber")),
        ("cap8_topcoded_rho", ("cap8", "seq_topcoded_huber")),
        ("cap8_ordinal_rho", ("cap8", "seq_ordinal_logit")),
    ):
        close(sequence_points[case].rho, sequence_reported[label], f"paper evidence {label}")
    close(sequence_points["cap8", "seq_topcoded_huber"].rho - sequence_points["cap8", "seq_ordinary_huber"].rho, sequence_reported["cap8_topcoded_gain"], "paper sequence gain")
    close(sequence_points["cap8", "seq_ordinal_logit"].rho - global_points["cap8", "ordinal_logit"].rho, sequence_reported["representation_gain"], "paper representation gain")
    close(global_points["cap8", "ordinal_logit"].rho, sequence_reported["global_ordinal_logit_cap8_rho"], "paper global ordinal logit rho")
    return evidence


def dojo_points(rows: list[dict[str, Any]]) -> tuple[float, float, int, dict[int, tuple[int, float, float, int]]]:
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_year[integer(row["year"], "Dojo year")].append(row)
    require(set(by_year) == DOJO_YEARS, "Dojo year coverage differs")
    year_points: dict[int, tuple[int, float, float, int]] = {}
    for year, year_rows in sorted(by_year.items()):
        tiers = np.asarray([integer(row["expert_tier_order"], "Dojo tier") for row in year_rows], dtype=np.float64)
        scores = np.asarray([number(row["latent_score"], "Dojo score") for row in year_rows], dtype=np.float64)
        rho = float(spearmanr(tiers, scores).statistic)
        require(math.isfinite(rho), f"Dojo {year}: undefined rho")
        left, right = np.triu_indices(len(year_rows), k=1)
        tier_diff = tiers[left] - tiers[right]
        score_diff = scores[left] - scores[right]
        eligible = tier_diff != 0
        pair_count = int(np.count_nonzero(eligible))
        require(pair_count > 0, f"Dojo {year}: no eligible pairs")
        correct = int(np.count_nonzero(eligible & (((tier_diff > 0) & (score_diff > 0)) | ((tier_diff < 0) & (score_diff < 0)))))
        year_points[year] = len(year_rows), rho, correct / pair_count, pair_count
    return (
        float(np.mean([point[1] for point in year_points.values()])),
        float(np.mean([point[2] for point in year_points.values()])),
        sum(point[3] for point in year_points.values()),
        year_points,
    )


def check_dojo(
    artifact: Path,
    global_rows: dict[tuple[str, str], list[dict[str, Any]]],
    sequence_rows: dict[tuple[str, str], list[dict[str, Any]]],
    evidence: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    native_indices = {
        "global": {row["chart_key"]: row for row in global_rows["native", "topcoded_huber"]},
        "sequence": {row["chart_key"]: row for row in sequence_rows["native", "seq_ordinal_logit"]},
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for row in read_jsonl(artifact / "dojo_public_matches.jsonl"):
        family = row["family"]
        require(family in DOJO_FAMILIES and row.get("condition") == DOJO_FAMILIES[family], "Dojo family/condition differs")
        require(row.get("schema_version") == "public_dojo_match_v1", "Dojo row schema differs")
        require(row["official_star"] == 10, "Dojo primary placement is not star 10")
        tier = integer(row["expert_tier_order"], "Dojo tier")
        require(0 <= tier <= 4, "Dojo tier is outside the primary panel")
        key = row["chart_key"]
        oof = native_indices[family].get(key)
        require(oof is not None, "Dojo placement lacks native OOF score")
        require(oof["evaluation_original_label"] == 10 and oof["outer_fold"] == row["outer_fold"], "Dojo chart label/fold differs")
        close(number(row["latent_score"], "Dojo score"), oof["latent_score"], "Dojo/native OOF score")
        identity = (row["year"], row["dan"], row["position"], key)
        require(identity not in identities[family], "duplicate Dojo placement")
        identities[family].add(identity)
        grouped[family].append(row)
    require(set(grouped) == set(DOJO_FAMILIES), "Dojo family coverage differs")
    require(identities["global"] == identities["sequence"], "Dojo families have different placements")
    require(all(len(rows) == 47 and len({row["chart_key"] for row in rows}) == 47 for rows in grouped.values()), "Dojo placement count differs")
    require({row["expert_tier_order"] for row in grouped["global"]} == set(range(5)), "Dojo tier coverage differs")
    summaries = indexed_rows(read_jsonl(artifact / "dojo_criterion_summary.jsonl"), ("family",), "Dojo summary")
    require(set(summaries) == {(family,) for family in DOJO_FAMILIES}, "Dojo summary coverage differs")
    per_year_rows = indexed_rows(read_jsonl(artifact / "dojo_per_year_metrics.jsonl"), ("family", "year"), "Dojo per-year metrics")
    require(set(per_year_rows) == {(family, year) for family in DOJO_FAMILIES for year in DOJO_YEARS}, "Dojo per-year coverage differs")
    paper = evidence["dojo"]
    require(paper["placement_count"] == 47 and paper["year_count"] == 6 and paper["tier_count"] == 5 and paper["pair_count"] == 152, "paper Dojo cardinalities differ")
    points: dict[str, tuple[float, float]] = {}
    for family, rows in grouped.items():
        rho, pair_accuracy, pair_count, yearly = dojo_points(rows)
        require(pair_count == 152, f"Dojo {family}: eligible pair count differs")
        summary = summaries[family,]
        require(summary["placement_count"] == 47 and summary["year_count"] == 6 and summary["tier_count"] == 5 and summary["pair_count"] == 152, f"Dojo {family}: summary counts differ")
        close(rho, summary["macro_year_spearman"]["point"], f"Dojo {family} macro-year rho")
        close(pair_accuracy, summary["within_year_pair_accuracy"]["point"], f"Dojo {family} macro-year pair accuracy")
        for year, (count, year_rho, year_pair, year_pair_count) in yearly.items():
            reported = per_year_rows[family, year]
            require(reported["placement_count"] == count and reported["pair_count"] == year_pair_count, f"Dojo {family}/{year}: row counts differ")
            close(year_rho, reported["spearman_rho"], f"Dojo {family}/{year} rho")
            close(year_pair, reported["within_year_pair_accuracy"], f"Dojo {family}/{year} pair accuracy")
        if family == "global":
            close(rho, paper["spearman_rho"], "paper Dojo rho")
            close(pair_accuracy, paper["pair_accuracy"], "paper Dojo pair accuracy")
        else:
            close(rho, paper["sequence_spearman_rho"], "paper sequence Dojo rho")
            close(pair_accuracy, paper["sequence_pair_accuracy"], "paper sequence Dojo pair accuracy")
        points[family] = rho, pair_accuracy
    return points


def read_macros(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PublicAnalysisError(f"cannot read {path}: {exc}") from exc
    matches = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text)
    macros = dict(matches)
    require(len(macros) == len(matches), f"{path}: duplicate numeric macro")
    return macros


def numeric_macro(macros: dict[str, str], name: str) -> float:
    raw = macros[name]
    require(re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", raw) is not None, f"{name}: expected numeric macro")
    result = float(raw)
    require(math.isfinite(result), f"{name}: non-finite macro")
    return result


def check_ceiling_points(
    paper: Path,
    global_rows: dict[tuple[str, str], list[dict[str, Any]]],
    sequence_rows: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    macros = read_macros(paper / "generated/ceiling_compression_numbers.tex")
    delta = numeric_macro(macros, "CeilDelta")
    require(delta == 0.05, "paper cap band changed")
    families = (
        ("Gbdt", "gbdt", global_rows, "latent_score"),
        ("Huber", "ordinary_huber", global_rows, "latent_score"),
        ("Ridge", "ridge", global_rows, "latent_score"),
        ("OneSided", "topcoded_huber", global_rows, "latent_score"),
        ("Reduction", "ordinal_logit_reduction", global_rows, "latent_score"),
        ("SeqHuber", "seq_ordinary_huber", sequence_rows, "expected_observed_label"),
        ("SeqOneSided", "seq_topcoded_huber", sequence_rows, "expected_observed_label"),
        ("SeqLogit", "seq_ordinal_logit", sequence_rows, "expected_observed_label"),
    )
    for tag, condition, groups, rating_field in families:
        rows = [row for row in groups["cap8", condition] if row["evaluation_original_label"] >= 8]
        require(len(rows) == 699, f"cap-8 {condition} panel count differs")
        near = [row for row in rows if abs(number(row[rating_field], "rating output") - 8) <= delta + 1e-12]
        above = [row for row in rows if number(row[rating_field], "rating output") > 8 + delta]
        require(len(near) == numeric_macro(macros, f"Ceil{tag}NearCount"), f"{condition} near-cap count differs")
        require(round(100 * len(near) / len(rows)) == numeric_macro(macros, f"Ceil{tag}NearPct"), f"{condition} near-cap percent differs")
        require(round(100 * len(above) / len(rows)) == numeric_macro(macros, f"Ceil{tag}AbovePct"), f"{condition} above-cap percent differs")
        if f"Ceil{tag}NearRho" in macros:
            require(len(near) >= 30, f"{condition}: near-cap rho has too few rows")
            rho = float(spearmanr([row["evaluation_original_label"] for row in near], [row["latent_score"] for row in near]).statistic)
            require(math.isfinite(rho), f"{condition}: near-cap rho undefined")
            require(round(rho, 3) == numeric_macro(macros, f"Ceil{tag}NearRho"), f"{condition}: near-cap rho differs")
        else:
            require(len(near) < 30, f"{condition}: paper omits reproducible near-cap rho")
    for tag, condition in (("Gbdt", "gbdt"), ("Huber", "ordinary_huber"), ("OneSided", "topcoded_huber")):
        rows = global_rows["cap8", condition]
        for label, suffix in ((8, "Eight"), (9, "Nine"), (10, "Ten")):
            scores = [row["latent_score"] for row in rows if row["evaluation_original_label"] == label]
            require(scores, f"{condition}: missing original star {label}")
            median = float(np.median(scores))
            require(round(median, 2) == numeric_macro(macros, f"Ceil{tag}Median{suffix}"), f"{condition} median star {label} differs")


def analyze_public_release(root: Path) -> dict[str, Any]:
    artifact = root / ARTIFACT
    global_cases = {(scenario, condition) for scenario in SCENARIOS for condition in GLOBAL_CONDITIONS}
    global_rows, reference = load_oof(artifact / "global_oof_scores.jsonl", schema="public_global_oof_score", expected_cases=global_cases)
    sequence_rows, _ = load_oof(artifact / "sequence_oof_scores.jsonl", schema="public_sequence_oof_score", expected_cases=set(SEQUENCE_CASES), reference=reference)
    global_points = {case: rank_points(rows, SCENARIOS[case[0]] if case[0] != "native" else None) for case, rows in global_rows.items()}
    sequence_points = {case: rank_points(rows, SCENARIOS[case[0]] if case[0] != "native" else None) for case, rows in sequence_rows.items()}
    check_metric_rows(artifact, global_points, sequence_points)
    check_contrasts(artifact, global_points, sequence_points)
    evidence = check_table_and_evidence(artifact, global_points, sequence_points)
    dojo = check_dojo(artifact, global_rows, sequence_rows, evidence)
    check_ceiling_points(root / PAPER, global_rows, sequence_rows)
    return {
        "status": "pass",
        "charts": len(reference),
        "global_oof_rows": sum(map(len, global_rows.values())),
        "sequence_oof_rows": sum(map(len, sequence_rows.values())),
        "cap8_hidden_charts": global_points["cap8", "gbdt"].panel_count,
        "cap8_comparable_pairs": global_points["cap8", "gbdt"].comparable_pairs,
        "dojo_placements_per_family": 47,
        "dojo_eligible_pairs": 152,
        "dojo_global_rho": dojo["global"][0],
        "dojo_global_pair_accuracy": dojo["global"][1],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        result = analyze_public_release(args.root)
    except (PublicAnalysisError, KeyError, TypeError, OSError, UnicodeError) as exc:
        print(f"public point analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
