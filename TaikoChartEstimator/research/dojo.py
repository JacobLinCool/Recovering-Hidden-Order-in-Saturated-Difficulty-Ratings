"""Dan-i Dojo course data and external-criterion statistics.

The Dan-i Dojo criterion is deliberately separate from model fitting.  Course
placements are joined to already frozen native out-of-fold (OOF) chart scores
only after those scores have been produced.  The primary panel holds the
official star rating and chart course fixed, then asks whether recovered scores
increase across the expert progression from 10th Dan through Tatsujin.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from TaikoChartEstimator.research.data import normalize_title

DOJO_SCHEMA_VERSION = "icassp2027_dojo_placement_v1"
MATCH_SCHEMA_VERSION = "icassp2027_dojo_match_v1"
HIGH_DAN_ORDER = {
    "10dan": 0,
    "kuroto": 1,
    "meijin": 2,
    "chojin": 3,
    "tatsujin": 4,
}
REGULAR_DAN_ORDER = {
    **{f"{level}kyu": 5 - level for level in range(5, 0, -1)},
    **{f"{level}dan": level + 4 for level in range(1, 11)},
    "kuroto": 15,
    "meijin": 16,
    "chojin": 17,
    "tatsujin": 18,
}


class DojoDataError(ValueError):
    """Raised when course or score evidence violates the frozen contract."""


@dataclass(frozen=True)
class MetricEstimate:
    """One point estimate with bootstrap and blocked-permutation evidence."""

    point: float
    ci_low: float
    ci_high: float
    valid_bootstrap_replicates: int
    undefined_bootstrap_replicates: int
    permutation_p_value: float
    valid_permutation_replicates: int
    undefined_permutation_replicates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "bootstrap_ci_low": self.ci_low,
            "bootstrap_ci_high": self.ci_high,
            "valid_bootstrap_replicates": self.valid_bootstrap_replicates,
            "undefined_bootstrap_replicates": self.undefined_bootstrap_replicates,
            "permutation_p_value": self.permutation_p_value,
            "valid_permutation_replicates": self.valid_permutation_replicates,
            "undefined_permutation_replicates": self.undefined_permutation_replicates,
        }


def validate_course_placements(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate a complete, unique course-placement snapshot."""

    if not rows:
        raise DojoDataError("course placement snapshot is empty")
    identities: set[tuple[int, str, int]] = set()
    for row in rows:
        if row.get("schema_version") != DOJO_SCHEMA_VERSION:
            raise DojoDataError("unsupported course placement schema")
        year = int(row["year"])
        dan = str(row["dan"])
        position = int(row["position"])
        if dan not in REGULAR_DAN_ORDER:
            raise DojoDataError(f"unknown regular Dan-i rank: {dan!r}")
        if int(row["tier_order"]) != REGULAR_DAN_ORDER[dan]:
            raise DojoDataError(f"incorrect tier order for {dan!r}")
        if position not in {1, 2, 3}:
            raise DojoDataError("course position must be 1, 2, or 3")
        identity = (year, dan, position)
        if identity in identities:
            raise DojoDataError(f"duplicate course placement: {identity}")
        identities.add(identity)
        title = str(row.get("title", ""))
        if not title or normalize_title(title) != row.get("normalized_title"):
            raise DojoDataError(f"invalid normalized title for {identity}")
        if row.get("difficulty") not in {"easy", "normal", "hard", "oni", "ura"}:
            raise DojoDataError(f"invalid difficulty for {identity}")
        star = int(row["official_star"])
        if not 1 <= star <= 10:
            raise DojoDataError(f"invalid official star for {identity}")
        if not str(row.get("source_api_url", "")).startswith("https://"):
            raise DojoDataError(f"missing HTTPS source API URL for {identity}")


def _collapse_score_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scenario") != scenario or row.get("condition") != condition:
            continue
        if row.get("difficulty") not in {None, "oni"}:
            continue
        normalized = str(row["normalized_title"])
        if normalize_title(normalized) != normalized:
            raise DojoDataError("OOF score contains a noncanonical title key")
        selected[normalized].append(dict(row))
    if not selected:
        raise DojoDataError(
            f"no OOF rows for scenario={scenario!r}, condition={condition!r}"
        )
    return selected


def _select_title_score(
    placement: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str]:
    if not candidates:
        return None, "title_absent_from_oof_panel"
    if len(candidates) == 1:
        return candidates[0], "exact_normalized_title"

    exact_title = [
        row for row in candidates if str(row.get("title", "")) == placement["title"]
    ]
    if len(exact_title) == 1:
        return exact_title[0], "exact_display_title"
    if len(exact_title) > 1:
        candidates = exact_title

    labels = {float(row["original_star"]) for row in candidates}
    scores = {round(float(row["latent_score"]), 12) for row in candidates}
    if len(labels) == 1 and len(scores) == 1:
        return candidates[0], "identical_duplicate_scores"
    return None, "ambiguous_title_collision"


def match_course_placements(
    placements: Sequence[Mapping[str, Any]],
    score_rows: Iterable[Mapping[str, Any]],
    *,
    scenario: str,
    condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join public course facts to frozen OOF scores without fuzzy matching."""

    validate_course_placements(placements)
    scores = _collapse_score_rows(
        score_rows,
        scenario=scenario,
        condition=condition,
    )
    matched: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for placement in placements:
        candidates = scores.get(str(placement["normalized_title"]), [])
        score, method = _select_title_score(placement, candidates)
        identity = {
            "year": int(placement["year"]),
            "dan": str(placement["dan"]),
            "position": int(placement["position"]),
            "title": str(placement["title"]),
            "normalized_title": str(placement["normalized_title"]),
            "difficulty": str(placement["difficulty"]),
            "official_star": int(placement["official_star"]),
        }
        if placement["difficulty"] != "oni":
            exclusions.append({**identity, "reason": "non_oni_course"})
            continue
        if score is None:
            exclusions.append({**identity, "reason": method})
            continue
        score_label = int(float(score["original_star"]))
        if score_label != int(placement["official_star"]):
            exclusions.append(
                {
                    **identity,
                    "reason": "official_star_mismatch",
                    "oof_official_star": score_label,
                }
            )
            continue
        matched.append(
            {
                "schema_version": MATCH_SCHEMA_VERSION,
                **identity,
                "tier_order": int(placement["tier_order"]),
                "high_tier_order": HIGH_DAN_ORDER.get(str(placement["dan"])),
                "song_no": str(placement["song_no"]),
                "source_course_url": str(placement["source_course_url"]),
                "scenario": scenario,
                "condition": condition,
                "chart_id": str(score["chart_id"]),
                "outer_fold": int(score["outer_fold"]),
                "latent_score": float(score["latent_score"]),
                "match_method": method,
            }
        )
    matched.sort(
        key=lambda row: (
            row["year"],
            row["tier_order"],
            row["position"],
            row["chart_id"],
        )
    )
    exclusions.sort(
        key=lambda row: (row["year"], REGULAR_DAN_ORDER[row["dan"]], row["position"])
    )
    return matched, exclusions


def primary_ten_star_panel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    panel = [
        dict(row)
        for row in rows
        if row["dan"] in HIGH_DAN_ORDER and int(row["official_star"]) == 10
    ]
    for row in panel:
        if int(row["high_tier_order"]) != HIGH_DAN_ORDER[str(row["dan"])]:
            raise DojoDataError("primary panel has an invalid high-tier order")
    return panel


def _group_by_year(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["year"])].append(row)
    return dict(sorted(grouped.items()))


def macro_year_spearman(rows: Sequence[Mapping[str, Any]]) -> float:
    correlations: list[float] = []
    for year_rows in _group_by_year(rows).values():
        tiers = np.asarray([float(row["analysis_tier"]) for row in year_rows])
        scores = np.asarray([float(row["latent_score"]) for row in year_rows])
        if np.unique(tiers).size < 2 or np.unique(scores).size < 2:
            continue
        value = float(stats.spearmanr(tiers, scores).statistic)
        if math.isfinite(value):
            correlations.append(value)
    if not correlations:
        return float("nan")
    return float(np.mean(correlations))


def within_year_pair_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_equal_star: bool = False,
    maximum_star: int | None = None,
) -> tuple[float, int]:
    """Macro-average strict concordance across years.

    Prediction ties receive no credit, matching the hidden-tail estimand used
    elsewhere in the paper.  Each year receives equal weight in the point
    estimate; the returned count is the total number of eligible pairs.
    """

    per_year: list[float] = []
    pair_count = 0
    for year_rows in _group_by_year(rows).values():
        correct = 0
        total = 0
        for left_index, left in enumerate(year_rows):
            for right in year_rows[left_index + 1 :]:
                left_tier = float(left["analysis_tier"])
                right_tier = float(right["analysis_tier"])
                if left_tier == right_tier:
                    continue
                left_star = int(left["official_star"])
                right_star = int(right["official_star"])
                if require_equal_star and left_star != right_star:
                    continue
                if maximum_star is not None and (
                    left_star > maximum_star or right_star > maximum_star
                ):
                    continue
                direction = (float(left["latent_score"]) - float(right["latent_score"])) * (
                    left_tier - right_tier
                )
                correct += int(direction > 0.0)
                total += 1
        if total:
            per_year.append(correct / total)
            pair_count += total
    if not per_year:
        return float("nan"), 0
    return float(np.mean(per_year)), pair_count


def _metric_value(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    if metric == "macro_year_spearman":
        return macro_year_spearman(rows)
    if metric == "within_year_pair_accuracy":
        return within_year_pair_accuracy(rows)[0]
    raise DojoDataError(f"unknown Dojo metric: {metric}")


def _resample_years(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    grouped = _group_by_year(rows)
    years = np.asarray(sorted(grouped), dtype=np.int64)
    sampled_years = rng.choice(years, size=len(years), replace=True)
    sampled: list[dict[str, Any]] = []
    for cluster_index, year in enumerate(sampled_years):
        source = grouped[int(year)]
        indices = rng.integers(0, len(source), size=len(source))
        for index in indices:
            sampled.append(
                {
                    **source[int(index)],
                    "year": cluster_index,
                }
            )
    return sampled


def estimate_with_uncertainty(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> MetricEstimate:
    """Estimate a metric with hierarchical year bootstrap and blocked permutation."""

    if bootstrap_replicates <= 0 or permutation_replicates <= 0:
        raise DojoDataError("replicate counts must be positive")
    point = _metric_value(rows, metric)
    if not math.isfinite(point):
        raise DojoDataError(f"primary metric is undefined: {metric}")
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(bootstrap_replicates):
        value = _metric_value(_resample_years(rows, rng), metric)
        if math.isfinite(value):
            bootstrap_values.append(value)
    if not bootstrap_values:
        raise DojoDataError(f"all bootstrap replicates are undefined: {metric}")

    permutation_values: list[float] = []
    grouped = _group_by_year(rows)
    for _ in range(permutation_replicates):
        permuted: list[dict[str, Any]] = []
        for year, year_rows in grouped.items():
            tiers = rng.permutation([row["analysis_tier"] for row in year_rows])
            for row, tier in zip(year_rows, tiers, strict=True):
                permuted.append({**row, "year": year, "analysis_tier": float(tier)})
        value = _metric_value(permuted, metric)
        if math.isfinite(value):
            permutation_values.append(value)
    if not permutation_values:
        raise DojoDataError(f"all permutation replicates are undefined: {metric}")
    upper_tail = sum(value >= point for value in permutation_values)
    return MetricEstimate(
        point=point,
        ci_low=float(np.quantile(bootstrap_values, 0.025)),
        ci_high=float(np.quantile(bootstrap_values, 0.975)),
        valid_bootstrap_replicates=len(bootstrap_values),
        undefined_bootstrap_replicates=bootstrap_replicates - len(bootstrap_values),
        permutation_p_value=(upper_tail + 1.0) / (len(permutation_values) + 1.0),
        valid_permutation_replicates=len(permutation_values),
        undefined_permutation_replicates=(
            permutation_replicates - len(permutation_values)
        ),
    )


def with_analysis_tier(
    rows: Sequence[Mapping[str, Any]], *, high_only: bool
) -> list[dict[str, Any]]:
    key = "high_tier_order" if high_only else "tier_order"
    output: list[dict[str, Any]] = []
    for row in rows:
        tier = row.get(key)
        if tier is None:
            continue
        output.append({**row, "analysis_tier": float(tier)})
    return output


def leave_one_year_out(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    years = sorted({int(row["year"]) for row in rows})
    output: list[dict[str, Any]] = []
    for year in years:
        retained = [row for row in rows if int(row["year"]) != year]
        pair_accuracy, pair_count = within_year_pair_accuracy(retained)
        output.append(
            {
                "held_out_year": year,
                "retained_placements": len(retained),
                "macro_year_spearman": macro_year_spearman(retained),
                "within_year_pair_accuracy": pair_accuracy,
                "pair_count": pair_count,
            }
        )
    return output
