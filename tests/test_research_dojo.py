from __future__ import annotations

import math
import unittest

from TaikoChartEstimator.research.data import normalize_title
from TaikoChartEstimator.research.dojo import (
    DOJO_SCHEMA_VERSION,
    DojoDataError,
    estimate_with_uncertainty,
    macro_year_spearman,
    match_course_placements,
    validate_course_placements,
    within_year_pair_accuracy,
)


def placement(
    title: str,
    *,
    year: int = 2025,
    dan: str = "10dan",
    position: int = 1,
    star: int = 10,
) -> dict:
    tier = {"10dan": 14, "kuroto": 15, "meijin": 16}[dan]
    return {
        "schema_version": DOJO_SCHEMA_VERSION,
        "year": year,
        "dan": dan,
        "tier_order": tier,
        "position": position,
        "title": title,
        "normalized_title": normalize_title(title),
        "difficulty": "oni",
        "official_star": star,
        "song_no": f"{year}-{position}",
        "source_api_url": "https://example.test/api",
        "source_course_url": "https://example.test/course",
    }


def score(title: str, value: float, *, star: int = 10, chart: str = "c") -> dict:
    return {
        "scenario": "native",
        "condition": "topcoded_huber",
        "difficulty": "oni",
        "normalized_title": normalize_title(title),
        "title": title,
        "original_star": star,
        "latent_score": value,
        "chart_id": chart,
        "outer_fold": 0,
    }


class DojoResearchTest(unittest.TestCase):
    def test_course_snapshot_rejects_duplicate_identity(self) -> None:
        row = placement("Alpha")
        validate_course_placements([row])
        with self.assertRaisesRegex(DojoDataError, "duplicate"):
            validate_course_placements([row, dict(row)])

    def test_join_is_exact_and_never_fuzzy(self) -> None:
        placements = [placement("Alpha"), placement("Alphb", position=2)]
        matched, excluded = match_course_placements(
            placements,
            [score("Alpha", 11.0)],
            scenario="native",
            condition="topcoded_huber",
        )
        self.assertEqual([row["title"] for row in matched], ["Alpha"])
        self.assertEqual(excluded[0]["reason"], "title_absent_from_oof_panel")

    def test_ambiguous_title_collision_fails_closed(self) -> None:
        rows = [score("Alpha", 10.0, chart="a"), score("Alpha", 11.0, chart="b")]
        rows[0].pop("title")
        rows[1].pop("title")
        matched, excluded = match_course_placements(
            [placement("Alpha")],
            rows,
            scenario="native",
            condition="topcoded_huber",
        )
        self.assertEqual(matched, [])
        self.assertEqual(excluded[0]["reason"], "ambiguous_title_collision")

    def test_year_macro_metrics_and_prediction_ties(self) -> None:
        rows = [
            {"year": 2024, "analysis_tier": 0, "latent_score": 1.0, "official_star": 10},
            {"year": 2024, "analysis_tier": 1, "latent_score": 2.0, "official_star": 10},
            {"year": 2025, "analysis_tier": 0, "latent_score": 3.0, "official_star": 9},
            {"year": 2025, "analysis_tier": 1, "latent_score": 3.0, "official_star": 9},
        ]
        self.assertAlmostEqual(macro_year_spearman(rows), 1.0)
        accuracy, count = within_year_pair_accuracy(rows)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(accuracy, 0.5)
        lower, lower_count = within_year_pair_accuracy(
            rows, require_equal_star=True, maximum_star=9
        )
        self.assertEqual(lower_count, 1)
        self.assertEqual(lower, 0.0)

    def test_uncertainty_is_deterministic_and_blocked_by_year(self) -> None:
        rows = [
            {"year": year, "analysis_tier": tier, "latent_score": float(tier), "official_star": 10}
            for year in (2024, 2025)
            for tier in (0, 1, 2)
        ]
        first = estimate_with_uncertainty(
            rows,
            metric="within_year_pair_accuracy",
            bootstrap_replicates=100,
            permutation_replicates=100,
            seed=2027,
        )
        second = estimate_with_uncertainty(
            rows,
            metric="within_year_pair_accuracy",
            bootstrap_replicates=100,
            permutation_replicates=100,
            seed=2027,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first.valid_bootstrap_replicates
            + first.undefined_bootstrap_replicates,
            100,
        )
        self.assertGreater(first.valid_bootstrap_replicates, 0)
        self.assertEqual(
            first.valid_permutation_replicates
            + first.undefined_permutation_replicates,
            100,
        )
        self.assertTrue(math.isfinite(first.permutation_p_value))

if __name__ == "__main__":
    unittest.main()
