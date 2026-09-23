from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research import analyze_public_release as public


ROOT = Path(__file__).resolve().parents[1]


class PublicPointAnalysisTest(unittest.TestCase):
    def test_current_release_recomputes_paper_points(self) -> None:
        result = public.analyze_public_release(ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["global_oof_rows"], 28_644)
        self.assertEqual(result["sequence_oof_rows"], 8_184)
        self.assertEqual(result["cap8_hidden_charts"], 699)
        self.assertEqual(result["cap8_comparable_pairs"], 159_318)
        self.assertEqual(result["dojo_eligible_pairs"], 152)

    def test_prediction_tie_gets_no_pair_credit(self) -> None:
        rows = [
            {"evaluation_original_label": label, "latent_score": score}
            for label, score in ((1, 1.0), (2, 1.0), (3, 3.0), (3, 2.0))
        ]
        point = public.rank_points(rows)
        self.assertEqual(point.comparable_pairs, 5)
        self.assertEqual(point.correct_pairs, 4)
        self.assertEqual(point.prediction_ties, 1)
        self.assertEqual(point.pair_accuracy, 0.8)

    def test_dojo_pair_point_is_macro_year_mean(self) -> None:
        rows = [
            {"year": 2020, "expert_tier_order": tier, "latent_score": score}
            for tier, score in ((0, 3.0), (1, 2.0), (2, 1.0))
        ]
        for year in range(2021, 2026):
            rows.extend(
                {"year": year, "expert_tier_order": tier, "latent_score": score}
                for tier, score in ((0, 0.0), (1, 1.0))
            )
        rho, pair_accuracy, pair_count, yearly = public.dojo_points(rows)
        self.assertAlmostEqual(rho, 2 / 3)
        self.assertAlmostEqual(pair_accuracy, 5 / 6)
        self.assertEqual(pair_count, 8)
        self.assertEqual(yearly[2020][3], 3)

    def test_missing_oof_row_fails_closed(self) -> None:
        original = public.read_jsonl

        def without_one_chart(path: Path):
            rows = original(path)
            return rows[:-1] if path.name == "global_oof_scores.jsonl" else rows

        with patch.object(public, "read_jsonl", side_effect=without_one_chart):
            with self.assertRaisesRegex(public.PublicAnalysisError, "incomplete OOF case"):
                public.analyze_public_release(ROOT)

    def test_changed_public_score_fails_against_reported_points(self) -> None:
        original = public.read_jsonl

        def with_changed_score(path: Path):
            rows = original(path)
            if path.name == "global_oof_scores.jsonl":
                row = next(
                    row
                    for row in rows
                    if row["scenario"] == "cap8"
                    and row["condition"] == "gbdt"
                    and row["evaluation_original_label"] == 10
                )
                row["latent_score"] += 100
            return rows

        with patch.object(public, "read_jsonl", side_effect=with_changed_score):
            with self.assertRaisesRegex(public.PublicAnalysisError, "correct pair count differs"):
                public.analyze_public_release(ROOT)


if __name__ == "__main__":
    unittest.main()
