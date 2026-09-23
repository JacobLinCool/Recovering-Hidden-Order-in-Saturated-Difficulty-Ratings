from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.research import build_topcoded_oof_tables as builder


def _feature(
    chart_id: str,
    *,
    group: str,
    star: int,
) -> dict[str, object]:
    return {
        "chart_id": chart_id,
        "song_id": f"song-{chart_id}",
        "title": f"Title {chart_id}",
        "normalized_title": group,
        "difficulty": "oni",
        "difficulty_id": 3,
        "star": star,
        "primary_split": "train",
    }


class BuildTopcodedOOFTablesTest(unittest.TestCase):
    def test_historical_source_cohort_must_be_unique_and_complete(self) -> None:
        cohort_hash = "a" * 64
        self.assertEqual(
            builder.unique_complete_source_hash(
                {
                    ("cap8", "ordinary"): {cohort_hash: [object()]},
                    ("cap8", "topcoded"): {cohort_hash: [object()]},
                }
            ),
            cohort_hash,
        )
        with self.assertRaises(builder.CanonicalEvidenceError):
            builder.unique_complete_source_hash(
                {
                    ("cap8", "ordinary"): {
                        cohort_hash: [object()],
                        "b" * 64: [object()],
                    },
                    ("cap8", "topcoded"): {
                        cohort_hash: [object()],
                        "b" * 64: [object()],
                    },
                }
            )

    def test_fold_assignments_reconstruct_leakage_free_partitions(self) -> None:
        features = {
            f"chart-{index}": _feature(
                f"chart-{index}",
                group=f"group-{index}",
                star=7 + index % 4,
            )
            for index in range(6)
        }
        rows = [
            {
                "chart_id": chart_id,
                "song_id": feature["song_id"],
                "normalized_title": feature["normalized_title"],
                "original_star": feature["star"],
                "outer_test_fold": index % 5,
                "inner_validation_for_outer_folds": [(index - 1) % 5],
            }
            for index, (chart_id, feature) in enumerate(features.items())
        ]

        with patch.object(builder, "EXPECTED_CHART_COUNT", len(rows)):
            canonical, reconstructed = builder._validate_fold_assignments(
                rows,
                features_by_chart=features,
                expected_assignment_by_group={
                    str(row["normalized_title"]): (
                        int(row["outer_test_fold"]),
                        tuple(row["inner_validation_for_outer_folds"]),
                    )
                    for row in rows
                },
                context="test",
            )

        self.assertEqual(len(canonical), len(rows))
        for partitions in reconstructed.values():
            self.assertFalse(partitions["train"] & partitions["validation"])
            self.assertFalse(partitions["train"] & partitions["test"])
            self.assertFalse(partitions["validation"] & partitions["test"])

    def test_fold_assignments_reject_split_title_group(self) -> None:
        features = {
            "chart-a": _feature("chart-a", group="same", star=8),
            "chart-b": _feature("chart-b", group="same", star=9),
        }
        rows = [
            {
                "chart_id": "chart-a",
                "song_id": "song-chart-a",
                "normalized_title": "same",
                "original_star": 8,
                "outer_test_fold": 0,
                "inner_validation_for_outer_folds": [1],
            },
            {
                "chart_id": "chart-b",
                "song_id": "song-chart-b",
                "normalized_title": "same",
                "original_star": 9,
                "outer_test_fold": 1,
                "inner_validation_for_outer_folds": [0],
            },
        ]
        with (
            patch.object(builder, "EXPECTED_CHART_COUNT", len(rows)),
            self.assertRaisesRegex(
                builder.CanonicalEvidenceError,
                "splits title group",
            ),
        ):
            builder._validate_fold_assignments(
                rows,
                features_by_chart=features,
                expected_assignment_by_group={"same": (0, (1,))},
                context="test",
            )

    def test_search_rows_require_the_deterministic_minimum(self) -> None:
        config = {"ridge_alpha_grid": [0.1, 1.0, 10.0]}
        rows = []
        for fold in range(2):
            for alpha, mae in ((0.1, 0.4), (1.0, 0.2), (10.0, 0.3)):
                rows.append(
                    {
                        "scenario": "cap8",
                        "condition": "ridge",
                        "outer_fold": fold,
                        "parameters": {"alpha": alpha},
                        "validation_observed_mae": mae,
                        "selected": alpha == 1.0,
                    }
                )
        with patch.object(builder, "EXPECTED_OUTER_FOLDS", 2):
            canonical, selected = builder._validate_search_rows(
                rows,
                scenario="cap8",
                condition="ridge",
                config=config,
                provenance={"run_id": "run", "raw_attempt": "attempt"},
                context="test",
            )
        self.assertEqual(len(canonical), 6)
        self.assertEqual(selected[0], ({"alpha": 1.0}, 0.2))

        rows[0]["selected"] = True
        with (
            patch.object(builder, "EXPECTED_OUTER_FOLDS", 2),
            self.assertRaisesRegex(
                builder.CanonicalEvidenceError,
                "exactly one",
            ),
        ):
            builder._validate_search_rows(
                rows,
                scenario="cap8",
                condition="ridge",
                config=config,
                provenance={"run_id": "run", "raw_attempt": "attempt"},
                context="test",
            )

    def test_prediction_rows_require_exact_oof_and_top_coding_contract(
        self,
    ) -> None:
        features = {
            "chart-a": _feature("chart-a", group="a", star=8),
            "chart-b": _feature("chart-b", group="b", star=10),
        }
        assignments = {
            "chart-a": {"outer_test_fold": 0},
            "chart-b": {"outer_test_fold": 1},
        }

        def row(chart_id: str, fold: int, latent: float) -> dict[str, object]:
            feature = features[chart_id]
            original = float(feature["star"])
            expected = latent
            return {
                "scenario": "cap8",
                "condition": "topcoded_huber",
                "seed": 2027,
                "outer_fold": fold,
                "chart_id": chart_id,
                "song_id": feature["song_id"],
                "title": feature["title"],
                "normalized_title": feature["normalized_title"],
                "difficulty": "oni",
                "difficulty_id": 3,
                "original_star": original,
                "observed_star": min(original, 8.0),
                "active_cap": 8.0,
                "is_top_coded": original >= 8.0,
                "is_strictly_hidden": original > 8.0,
                "latent_score": latent,
                "expected_observed_label": expected,
                "clipped_observed_prediction": min(max(expected, 1.0), 8.0),
                "primary_split": "train",
            }

        rows = [row("chart-a", 0, 7.5), row("chart-b", 1, 9.5)]
        with patch.object(builder, "EXPECTED_CHART_COUNT", len(rows)):
            canonical = builder._validate_prediction_rows(
                rows,
                scenario="cap8",
                condition="topcoded_huber",
                seed=2027,
                cap=8.0,
                assignments_by_chart=assignments,
                features_by_chart=features,
                provenance={"run_id": "run", "raw_attempt": "attempt"},
                context="test",
            )
        self.assertEqual(len(canonical), 2)

        rows[1]["observed_star"] = 10.0
        with (
            patch.object(builder, "EXPECTED_CHART_COUNT", len(rows)),
            self.assertRaisesRegex(
                builder.CanonicalEvidenceError,
                "top-coding contract",
            ),
        ):
            builder._validate_prediction_rows(
                rows,
                scenario="cap8",
                condition="topcoded_huber",
                seed=2027,
                cap=8.0,
                assignments_by_chart=assignments,
                features_by_chart=features,
                provenance={"run_id": "run", "raw_attempt": "attempt"},
                context="test",
            )


if __name__ == "__main__":
    unittest.main()
