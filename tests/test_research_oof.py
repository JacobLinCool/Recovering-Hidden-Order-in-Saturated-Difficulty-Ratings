from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import spearmanr

from TaikoChartEstimator.research.oof import (
    build_hidden_tail_panel,
    build_song_group_folds,
    hidden_tail_rank_metrics,
    paired_song_group_bootstrap_difference,
    top_code_targets,
)


class ResearchOOFTest(unittest.TestCase):
    def test_group_folds_are_balanced_deterministic_and_leakage_free(
        self,
    ) -> None:
        groups = [f"song-{index}" for index in range(17)]
        first = build_song_group_folds(
            [*groups, "song-3", "song-8"],
            n_folds=5,
            seed=2027,
            validation_fraction=0.2,
        )
        second = build_song_group_folds(
            list(reversed(groups)),
            n_folds=5,
            seed=2027,
            validation_fraction=0.2,
        )
        self.assertEqual(first, second)

        test_memberships = [
            group for fold in first for group in fold.test_groups
        ]
        self.assertCountEqual(test_memberships, groups)
        self.assertEqual(len(test_memberships), len(set(test_memberships)))
        test_sizes = [len(fold.test_groups) for fold in first]
        self.assertLessEqual(max(test_sizes) - min(test_sizes), 1)

        all_groups = set(groups)
        for fold in first:
            train = set(fold.train_groups)
            validation = set(fold.validation_groups)
            test = set(fold.test_groups)
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertTrue(test)
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, all_groups)

    def test_group_fold_validation_rejects_impossible_partitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_song_group_folds(["a", "b"], n_folds=3)
        with self.assertRaisesRegex(ValueError, "at least two non-test"):
            build_song_group_folds(["a", "b"], n_folds=2)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            build_song_group_folds(
                ["a", "b", "c", "d"],
                n_folds=2,
                validation_fraction=1.0,
            )

    def test_top_coding_is_course_selective_and_retains_originals(self) -> None:
        targets = np.asarray([6.0, 7.0, 8.0, 9.0, 10.0, 9.0])
        courses = np.asarray([3, 3, 3, 3, 3, 2])
        coded = top_code_targets(
            targets,
            courses,
            cap=8,
            selected_course_ids={3},
        )
        self.assertEqual(coded.original_targets, (6.0, 7.0, 8.0, 9.0, 10.0, 9.0))
        self.assertEqual(coded.observed_targets, (6.0, 7.0, 8.0, 8.0, 8.0, 9.0))
        self.assertEqual(
            coded.is_top_coded,
            (False, False, True, True, True, False),
        )
        self.assertEqual(
            coded.is_hidden_tail,
            (False, False, False, True, True, False),
        )
        np.testing.assert_array_equal(
            targets, np.asarray([6.0, 7.0, 8.0, 9.0, 10.0, 9.0])
        )

    def test_top_coding_rejects_non_integral_course_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            top_code_targets(
                [8.0, 9.0],
                [3, 3.5],
                cap=8,
                selected_course_ids={3},
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            top_code_targets(
                [8.0],
                [3],
                cap=8,
                selected_course_ids=set(),
            )

    def test_hidden_tail_metrics_handle_target_and_prediction_ties(self) -> None:
        coded = top_code_targets(
            [8.0, 9.0, 10.0, 10.0, 7.0],
            [3, 3, 3, 3, 3],
            cap=8,
            selected_course_ids={3},
        )
        panel = build_hidden_tail_panel(
            coded,
            ["a", "b", "c", "d", "e"],
        )
        predictions = [0.0, 0.0, 2.0, 1.0, -1.0]
        metrics = hidden_tail_rank_metrics(panel, predictions)

        expected_rho = float(
            spearmanr(
                [8.0, 9.0, 10.0, 10.0],
                [0.0, 0.0, 2.0, 1.0],
            ).statistic
        )
        self.assertAlmostEqual(metrics["spearman_rho"], expected_rho)
        self.assertEqual(metrics["hidden_tail_count"], 4)
        self.assertEqual(metrics["strictly_hidden_count"], 3)
        self.assertEqual(metrics["total_pair_count"], 6)
        self.assertEqual(metrics["target_tie_pair_count"], 1)
        self.assertEqual(metrics["comparable_pair_count"], 5)
        self.assertEqual(metrics["strict_pair_correct_count"], 4)
        self.assertEqual(metrics["prediction_tie_comparable_pair_count"], 1)
        self.assertAlmostEqual(
            metrics["strict_comparable_pair_accuracy"], 0.8
        )

    def test_empty_and_constant_panels_report_undefined_metrics(self) -> None:
        empty_coded = top_code_targets(
            [6.0, 7.0],
            [3, 3],
            cap=8,
            selected_course_ids={3},
        )
        empty_panel = build_hidden_tail_panel(empty_coded, ["a", "b"])
        empty = hidden_tail_rank_metrics(empty_panel, [1.0, 2.0])
        self.assertEqual(empty["hidden_tail_count"], 0)
        self.assertIsNone(empty["spearman_rho"])
        self.assertEqual(
            empty["spearman_undefined_reason"], "fewer_than_two_rows"
        )
        self.assertIsNone(empty["strict_comparable_pair_accuracy"])
        self.assertEqual(
            empty["strict_pair_accuracy_undefined_reason"],
            "no_comparable_target_pairs",
        )

        coded = top_code_targets(
            [8.0, 9.0, 10.0],
            [3, 3, 3],
            cap=8,
            selected_course_ids={3},
        )
        panel = build_hidden_tail_panel(coded, ["a", "b", "c"])
        constant = hidden_tail_rank_metrics(panel, [1.0, 1.0, 1.0])
        self.assertIsNone(constant["spearman_rho"])
        self.assertEqual(
            constant["spearman_undefined_reason"], "constant_prediction"
        )
        self.assertEqual(constant["comparable_pair_count"], 3)
        self.assertEqual(constant["strict_pair_correct_count"], 0)
        self.assertEqual(
            constant["prediction_tie_comparable_pair_count"], 3
        )
        self.assertEqual(
            constant["strict_comparable_pair_accuracy"], 0.0
        )

    def test_paired_song_bootstrap_is_deterministic_and_paired(self) -> None:
        targets = [8.0, 9.0, 10.0, 8.0, 9.0, 10.0]
        coded = top_code_targets(
            targets,
            [3] * len(targets),
            cap=7,
            selected_course_ids={3},
        )
        panel = build_hidden_tail_panel(
            coded,
            ["a", "b", "c", "d", "e", "f"],
        )
        first = targets
        second = [-value for value in targets]
        result = paired_song_group_bootstrap_difference(
            panel,
            first,
            second,
            replicates=200,
            seed=2027,
        )
        repeated = paired_song_group_bootstrap_difference(
            panel,
            first,
            second,
            replicates=200,
            seed=2027,
        )
        self.assertEqual(result, repeated)
        self.assertAlmostEqual(
            result["spearman_difference"]["point_difference"], 2.0
        )
        self.assertAlmostEqual(
            result["strict_pair_accuracy_difference"][
                "point_difference"
            ],
            1.0,
        )
        self.assertEqual(
            result["spearman_difference"]["valid_replicates"]
            + result["spearman_difference"]["undefined_replicates"],
            200,
        )
        self.assertEqual(
            result["strict_pair_accuracy_difference"]["valid_replicates"]
            + result["strict_pair_accuracy_difference"][
                "undefined_replicates"
            ],
            200,
        )
        self.assertIsNone(result["bootstrap_undefined_reason"])

    def test_bootstrap_reports_empty_panel_without_fabricating_values(
        self,
    ) -> None:
        coded = top_code_targets(
            [6.0, 7.0],
            [3, 3],
            cap=8,
            selected_course_ids={3},
        )
        panel = build_hidden_tail_panel(coded, ["a", "b"])
        result = paired_song_group_bootstrap_difference(
            panel,
            [1.0, 2.0],
            [2.0, 1.0],
            replicates=20,
        )
        self.assertEqual(
            result["bootstrap_undefined_reason"], "empty_hidden_tail_panel"
        )
        self.assertIsNone(
            result["spearman_difference"]["point_difference"]
        )
        self.assertIsNone(
            result["spearman_difference"]["bootstrap_ci_low"]
        )
        self.assertEqual(
            result["spearman_difference"]["valid_replicates"], 0
        )
        self.assertEqual(
            result["spearman_difference"]["undefined_replicates"], 20
        )


if __name__ == "__main__":
    unittest.main()
