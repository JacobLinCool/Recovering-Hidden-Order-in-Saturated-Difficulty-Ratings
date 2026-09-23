import unittest

import numpy as np

from TaikoChartEstimator.research.ordinal_reduction import (
    OrdinalLogitReduction,
)


class OrdinalLogitReductionTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2027)
        self.values = rng.normal(size=(240, 3))
        latent = (
            1.6 * self.values[:, 0]
            - 0.8 * self.values[:, 1]
            + 0.2 * self.values[:, 2]
        )
        self.targets = np.digitize(latent, [-1.0, -0.2, 0.6]) + 1

    def test_probabilities_are_valid_and_exceedance_is_monotone(self) -> None:
        model = OrdinalLogitReduction(C=1.0).fit(
            self.values,
            self.targets,
        )
        exceedance = model.predict_exceedance_proba(self.values[:20])
        probabilities = model.predict_proba(self.values[:20])
        self.assertTrue(np.all(np.diff(exceedance, axis=1) <= 1e-12))
        self.assertTrue(np.all(probabilities >= 0.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        expected = model.predict_expected_label(self.values[:20])
        self.assertTrue(np.all(expected >= self.targets.min()))
        self.assertTrue(np.all(expected <= self.targets.max()))

    def test_fit_is_deterministic(self) -> None:
        first = OrdinalLogitReduction(C=0.5).fit(
            self.values,
            self.targets,
        )
        second = OrdinalLogitReduction(C=0.5).fit(
            self.values,
            self.targets,
        )
        np.testing.assert_allclose(
            first.predict_expected_label(self.values),
            second.predict_expected_label(self.values),
        )
        self.assertEqual(
            first.coefficient_record(["a", "b", "c"]),
            second.coefficient_record(["a", "b", "c"]),
        )

    def test_rejects_noncontiguous_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            OrdinalLogitReduction(C=1.0).fit(
                self.values[:6],
                np.asarray([1, 1, 3, 3, 4, 4]),
            )

    def test_rejects_invalid_inputs_and_unfitted_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            OrdinalLogitReduction(C=0.0)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            OrdinalLogitReduction(C=1.0).predict([[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            OrdinalLogitReduction(C=1.0).fit(
                [[1.0], [float("nan")]],
                [1, 2],
            )


if __name__ == "__main__":
    unittest.main()
