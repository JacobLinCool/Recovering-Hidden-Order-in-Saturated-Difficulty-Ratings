from __future__ import annotations

import unittest

import numpy as np

from TaikoChartEstimator.research.censored_linear import (
    GlobalLinearHuberRegressor,
)


class ResearchCensoredLinearTest(unittest.TestCase):
    def test_censored_fit_recovers_signal_beyond_observed_ceiling(self) -> None:
        features = np.arange(6, dtype=np.float64).reshape(-1, 1)
        targets = np.asarray([2.0, 3.0, 4.0, 5.0, 5.0, 5.0])
        right = targets == 5.0
        left = np.zeros(len(targets), dtype=bool)

        ordinary = GlobalLinearHuberRegressor(
            alpha=1e-6,
            censored=False,
            tolerance=1e-9,
        ).fit(
            features,
            targets,
            is_left_censored=left,
            is_right_censored=right,
        )
        censored = GlobalLinearHuberRegressor(
            alpha=1e-6,
            censored=True,
            tolerance=1e-9,
        ).fit(
            features,
            targets,
            is_left_censored=left,
            is_right_censored=right,
        )

        ordinary_tail = float(ordinary.predict([[5.0]])[0])
        censored_tail = float(censored.predict([[5.0]])[0])
        self.assertGreater(censored_tail, ordinary_tail)
        self.assertGreaterEqual(censored_tail, 5.0)

    def test_fit_is_deterministic_and_coefficients_are_auditable(self) -> None:
        features = np.asarray(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.0]]
        )
        targets = np.asarray([1.0, 2.0, 3.0, 3.0])
        left = np.asarray([True, False, False, False])
        right = np.asarray([False, False, True, True])

        def fitted() -> GlobalLinearHuberRegressor:
            return GlobalLinearHuberRegressor(
                alpha=0.01,
                censored=True,
            ).fit(
                features,
                targets,
                is_left_censored=left,
                is_right_censored=right,
            )

        first = fitted()
        second = fitted()
        np.testing.assert_allclose(
            first.predict(features),
            second.predict(features),
            rtol=0.0,
            atol=1e-12,
        )
        record = first.coefficient_record(["a", "b"])
        self.assertEqual(record["feature_names"], ["a", "b"])
        self.assertTrue(record["optimization"]["success"])


if __name__ == "__main__":
    unittest.main()
