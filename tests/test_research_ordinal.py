from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from TaikoChartEstimator.research.ordinal import (
    CumulativeLinkOrdinalRegressor,
)


def _ordered_data() -> tuple[np.ndarray, np.ndarray]:
    first = np.linspace(-3.0, 3.0, 180)
    second = np.sin(first * 1.7)
    features = np.column_stack((first, second))
    latent = 1.4 * first - 0.15 * second
    targets = np.digitize(latent, [-1.8, -0.45, 0.7, 2.0]) + 1
    return features, targets


class ResearchOrdinalTest(unittest.TestCase):
    def test_both_links_learn_ordered_thresholds_and_monotone_predictions(
        self,
    ) -> None:
        features, targets = _ordered_data()
        probe = np.column_stack(
            (np.linspace(-3.5, 3.5, 41), np.zeros(41))
        )

        for link in ("logit", "probit"):
            with self.subTest(link=link):
                model = CumulativeLinkOrdinalRegressor(
                    alpha=0.02,
                    link=link,
                ).fit(features, targets)

                self.assertTrue(np.all(np.diff(model.thresholds_) > 0.0))
                self.assertEqual(len(model.thresholds_), 4)
                probabilities = model.predict_proba(probe)
                np.testing.assert_allclose(
                    probabilities.sum(axis=1),
                    np.ones(len(probe)),
                    rtol=0.0,
                    atol=1e-14,
                )
                self.assertTrue(np.all(probabilities >= 0.0))

                latent = model.predict_latent(probe)
                expected = model.predict_expected_label(probe)
                self.assertTrue(np.all(np.diff(latent) > 0.0))
                self.assertTrue(np.all(np.diff(expected) > 0.0))
                self.assertEqual(model.predict(probe).shape, (len(probe),))
                accuracy = np.mean(model.predict(features) == targets)
                self.assertGreater(accuracy, 0.8)

    def test_fit_is_deterministic_and_record_is_json_serializable(self) -> None:
        features, targets = _ordered_data()

        def fitted() -> CumulativeLinkOrdinalRegressor:
            return CumulativeLinkOrdinalRegressor(
                alpha=0.05,
                link="logit",
                tolerance=1e-10,
            ).fit(features, targets)

        first = fitted()
        second = fitted()
        np.testing.assert_allclose(first.coef_, second.coef_, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            first.thresholds_,
            second.thresholds_,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            first.predict_proba(features),
            second.predict_proba(features),
            rtol=0.0,
            atol=0.0,
        )

        record = first.coefficient_record(["linear", "periodic"])
        self.assertEqual(record["classes"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(record["feature_names"], ["linear", "periodic"])
        self.assertTrue(record["optimization"]["success"])
        json.dumps(record, allow_nan=False)

    def test_constant_feature_is_supported_without_non_finite_values(self) -> None:
        features, targets = _ordered_data()
        features = np.column_stack((features, np.ones(len(features))))
        model = CumulativeLinkOrdinalRegressor(
            alpha=0.1,
            link="probit",
        ).fit(features, targets)

        self.assertEqual(model.feature_scale_[-1], 1.0)
        self.assertTrue(np.all(np.isfinite(model.predict_latent(features))))
        self.assertTrue(
            np.all(np.isfinite(model.predict_expected_label(features)))
        )

    def test_invalid_hyperparameters_and_fit_inputs_are_rejected(self) -> None:
        for kwargs in (
            {"alpha": 0.0},
            {"alpha": float("nan")},
            {"alpha": 0.1, "link": "cauchit"},
            {"alpha": 0.1, "max_iter": 0},
            {"alpha": 0.1, "max_iter": 1.5},
            {"alpha": 0.1, "tolerance": 0.0},
            {"alpha": 0.1, "min_threshold_gap": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CumulativeLinkOrdinalRegressor(**kwargs)

        model = CumulativeLinkOrdinalRegressor(alpha=0.1)
        with self.assertRaisesRegex(ValueError, "non-empty matrix"):
            model.fit(np.empty((0, 2)), np.empty(0))
        with self.assertRaisesRegex(ValueError, "shape"):
            model.fit([[0.0], [1.0]], [1.0])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.fit([[0.0], [np.inf]], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "numeric"):
            model.fit([[0.0], [1.0]], ["low", "high"])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.fit([[0.0], [1.0]], [1.0, np.nan])
        with self.assertRaisesRegex(ValueError, "at least two"):
            model.fit([[0.0], [1.0]], [1.0, 1.0])

    def test_prediction_and_record_validation(self) -> None:
        unfitted = CumulativeLinkOrdinalRegressor(alpha=0.1)
        with self.assertRaisesRegex(RuntimeError, "fit must be called"):
            unfitted.predict([[0.0]])
        with self.assertRaisesRegex(RuntimeError, "fit must be called"):
            unfitted.coefficient_record(["x"])

        features, targets = _ordered_data()
        fitted = CumulativeLinkOrdinalRegressor(alpha=0.1).fit(
            features,
            targets,
        )
        with self.assertRaisesRegex(ValueError, "expected 2 features"):
            fitted.predict([[0.0]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fitted.predict([[0.0, np.nan]])
        with self.assertRaisesRegex(ValueError, "feature names"):
            fitted.coefficient_record(["x"])
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            fitted.coefficient_record(["x", ""])

    def test_optimizer_failure_and_non_finite_result_are_rejected(self) -> None:
        features, targets = _ordered_data()
        initial_length = features.shape[1] + len(np.unique(targets)) - 1

        failed = SimpleNamespace(
            x=np.zeros(initial_length),
            jac=np.ones(initial_length),
            fun=1.0,
            success=False,
            status=1,
            message="iteration limit",
            nit=1,
            nfev=2,
        )
        with patch(
            "TaikoChartEstimator.research.ordinal.minimize",
            return_value=failed,
        ):
            with self.assertRaisesRegex(RuntimeError, "optimization failed"):
                CumulativeLinkOrdinalRegressor(alpha=0.1).fit(
                    features,
                    targets,
                )

        non_finite = SimpleNamespace(
            x=np.full(initial_length, np.nan),
            jac=np.ones(initial_length),
            fun=np.nan,
            success=True,
            status=0,
            message="converged",
            nit=1,
            nfev=2,
        )
        with patch(
            "TaikoChartEstimator.research.ordinal.minimize",
            return_value=non_finite,
        ):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                CumulativeLinkOrdinalRegressor(alpha=0.1).fit(
                    features,
                    targets,
                )


if __name__ == "__main__":
    unittest.main()
