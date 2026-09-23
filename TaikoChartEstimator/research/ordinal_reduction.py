"""Deterministic threshold-reduction baseline for ordinal ratings."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def _matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _ordinal_targets(
    values: Sequence[int] | np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    targets = np.asarray(values)
    if targets.ndim != 1 or len(targets) != length:
        raise ValueError(
            f"targets must have shape ({length},), got {targets.shape}"
        )
    numeric = targets.astype(np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
        raise ValueError("targets must contain finite integer labels")
    integer = numeric.astype(np.int64)
    classes = np.unique(integer)
    if len(classes) < 2:
        raise ValueError("ordinal fitting requires at least two target classes")
    expected = np.arange(classes[0], classes[-1] + 1, dtype=np.int64)
    if not np.array_equal(classes, expected):
        raise ValueError(
            "ordinal target classes must be contiguous; "
            f"observed={classes.tolist()}"
        )
    return integer


class OrdinalLogitReduction:
    """Fit one logistic exceedance classifier per ordinal boundary.

    For ordered classes ``c_1 < ... < c_K``, boundary ``k`` estimates
    ``P(Y > c_k | X)``. At prediction time the exceedance probabilities are
    monotonized with a cumulative minimum before conversion to a categorical
    distribution. The continuous expected label is the baseline's ordinal
    content score.
    """

    def __init__(
        self,
        *,
        C: float,
        max_iter: int = 5_000,
        tolerance: float = 1e-10,
        random_state: int = 2027,
    ) -> None:
        if C <= 0.0:
            raise ValueError("C must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state)

    def fit(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[int] | np.ndarray,
    ) -> "OrdinalLogitReduction":
        raw_matrix = _matrix(values, "values")
        target_vector = _ordinal_targets(targets, length=len(raw_matrix))
        self.classes_ = np.unique(target_vector)
        self.thresholds_ = self.classes_[:-1].copy()
        self.feature_mean_ = raw_matrix.mean(axis=0)
        scale = raw_matrix.std(axis=0, ddof=0)
        self.feature_scale_ = np.where(scale > 0.0, scale, 1.0)
        matrix = (raw_matrix - self.feature_mean_) / self.feature_scale_

        models: list[LogisticRegression] = []
        iterations: list[int] = []
        for threshold in self.thresholds_:
            binary = target_vector > threshold
            if np.unique(binary).size != 2:
                raise ValueError(
                    f"ordinal boundary {int(threshold)} lacks both classes"
                )
            model = LogisticRegression(
                C=self.C,
                penalty="l2",
                solver="lbfgs",
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tolerance,
                random_state=self.random_state,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(matrix, binary.astype(np.int64))
            convergence = [
                warning
                for warning in caught
                if issubclass(warning.category, ConvergenceWarning)
            ]
            if convergence or int(model.n_iter_[0]) >= self.max_iter:
                raise RuntimeError(
                    f"ordinal boundary {int(threshold)} failed to converge"
                )
            if not np.all(np.isfinite(model.coef_)) or not np.all(
                np.isfinite(model.intercept_)
            ):
                raise RuntimeError(
                    f"ordinal boundary {int(threshold)} returned non-finite parameters"
                )
            models.append(model)
            iterations.append(int(model.n_iter_[0]))

        self.models_ = tuple(models)
        self.iterations_ = tuple(iterations)
        self.n_features_in_ = raw_matrix.shape[1]
        return self

    def _standardize(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        matrix = _matrix(values, "values")
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before prediction")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {matrix.shape[1]}"
            )
        return (matrix - self.feature_mean_) / self.feature_scale_

    def predict_exceedance_proba(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        matrix = self._standardize(values)
        raw = np.column_stack(
            [model.predict_proba(matrix)[:, 1] for model in self.models_]
        )
        probabilities = np.minimum.accumulate(raw, axis=1)
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise RuntimeError("invalid ordinal exceedance probabilities")
        return probabilities

    def predict_proba(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        exceedance = self.predict_exceedance_proba(values)
        probabilities = np.column_stack(
            (
                1.0 - exceedance[:, 0],
                exceedance[:, :-1] - exceedance[:, 1:],
                exceedance[:, -1],
            )
        )
        if np.any(probabilities < -1e-12):
            raise RuntimeError("ordinal class probabilities are negative")
        probabilities = np.maximum(probabilities, 0.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
            raise RuntimeError("ordinal class probabilities cannot be normalized")
        return probabilities / row_sums

    def predict_expected_label(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(values)
        expected = probabilities @ self.classes_.astype(np.float64)
        if not np.all(np.isfinite(expected)):
            raise RuntimeError("ordinal expected labels are non-finite")
        return expected

    def predict_latent(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        return self.predict_expected_label(values)

    def predict(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return self.classes_[probabilities.argmax(axis=1)]

    def coefficient_record(self, feature_names: Sequence[str]) -> dict[str, Any]:
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before coefficients are available")
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} feature names, "
                f"got {len(feature_names)}"
            )
        return {
            "model": "ordinal_logit_reduction",
            "feature_names": list(feature_names),
            "classes": self.classes_.tolist(),
            "thresholds": self.thresholds_.tolist(),
            "C": self.C,
            "feature_mean": self.feature_mean_.tolist(),
            "feature_scale": self.feature_scale_.tolist(),
            "standardized_coefficients_by_threshold": [
                model.coef_[0].tolist() for model in self.models_
            ],
            "standardized_intercepts_by_threshold": [
                float(model.intercept_[0]) for model in self.models_
            ],
            "iterations_by_threshold": list(self.iterations_),
        }
