"""Deterministic convex linear regression for censored star labels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize


def _matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _vector(
    values: Sequence[float] | Sequence[bool] | np.ndarray,
    name: str,
    *,
    length: int,
    dtype: Any,
) -> np.ndarray:
    vector = np.asarray(values, dtype=dtype)
    if vector.ndim != 1 or len(vector) != length:
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}")
    if np.issubdtype(vector.dtype, np.floating) and not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _loss_and_gradient(
    parameters: np.ndarray,
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    is_left_censored: np.ndarray,
    is_right_censored: np.ndarray,
    censored: bool,
    alpha: float,
    huber_delta: float,
) -> tuple[float, np.ndarray]:
    """Return the convex objective and its deterministic subgradient."""

    coefficients = parameters[:-1]
    intercept = parameters[-1]
    predictions = matrix @ coefficients + intercept
    gradient_prediction = np.zeros_like(predictions)
    losses = np.zeros_like(predictions)

    if censored:
        uncensored = ~(is_left_censored | is_right_censored)
    else:
        uncensored = np.ones(len(targets), dtype=bool)

    residual = predictions[uncensored] - targets[uncensored]
    absolute = np.abs(residual)
    quadratic = absolute <= huber_delta
    uncensored_losses = np.empty_like(residual)
    uncensored_losses[quadratic] = 0.5 * np.square(residual[quadratic])
    uncensored_losses[~quadratic] = huber_delta * (
        absolute[~quadratic] - 0.5 * huber_delta
    )
    uncensored_gradient = np.empty_like(residual)
    uncensored_gradient[quadratic] = residual[quadratic]
    uncensored_gradient[~quadratic] = (
        huber_delta * np.sign(residual[~quadratic])
    )
    losses[uncensored] = uncensored_losses
    gradient_prediction[uncensored] = uncensored_gradient

    if censored:
        right_shortfall = targets[is_right_censored] - predictions[
            is_right_censored
        ]
        active_right = right_shortfall > 0.0
        losses[is_right_censored] = np.maximum(right_shortfall, 0.0)
        right_gradient = np.zeros_like(right_shortfall)
        right_gradient[active_right] = -1.0
        gradient_prediction[is_right_censored] = right_gradient

        left_overshoot = predictions[is_left_censored] - targets[
            is_left_censored
        ]
        active_left = left_overshoot > 0.0
        losses[is_left_censored] = np.maximum(left_overshoot, 0.0)
        left_gradient = np.zeros_like(left_overshoot)
        left_gradient[active_left] = 1.0
        gradient_prediction[is_left_censored] = left_gradient

    sample_count = len(targets)
    objective = float(
        losses.mean() + 0.5 * alpha * np.dot(coefficients, coefficients)
    )
    gradient = np.concatenate(
        (
            matrix.T @ gradient_prediction / sample_count + alpha * coefficients,
            np.asarray([gradient_prediction.mean()]),
        )
    )
    return objective, gradient


@dataclass(frozen=True)
class OptimizationSummary:
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    objective: float
    gradient_infinity_norm: float


class GlobalLinearHuberRegressor:
    """Standardized linear Huber model with optional boundary censoring.

    The estimator follows the small scikit-learn-style ``fit``/``predict``
    contract needed by the research pipeline, while keeping the exact
    censoring masks explicit at fit time.
    """

    def __init__(
        self,
        *,
        alpha: float,
        censored: bool,
        huber_delta: float = 1.0,
        max_iter: int = 5_000,
        tolerance: float = 1e-10,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.alpha = float(alpha)
        self.censored = bool(censored)
        self.huber_delta = float(huber_delta)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def fit(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
        *,
        is_left_censored: Sequence[bool] | np.ndarray,
        is_right_censored: Sequence[bool] | np.ndarray,
    ) -> "GlobalLinearHuberRegressor":
        raw_matrix = _matrix(values, "values")
        target_vector = _vector(
            targets,
            "targets",
            length=len(raw_matrix),
            dtype=np.float64,
        )
        left = _vector(
            is_left_censored,
            "is_left_censored",
            length=len(raw_matrix),
            dtype=bool,
        )
        right = _vector(
            is_right_censored,
            "is_right_censored",
            length=len(raw_matrix),
            dtype=bool,
        )
        if np.any(left & right):
            raise ValueError("an observation cannot be both left- and right-censored")

        self.feature_mean_ = raw_matrix.mean(axis=0)
        scale = raw_matrix.std(axis=0, ddof=0)
        self.feature_scale_ = np.where(scale > 0.0, scale, 1.0)
        matrix = (raw_matrix - self.feature_mean_) / self.feature_scale_

        design = np.column_stack((matrix, np.ones(len(matrix), dtype=np.float64)))
        initial, *_ = np.linalg.lstsq(design, target_vector, rcond=None)

        objective = partial(
            _loss_and_gradient,
            is_left_censored=left,
            is_right_censored=right,
            censored=self.censored,
            alpha=self.alpha,
            huber_delta=self.huber_delta,
        )
        result = minimize(
            objective,
            initial,
            args=(matrix, target_vector),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
                "maxls": 50,
            },
        )
        summary = OptimizationSummary(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            objective=float(result.fun),
            gradient_infinity_norm=float(np.max(np.abs(result.jac))),
        )
        self.optimization_summary_ = summary
        if not summary.success:
            raise RuntimeError(
                "linear censored optimization failed: "
                f"status={summary.status}, message={summary.message}"
            )
        parameters = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(parameters)):
            raise RuntimeError("linear censored optimization returned non-finite values")
        self.coef_ = parameters[:-1]
        self.intercept_ = float(parameters[-1])
        self.n_features_in_ = raw_matrix.shape[1]
        return self

    def predict(
        self, values: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        matrix = _matrix(values, "values")
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {matrix.shape[1]}"
            )
        standardized = (matrix - self.feature_mean_) / self.feature_scale_
        predictions = standardized @ self.coef_ + self.intercept_
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("prediction produced non-finite values")
        return predictions

    def coefficient_record(self, feature_names: Sequence[str]) -> dict[str, Any]:
        """Return standardized and original-unit coefficients for audit."""

        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before coefficients are available")
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} feature names, got {len(feature_names)}"
            )
        raw_coefficients = self.coef_ / self.feature_scale_
        raw_intercept = self.intercept_ - float(
            np.dot(self.feature_mean_, raw_coefficients)
        )
        return {
            "feature_names": list(feature_names),
            "standardized_coefficients": self.coef_.tolist(),
            "raw_coefficients": raw_coefficients.tolist(),
            "standardized_intercept": self.intercept_,
            "raw_intercept": raw_intercept,
            "feature_mean": self.feature_mean_.tolist(),
            "feature_scale": self.feature_scale_.tolist(),
            "optimization": {
                "success": self.optimization_summary_.success,
                "status": self.optimization_summary_.status,
                "message": self.optimization_summary_.message,
                "iterations": self.optimization_summary_.iterations,
                "function_evaluations": (
                    self.optimization_summary_.function_evaluations
                ),
                "objective": self.optimization_summary_.objective,
                "gradient_infinity_norm": (
                    self.optimization_summary_.gradient_infinity_norm
                ),
            },
        }
