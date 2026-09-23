"""Deterministic cumulative-link baselines for ordered difficulty labels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr, ndtri

OrdinalLink = Literal["logit", "probit"]


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


def _targets(
    values: Sequence[float] | np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != length:
        raise ValueError(f"targets must have shape ({length},), got {raw.shape}")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise ValueError("targets must contain numeric ordered labels")
    targets = raw.astype(np.float64)
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets contains non-finite values")
    return targets


def _softplus_inverse(values: np.ndarray) -> np.ndarray:
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("softplus inverse requires positive finite values")
    return values + np.log(-np.expm1(-values))


def _thresholds_from_parameters(
    parameters: np.ndarray,
    *,
    min_threshold_gap: float,
) -> np.ndarray:
    thresholds = np.empty(len(parameters), dtype=np.float64)
    thresholds[0] = parameters[0]
    if len(parameters) > 1:
        gaps = min_threshold_gap + np.logaddexp(0.0, parameters[1:])
        thresholds[1:] = thresholds[0] + np.cumsum(gaps)
    return thresholds


def _log_cdf(values: np.ndarray, link: OrdinalLink) -> np.ndarray:
    if link == "logit":
        return -np.logaddexp(0.0, -values)
    return log_ndtr(values)


def _log_survival(values: np.ndarray, link: OrdinalLink) -> np.ndarray:
    if link == "logit":
        return -np.logaddexp(0.0, values)
    return log_ndtr(-values)


def _log_density(values: np.ndarray, link: OrdinalLink) -> np.ndarray:
    if link == "logit":
        return -np.logaddexp(0.0, -values) - np.logaddexp(0.0, values)
    return -0.5 * np.square(values) - 0.5 * np.log(2.0 * np.pi)


def _log_difference(
    log_larger: np.ndarray,
    log_smaller: np.ndarray,
) -> np.ndarray:
    """Compute ``log(exp(log_larger) - exp(log_smaller))`` stably."""

    delta = log_smaller - log_larger
    return log_larger + np.log(-np.expm1(delta))


def _interval_log_probability(
    lower: np.ndarray,
    upper: np.ndarray,
    link: OrdinalLink,
) -> np.ndarray:
    if np.any(lower >= upper):
        raise RuntimeError("ordinal thresholds are not strictly ordered")

    if link == "logit":
        gap = upper - lower
        return (
            _log_cdf(upper, link)
            + _log_survival(lower, link)
            + np.log(-np.expm1(-gap))
        )

    result = np.empty_like(lower)
    lower_tail = upper <= 0.0
    upper_tail = lower >= 0.0
    central = ~(lower_tail | upper_tail)

    if np.any(lower_tail):
        upper_log_cdf = _log_cdf(upper[lower_tail], link)
        lower_log_cdf = _log_cdf(lower[lower_tail], link)
        result[lower_tail] = _log_difference(
            upper_log_cdf,
            lower_log_cdf,
        )
    if np.any(upper_tail):
        lower_log_survival = _log_survival(lower[upper_tail], link)
        upper_log_survival = _log_survival(upper[upper_tail], link)
        result[upper_tail] = _log_difference(
            lower_log_survival,
            upper_log_survival,
        )
    if np.any(central):
        probability = ndtr(upper[central]) - ndtr(lower[central])
        result[central] = np.log(probability)
    return result


def _density_probability_ratio(
    values: np.ndarray,
    log_probability: np.ndarray,
    link: OrdinalLink,
) -> np.ndarray:
    log_ratio = _log_density(values, link) - log_probability
    maximum_log = np.log(np.finfo(np.float64).max) - 2.0
    return np.exp(np.minimum(log_ratio, maximum_log))


def _loss_and_gradient(
    parameters: np.ndarray,
    matrix: np.ndarray,
    class_indices: np.ndarray,
    *,
    class_count: int,
    link: OrdinalLink,
    alpha: float,
    min_threshold_gap: float,
) -> tuple[float, np.ndarray]:
    feature_count = matrix.shape[1]
    coefficients = parameters[:feature_count]
    threshold_parameters = parameters[feature_count:]
    thresholds = _thresholds_from_parameters(
        threshold_parameters,
        min_threshold_gap=min_threshold_gap,
    )
    latent = matrix @ coefficients

    sample_count = len(matrix)
    log_probabilities = np.empty(sample_count, dtype=np.float64)
    latent_gradient = np.empty(sample_count, dtype=np.float64)
    threshold_gradient = np.zeros(class_count - 1, dtype=np.float64)

    first = class_indices == 0
    if np.any(first):
        upper = thresholds[0] - latent[first]
        log_probability = _log_cdf(upper, link)
        ratio = _density_probability_ratio(upper, log_probability, link)
        log_probabilities[first] = log_probability
        latent_gradient[first] = ratio
        threshold_gradient[0] -= float(ratio.sum())

    last = class_indices == class_count - 1
    if np.any(last):
        lower = thresholds[-1] - latent[last]
        log_probability = _log_survival(lower, link)
        ratio = _density_probability_ratio(lower, log_probability, link)
        log_probabilities[last] = log_probability
        latent_gradient[last] = -ratio
        threshold_gradient[-1] += float(ratio.sum())

    middle = ~(first | last)
    if np.any(middle):
        middle_indices = class_indices[middle]
        lower = thresholds[middle_indices - 1] - latent[middle]
        upper = thresholds[middle_indices] - latent[middle]
        log_probability = _interval_log_probability(lower, upper, link)
        lower_ratio = _density_probability_ratio(
            lower,
            log_probability,
            link,
        )
        upper_ratio = _density_probability_ratio(
            upper,
            log_probability,
            link,
        )
        log_probabilities[middle] = log_probability
        latent_gradient[middle] = upper_ratio - lower_ratio
        np.add.at(threshold_gradient, middle_indices - 1, lower_ratio)
        np.add.at(threshold_gradient, middle_indices, -upper_ratio)

    objective = float(
        -log_probabilities.mean()
        + 0.5 * alpha * np.dot(coefficients, coefficients)
    )
    coefficient_gradient = (
        matrix.T @ latent_gradient / sample_count + alpha * coefficients
    )
    threshold_gradient /= sample_count

    parameter_gradient = np.empty_like(threshold_parameters)
    parameter_gradient[0] = threshold_gradient.sum()
    if len(threshold_parameters) > 1:
        suffix_gradient = np.cumsum(threshold_gradient[:0:-1])[::-1]
        gap_derivative = expit(threshold_parameters[1:])
        parameter_gradient[1:] = suffix_gradient * gap_derivative

    gradient = np.concatenate((coefficient_gradient, parameter_gradient))
    return objective, gradient


@dataclass(frozen=True)
class OrdinalOptimizationSummary:
    """JSON-compatible audit fields from deterministic SciPy optimization."""

    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    objective: float
    gradient_infinity_norm: float

    def as_record(self) -> dict[str, bool | int | float | str]:
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "objective": self.objective,
            "gradient_infinity_norm": self.gradient_infinity_norm,
        }


class CumulativeLinkOrdinalRegressor:
    """Standardized linear cumulative-link model with learned cut-points.

    ``P(Y <= k | x) = F(threshold[k] - x @ coefficients)`` where ``F`` is
    either the logistic or standard-normal CDF. Cut-points use positive
    softplus gaps and are therefore strictly ordered throughout optimization.
    No separate standardized intercept is learned because it is not
    identifiable jointly with freely located cut-points.
    """

    def __init__(
        self,
        *,
        alpha: float,
        link: OrdinalLink = "logit",
        max_iter: int = 5_000,
        tolerance: float = 1e-9,
        min_threshold_gap: float = 1e-6,
    ) -> None:
        if not isinstance(alpha, (int, float)) or not np.isfinite(alpha):
            raise ValueError("alpha must be a finite positive number")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if link not in ("logit", "probit"):
            raise ValueError("link must be either 'logit' or 'probit'")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise ValueError("max_iter must be a positive integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if (
            not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("tolerance must be a finite positive number")
        if (
            not isinstance(min_threshold_gap, (int, float))
            or not np.isfinite(min_threshold_gap)
            or min_threshold_gap <= 0.0
        ):
            raise ValueError(
                "min_threshold_gap must be a finite positive number"
            )
        self.alpha = float(alpha)
        self.link: OrdinalLink = link
        self.max_iter = max_iter
        self.tolerance = float(tolerance)
        self.min_threshold_gap = float(min_threshold_gap)

    def _initial_threshold_parameters(
        self,
        class_indices: np.ndarray,
        class_count: int,
    ) -> np.ndarray:
        counts = np.bincount(class_indices, minlength=class_count)
        cumulative = np.cumsum(counts)[:-1] / len(class_indices)
        if self.link == "logit":
            thresholds = np.log(cumulative) - np.log1p(-cumulative)
        else:
            thresholds = ndtri(cumulative)

        minimum_initial_gap = self.min_threshold_gap + 1e-2
        adjusted = thresholds.copy()
        for index in range(1, len(adjusted)):
            adjusted[index] = max(
                adjusted[index],
                adjusted[index - 1] + minimum_initial_gap,
            )
        parameters = np.empty_like(adjusted)
        parameters[0] = adjusted[0]
        if len(parameters) > 1:
            positive_gap = np.diff(adjusted) - self.min_threshold_gap
            parameters[1:] = _softplus_inverse(positive_gap)
        return parameters

    def fit(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[float] | np.ndarray,
    ) -> "CumulativeLinkOrdinalRegressor":
        raw_matrix = _matrix(values, "values")
        target_vector = _targets(targets, length=len(raw_matrix))
        classes, class_indices = np.unique(
            target_vector,
            return_inverse=True,
        )
        if len(classes) < 2:
            raise ValueError("targets must contain at least two ordered classes")

        feature_mean = raw_matrix.mean(axis=0)
        scale = raw_matrix.std(axis=0, ddof=0)
        feature_scale = np.where(scale > 0.0, scale, 1.0)
        matrix = (raw_matrix - feature_mean) / feature_scale

        initial = np.concatenate(
            (
                np.zeros(raw_matrix.shape[1], dtype=np.float64),
                self._initial_threshold_parameters(class_indices, len(classes)),
            )
        )
        objective = partial(
            _loss_and_gradient,
            class_count=len(classes),
            link=self.link,
            alpha=self.alpha,
            min_threshold_gap=self.min_threshold_gap,
        )
        result = minimize(
            objective,
            initial,
            args=(matrix, class_indices),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
                "maxls": 50,
            },
        )

        parameters = np.asarray(result.x, dtype=np.float64)
        jacobian = np.asarray(result.jac, dtype=np.float64)
        if (
            not np.all(np.isfinite(parameters))
            or not np.isfinite(result.fun)
            or not np.all(np.isfinite(jacobian))
        ):
            raise RuntimeError(
                "ordinal optimization returned non-finite values"
            )
        summary = OrdinalOptimizationSummary(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            objective=float(result.fun),
            gradient_infinity_norm=float(np.max(np.abs(jacobian))),
        )
        if not summary.success:
            raise RuntimeError(
                "ordinal optimization failed: "
                f"status={summary.status}, message={summary.message}"
            )

        feature_count = raw_matrix.shape[1]
        thresholds = _thresholds_from_parameters(
            parameters[feature_count:],
            min_threshold_gap=self.min_threshold_gap,
        )
        if not np.all(np.isfinite(thresholds)):
            raise RuntimeError(
                "ordinal optimization returned non-finite thresholds"
            )
        if np.any(np.diff(thresholds) <= 0.0):
            raise RuntimeError(
                "ordinal optimization returned unordered thresholds"
            )

        self.classes_ = classes
        self.coef_ = parameters[:feature_count]
        self.thresholds_ = thresholds
        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        self.n_features_in_ = feature_count
        self.optimization_summary_ = summary
        return self

    def _validated_prediction_matrix(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        matrix = _matrix(values, "values")
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before prediction")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {matrix.shape[1]}"
            )
        return (matrix - self.feature_mean_) / self.feature_scale_

    def predict_latent(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Return the unbounded latent ordinal score ``x @ coefficients``."""

        matrix = self._validated_prediction_matrix(values)
        latent = matrix @ self.coef_
        if not np.all(np.isfinite(latent)):
            raise RuntimeError("latent prediction produced non-finite values")
        return latent

    def predict_proba(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Return one probability column per sorted value in ``classes_``."""

        latent = self.predict_latent(values)
        offsets = self.thresholds_[None, :] - latent[:, None]
        cumulative = (
            expit(offsets) if self.link == "logit" else ndtr(offsets)
        )
        probabilities = np.column_stack(
            (
                cumulative[:, 0],
                np.diff(cumulative, axis=1),
                1.0 - cumulative[:, -1],
            )
        )
        probabilities = np.maximum(probabilities, 0.0)
        totals = probabilities.sum(axis=1)
        if (
            not np.all(np.isfinite(probabilities))
            or not np.all(np.isfinite(totals))
            or np.any(totals <= 0.0)
        ):
            raise RuntimeError(
                "ordinal probability prediction produced invalid values"
            )
        probabilities /= totals[:, None]
        return probabilities

    def predict_category(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Return the maximum-probability observed category."""

        probabilities = self.predict_proba(values)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Alias for :meth:`predict_category`."""

        return self.predict_category(values)

    def predict_expected_label(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Return the posterior mean on the numeric observed-label scale."""

        expected = self.predict_proba(values) @ self.classes_
        if not np.all(np.isfinite(expected)):
            raise RuntimeError(
                "expected-label prediction produced non-finite values"
            )
        return expected

    def coefficient_record(
        self,
        feature_names: Sequence[str],
    ) -> dict[str, Any]:
        """Return a JSON-serializable model and optimization audit record."""

        if not hasattr(self, "coef_"):
            raise RuntimeError(
                "fit must be called before coefficients are available"
            )
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} feature names, "
                f"got {len(feature_names)}"
            )
        if any(not isinstance(name, str) or not name for name in feature_names):
            raise ValueError("feature names must be non-empty strings")

        raw_coefficients = self.coef_ / self.feature_scale_
        raw_intercept = -float(
            np.dot(self.feature_mean_, raw_coefficients)
        )
        return {
            "model": "cumulative_link_ordinal",
            "link": self.link,
            "alpha": self.alpha,
            "min_threshold_gap": self.min_threshold_gap,
            "feature_names": list(feature_names),
            "classes": self.classes_.tolist(),
            "thresholds": self.thresholds_.tolist(),
            "standardized_coefficients": self.coef_.tolist(),
            "raw_coefficients": raw_coefficients.tolist(),
            "standardized_intercept": 0.0,
            "raw_intercept": raw_intercept,
            "feature_mean": self.feature_mean_.tolist(),
            "feature_scale": self.feature_scale_.tolist(),
            "optimization": self.optimization_summary_.as_record(),
        }
