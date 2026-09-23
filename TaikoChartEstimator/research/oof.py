"""Deterministic out-of-fold and hidden-tail evaluation utilities.

The functions in this module keep songs, rather than individual charts, as
the resampling and partitioning unit.  They are intentionally independent of
the model runner so the same folds and hidden-tail metrics can be shared by
linear, ordinal, and neural conditions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Collection, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass(frozen=True)
class SongGroupFold:
    """One outer test fold with an inner validation split."""

    outer_fold: int
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class TopCodedTargets:
    """Original and observed targets under a course-selective top code.

    ``is_top_coded`` marks selected-course observations at or above the cap.
    ``is_hidden_tail`` is the strict subset whose original target is above the
    cap and is therefore changed by top coding.
    """

    original_targets: tuple[float, ...]
    observed_targets: tuple[float, ...]
    course_ids: tuple[int, ...]
    is_top_coded: tuple[bool, ...]
    is_hidden_tail: tuple[bool, ...]
    cap: float
    selected_course_ids: tuple[int, ...]


@dataclass(frozen=True)
class HiddenTailPanel:
    """Complete observed top category under an artificial cap.

    The panel includes rows whose original target equals the cap as well as
    rows whose original target was strictly higher. This is the identifiable
    evaluation analogue of a real top-coded category: the learner observes the
    same maximum label for every row, while held-out original labels reveal
    ordering across the entire category.
    """

    source_size: int
    source_indices: tuple[int, ...]
    song_groups: tuple[str, ...]
    original_targets: tuple[float, ...]
    observed_targets: tuple[float, ...]
    is_strictly_hidden: tuple[bool, ...]


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return converted


def _group_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    groups: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        groups.append(value)
    return tuple(groups)


def _float_values(
    values: Sequence[float] | np.ndarray,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {raw.shape}")
    if not allow_empty and len(raw) == 0:
        raise ValueError(f"{name} must not be empty")
    converted_values = [
        _finite_real(value, f"{name}[{index}]")
        for index, value in enumerate(raw.tolist())
    ]
    return np.asarray(converted_values, dtype=np.float64)


def _course_values(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {raw.shape}")
    if len(raw) == 0:
        raise ValueError(f"{name} must not be empty")
    converted = [
        _integer(value, f"{name}[{index}]")
        for index, value in enumerate(raw.tolist())
    ]
    return np.asarray(converted, dtype=np.int64)


def _stable_digest(group: str, *, seed: int, namespace: str) -> bytes:
    payload = f"{namespace}\0{seed}\0{group}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def build_song_group_folds(
    song_groups: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 2027,
    validation_fraction: float = 0.1,
) -> tuple[SongGroupFold, ...]:
    """Build balanced deterministic outer folds and inner validation splits.

    Repeated song-group values are deduplicated before assignment.  A stable
    hash ordering followed by round-robin assignment keeps outer fold sizes
    within one group of each other.  For every outer fold, validation groups
    are selected only from that fold's outer-training groups using a separate
    fold-specific hash namespace.
    """

    groups = _group_ids(song_groups, "song_groups")
    if not groups:
        raise ValueError("song_groups must not be empty")
    fold_count = _integer(n_folds, "n_folds")
    split_seed = _integer(seed, "seed")
    if fold_count < 2:
        raise ValueError("n_folds must be at least 2")
    fraction = _finite_real(validation_fraction, "validation_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    unique_groups = tuple(sorted(set(groups)))
    if len(unique_groups) < fold_count:
        raise ValueError(
            f"n_folds={fold_count} exceeds {len(unique_groups)} unique groups"
        )
    outer_order = sorted(
        unique_groups,
        key=lambda group: (
            _stable_digest(group, seed=split_seed, namespace="outer"),
            group,
        ),
    )
    outer_by_group = {
        group: index % fold_count for index, group in enumerate(outer_order)
    }

    folds: list[SongGroupFold] = []
    for outer_fold in range(fold_count):
        test = tuple(
            sorted(
                group
                for group in unique_groups
                if outer_by_group[group] == outer_fold
            )
        )
        outer_training = [
            group
            for group in unique_groups
            if outer_by_group[group] != outer_fold
        ]
        if len(outer_training) < 2:
            raise ValueError(
                "each outer fold needs at least two non-test groups so train "
                "and validation are both non-empty"
            )
        validation_count = min(
            len(outer_training) - 1,
            max(1, math.ceil(len(outer_training) * fraction)),
        )
        inner_order = sorted(
            outer_training,
            key=lambda group: (
                _stable_digest(
                    group,
                    seed=split_seed,
                    namespace=f"inner-validation:{outer_fold}",
                ),
                group,
            ),
        )
        validation_set = set(inner_order[:validation_count])
        validation = tuple(sorted(validation_set))
        train = tuple(
            sorted(group for group in outer_training if group not in validation_set)
        )
        folds.append(
            SongGroupFold(
                outer_fold=outer_fold,
                train_groups=train,
                validation_groups=validation,
                test_groups=test,
            )
        )

    test_memberships = [
        group for fold in folds for group in fold.test_groups
    ]
    if len(test_memberships) != len(set(test_memberships)):
        raise RuntimeError("a song group was assigned to multiple outer test folds")
    if set(test_memberships) != set(unique_groups):
        raise RuntimeError("not every song group received an outer test fold")
    return tuple(folds)


def top_code_targets(
    targets: Sequence[float] | np.ndarray,
    course_ids: Sequence[int] | np.ndarray,
    *,
    cap: float,
    selected_course_ids: Collection[int],
) -> TopCodedTargets:
    """Top-code selected courses while preserving their original targets."""

    original = _float_values(targets, "targets")
    courses = _course_values(course_ids, "course_ids")
    if len(original) != len(courses):
        raise ValueError(
            "targets and course_ids must have the same length: "
            f"{len(original)} != {len(courses)}"
        )
    cap_value = _finite_real(cap, "cap")
    selected = tuple(
        sorted(
            {
                _integer(course_id, "selected_course_ids entry")
                for course_id in selected_course_ids
            }
        )
    )
    if not selected:
        raise ValueError("selected_course_ids must not be empty")

    selected_mask = np.isin(courses, np.asarray(selected, dtype=np.int64))
    observed = original.copy()
    observed[selected_mask] = np.minimum(observed[selected_mask], cap_value)
    at_or_above_cap = selected_mask & (original >= cap_value)
    hidden_tail = selected_mask & (original > cap_value)
    return TopCodedTargets(
        original_targets=tuple(float(value) for value in original),
        observed_targets=tuple(float(value) for value in observed),
        course_ids=tuple(int(value) for value in courses),
        is_top_coded=tuple(bool(value) for value in at_or_above_cap),
        is_hidden_tail=tuple(bool(value) for value in hidden_tail),
        cap=cap_value,
        selected_course_ids=selected,
    )


def build_hidden_tail_panel(
    top_coded: TopCodedTargets,
    song_groups: Sequence[str],
) -> HiddenTailPanel:
    """Select the complete synthetic top category with song provenance."""

    groups = _group_ids(song_groups, "song_groups")
    source_size = len(top_coded.original_targets)
    fields = {
        "observed_targets": len(top_coded.observed_targets),
        "course_ids": len(top_coded.course_ids),
        "is_top_coded": len(top_coded.is_top_coded),
        "is_hidden_tail": len(top_coded.is_hidden_tail),
        "song_groups": len(groups),
    }
    if any(length != source_size for length in fields.values()):
        raise ValueError(
            "top-coded fields and song_groups have inconsistent lengths: "
            f"original_targets={source_size}, {fields}"
        )
    indices = tuple(
        index
        for index, is_top_coded in enumerate(top_coded.is_top_coded)
        if is_top_coded
    )
    return HiddenTailPanel(
        source_size=source_size,
        source_indices=indices,
        song_groups=tuple(groups[index] for index in indices),
        original_targets=tuple(
            top_coded.original_targets[index] for index in indices
        ),
        observed_targets=tuple(
            top_coded.observed_targets[index] for index in indices
        ),
        is_strictly_hidden=tuple(
            top_coded.is_hidden_tail[index] for index in indices
        ),
    )


def _strict_pair_counts(
    targets: np.ndarray, predictions: np.ndarray
) -> tuple[int, int, int]:
    """Count target-comparable, correctly ordered, and prediction-tied pairs."""

    if len(targets) < 2:
        return 0, 0, 0
    order = np.argsort(targets, kind="mergesort")
    sorted_targets = targets[order]
    sorted_predictions = predictions[order]
    _, prediction_ranks = np.unique(
        sorted_predictions, return_inverse=True
    )

    tree = [0] * (int(prediction_ranks.max()) + 2)

    def add(rank: int) -> None:
        tree_index = rank + 1
        while tree_index < len(tree):
            tree[tree_index] += 1
            tree_index += tree_index & -tree_index

    def prefix_count(exclusive_rank: int) -> int:
        total = 0
        tree_index = exclusive_rank
        while tree_index > 0:
            total += tree[tree_index]
            tree_index -= tree_index & -tree_index
        return total

    comparable = 0
    correct = 0
    prediction_ties = 0
    previous_count = 0
    start = 0
    while start < len(sorted_targets):
        stop = start + 1
        while (
            stop < len(sorted_targets)
            and sorted_targets[stop] == sorted_targets[start]
        ):
            stop += 1
        for index in range(start, stop):
            rank = int(prediction_ranks[index])
            less = prefix_count(rank)
            less_or_equal = prefix_count(rank + 1)
            comparable += previous_count
            correct += less
            prediction_ties += less_or_equal - less
        for index in range(start, stop):
            add(int(prediction_ranks[index]))
        previous_count += stop - start
        start = stop
    return comparable, correct, prediction_ties


def strict_pair_accuracy(
    targets: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
) -> float | None:
    """Return accuracy over pairs whose target labels are strictly unequal."""

    target_array = _float_values(targets, "targets")
    prediction_array = _float_values(predictions, "predictions")
    if len(target_array) != len(prediction_array):
        raise ValueError("targets and predictions must have the same length")
    comparable, correct, _ = _strict_pair_counts(target_array, prediction_array)
    return float(correct / comparable) if comparable else None


def _rank_metrics_arrays(
    targets: np.ndarray,
    predictions: np.ndarray,
    song_groups: Sequence[str],
) -> dict[str, float | int | str | None]:
    count = len(targets)
    distinct_targets = int(np.unique(targets).size)
    distinct_predictions = int(np.unique(predictions).size)
    spearman_reason: str | None = None
    if count < 2:
        spearman = None
        spearman_reason = "fewer_than_two_rows"
    elif distinct_targets < 2:
        spearman = None
        spearman_reason = "constant_target"
    elif distinct_predictions < 2:
        spearman = None
        spearman_reason = "constant_prediction"
    else:
        statistic = float(spearmanr(targets, predictions).statistic)
        if math.isfinite(statistic):
            spearman = statistic
        else:
            spearman = None
            spearman_reason = "non_finite_statistic"

    kendall_reason: str | None = None
    if count < 2:
        kendall = None
        kendall_reason = "fewer_than_two_rows"
    elif distinct_targets < 2:
        kendall = None
        kendall_reason = "constant_target"
    elif distinct_predictions < 2:
        kendall = None
        kendall_reason = "constant_prediction"
    else:
        statistic = float(kendalltau(targets, predictions, variant="b").statistic)
        if math.isfinite(statistic):
            kendall = statistic
        else:
            kendall = None
            kendall_reason = "non_finite_statistic"

    comparable, correct, prediction_ties = _strict_pair_counts(
        targets, predictions
    )
    pair_reason = None if comparable else "no_comparable_target_pairs"
    total_pairs = count * (count - 1) // 2
    return {
        "hidden_tail_count": count,
        "song_group_count": len(set(song_groups)),
        "distinct_target_count": distinct_targets,
        "distinct_prediction_count": distinct_predictions,
        "total_pair_count": total_pairs,
        "target_tie_pair_count": total_pairs - comparable,
        "comparable_pair_count": comparable,
        "strict_pair_correct_count": correct,
        "strict_pair_incorrect_count": comparable - correct,
        "prediction_tie_comparable_pair_count": prediction_ties,
        "strict_comparable_pair_accuracy": (
            float(correct / comparable) if comparable else None
        ),
        "strict_pair_accuracy_undefined_reason": pair_reason,
        "spearman_rho": spearman,
        "spearman_undefined_reason": spearman_reason,
        "kendall_tau_b": kendall,
        "kendall_undefined_reason": kendall_reason,
    }


def _panel_predictions(
    panel: HiddenTailPanel,
    predictions: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    source_predictions = _float_values(predictions, name)
    if len(source_predictions) != panel.source_size:
        raise ValueError(
            f"{name} has {len(source_predictions)} rows; "
            f"the panel source has {panel.source_size}"
        )
    return source_predictions[np.asarray(panel.source_indices, dtype=np.int64)]


def hidden_tail_rank_metrics(
    panel: HiddenTailPanel,
    predictions: Sequence[float] | np.ndarray,
) -> dict[str, float | int | str | None]:
    """Compute tie-aware Spearman and strict comparable-pair accuracy."""

    panel_predictions = _panel_predictions(panel, predictions, "predictions")
    targets = np.asarray(panel.original_targets, dtype=np.float64)
    metrics = _rank_metrics_arrays(
        targets,
        panel_predictions,
        panel.song_groups,
    )
    metrics["strictly_hidden_count"] = int(sum(panel.is_strictly_hidden))
    return metrics


def _difference_summary(
    values: Sequence[float],
    *,
    point_difference: float | None,
    requested_replicates: int,
    confidence_level: float,
    forced_reason: str | None = None,
) -> dict[str, float | int | str | None]:
    valid = np.asarray(values, dtype=np.float64)
    undefined_replicates = requested_replicates - len(valid)
    if forced_reason is not None:
        reason = forced_reason
    elif len(valid) == 0:
        reason = "no_valid_bootstrap_replicates"
    else:
        reason = None
    if len(valid):
        tail = (1.0 - confidence_level) / 2.0
        low, high = np.quantile(valid, [tail, 1.0 - tail])
        mean = float(valid.mean())
        ci_low = float(low)
        ci_high = float(high)
    else:
        mean = None
        ci_low = None
        ci_high = None
    return {
        "point_difference": point_difference,
        "bootstrap_mean_difference": mean,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "valid_replicates": len(valid),
        "undefined_replicates": undefined_replicates,
        "undefined_reason": reason,
    }


def paired_song_group_bootstrap_difference(
    panel: HiddenTailPanel,
    first_predictions: Sequence[float] | np.ndarray,
    second_predictions: Sequence[float] | np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 2027,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap paired rank-metric differences over whole song groups.

    Differences are always ``first - second``.  Every replicate draws the
    panel's unique song groups with replacement and retains all rows belonging
    to each drawn group.  Undefined rank metrics are counted rather than
    silently replaced with zero.
    """

    replicate_count = _integer(replicates, "replicates")
    bootstrap_seed = _integer(seed, "seed")
    if replicate_count <= 0:
        raise ValueError("replicates must be positive")
    confidence = _finite_real(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    first = _panel_predictions(panel, first_predictions, "first_predictions")
    second = _panel_predictions(panel, second_predictions, "second_predictions")
    targets = np.asarray(panel.original_targets, dtype=np.float64)
    groups = np.asarray(panel.song_groups, dtype=object)
    first_metrics = _rank_metrics_arrays(targets, first, panel.song_groups)
    second_metrics = _rank_metrics_arrays(targets, second, panel.song_groups)

    first_spearman = first_metrics["spearman_rho"]
    second_spearman = second_metrics["spearman_rho"]
    spearman_point = (
        float(first_spearman) - float(second_spearman)
        if first_spearman is not None and second_spearman is not None
        else None
    )
    first_pairs = first_metrics["strict_comparable_pair_accuracy"]
    second_pairs = second_metrics["strict_comparable_pair_accuracy"]
    pair_point = (
        float(first_pairs) - float(second_pairs)
        if first_pairs is not None and second_pairs is not None
        else None
    )

    unique_groups = tuple(sorted(set(panel.song_groups)))
    global_reason = (
        "empty_hidden_tail_panel"
        if not unique_groups
        else "fewer_than_two_song_groups"
        if len(unique_groups) < 2
        else None
    )
    spearman_differences: list[float] = []
    pair_differences: list[float] = []
    if global_reason is None:
        positions_by_group = {
            group: np.flatnonzero(groups == group)
            for group in unique_groups
        }
        generator = np.random.default_rng(bootstrap_seed)
        for _ in range(replicate_count):
            draws = generator.integers(
                0, len(unique_groups), size=len(unique_groups)
            )
            positions = np.concatenate(
                [positions_by_group[unique_groups[index]] for index in draws]
            )
            sampled_groups = tuple(str(groups[index]) for index in positions)
            sampled_targets = targets[positions]
            first_sample = _rank_metrics_arrays(
                sampled_targets, first[positions], sampled_groups
            )
            second_sample = _rank_metrics_arrays(
                sampled_targets, second[positions], sampled_groups
            )
            first_rho = first_sample["spearman_rho"]
            second_rho = second_sample["spearman_rho"]
            if first_rho is not None and second_rho is not None:
                spearman_differences.append(
                    float(first_rho) - float(second_rho)
                )
            first_accuracy = first_sample[
                "strict_comparable_pair_accuracy"
            ]
            second_accuracy = second_sample[
                "strict_comparable_pair_accuracy"
            ]
            if first_accuracy is not None and second_accuracy is not None:
                pair_differences.append(
                    float(first_accuracy) - float(second_accuracy)
                )

    return {
        "hidden_tail_count": len(panel.source_indices),
        "song_group_count": len(unique_groups),
        "bootstrap_replicates": replicate_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence,
        "bootstrap_undefined_reason": global_reason,
        "first_metrics": first_metrics,
        "second_metrics": second_metrics,
        "spearman_difference": _difference_summary(
            spearman_differences,
            point_difference=spearman_point,
            requested_replicates=replicate_count,
            confidence_level=confidence,
            forced_reason=global_reason,
        ),
        "strict_pair_accuracy_difference": _difference_summary(
            pair_differences,
            point_difference=pair_point,
            requested_replicates=replicate_count,
            confidence_level=confidence,
            forced_reason=global_reason,
        ),
    }
