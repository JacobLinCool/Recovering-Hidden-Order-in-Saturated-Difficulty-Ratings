"""Reproducible research utilities for the hidden-order difficulty-rating study."""

from .data import (
    DATASET_NAME,
    DIFFICULTIES,
    ManifestError,
    build_dataset_manifest,
    load_manifest,
    normalize_title,
    validate_dataset_manifest,
)
from .sequence import (
    SEQUENCE_CONDITIONS,
    SEQUENCE_TRAINING_SOURCE_PATHS,
    BeatSynchronousSequenceEstimator,
    OrderedLogitHead,
    SequenceEstimatorConfig,
    SequenceEstimatorOutput,
    standardize_from_training_scores,
)
from .sequence_data import (
    ObservedLabelCollator,
    OniSequenceDataset,
    SequenceChartRecord,
    load_sequence_chart_records,
    partition_record_indices,
)

__all__ = [
    "DATASET_NAME",
    "DIFFICULTIES",
    "ManifestError",
    "SEQUENCE_CONDITIONS",
    "SEQUENCE_TRAINING_SOURCE_PATHS",
    "BeatSynchronousSequenceEstimator",
    "OrderedLogitHead",
    "ObservedLabelCollator",
    "OniSequenceDataset",
    "SequenceEstimatorConfig",
    "SequenceEstimatorOutput",
    "SequenceChartRecord",
    "build_dataset_manifest",
    "load_manifest",
    "normalize_title",
    "load_sequence_chart_records",
    "partition_record_indices",
    "standardize_from_training_scores",
    "validate_dataset_manifest",
]
