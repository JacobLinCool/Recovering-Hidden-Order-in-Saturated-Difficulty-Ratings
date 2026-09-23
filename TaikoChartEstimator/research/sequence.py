"""Signal-native estimators for the top-coded OOF research protocol.

This module deliberately contains only the representation and observation
models needed by the paper's sequence comparison. It does not expose a new
package-level API and does not reuse the older multi-task MIL heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from TaikoChartEstimator.model.v2.encoder import InstanceEncoder

SEQUENCE_CONDITIONS = frozenset(
    {
        "seq_ordinary_huber",
        "seq_topcoded_huber",
        "seq_ordinal_logit",
    }
)

SEQUENCE_TRAINING_SOURCE_PATHS = (
    "TaikoChartEstimator/constants.py",
    "TaikoChartEstimator/data/v2/dataset.py",
    "TaikoChartEstimator/data/v2/tokenizer.py",
    "TaikoChartEstimator/model/v2/encoder.py",
    "TaikoChartEstimator/research/data.py",
    "TaikoChartEstimator/research/evidence.py",
    "TaikoChartEstimator/research/oof.py",
    "TaikoChartEstimator/research/sequence.py",
    "TaikoChartEstimator/research/sequence_data.py",
    "scripts/research/run_sequence_oof_fold.py",
    "pyproject.toml",
    "uv.lock",
)


@dataclass(frozen=True)
class SequenceEstimatorConfig:
    """Frozen architecture parameters for the sequence comparison."""

    d_model: int = 256
    encoder_layers: int = 4
    attention_heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    head_hidden_dim: int = 128
    max_tokens_per_instance: int = 128
    encoder_pooling: str = "mean"
    minimum_threshold_gap: float = 1e-4

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.encoder_layers < 1:
            raise ValueError("encoder_layers must be positive")
        if self.attention_heads < 1 or self.d_model % self.attention_heads:
            raise ValueError("attention_heads must divide d_model")
        if self.encoder_pooling != "mean":
            raise ValueError("the frozen sequence protocol requires mean pooling")
        if self.minimum_threshold_gap <= 0:
            raise ValueError("minimum_threshold_gap must be positive")


@dataclass(frozen=True)
class SequenceEstimatorOutput:
    """Outputs used by observed-label selection and hidden-order evaluation."""

    latent_score: torch.Tensor
    expected_observed_label: torch.Tensor
    thresholds: torch.Tensor | None


class OrderedLogitHead(nn.Module):
    """Cumulative-logit thresholds with ordering guaranteed by construction."""

    def __init__(self, maximum_label: int, minimum_gap: float = 1e-4) -> None:
        super().__init__()
        if maximum_label < 2:
            raise ValueError("maximum_label must be at least 2")
        if minimum_gap <= 0:
            raise ValueError("minimum_gap must be positive")
        self.maximum_label = int(maximum_label)
        self.minimum_gap = float(minimum_gap)
        self.first_threshold = nn.Parameter(torch.tensor(-2.0))
        if self.maximum_label > 2:
            desired_gap = 4.0 / (self.maximum_label - 2)
            inverse_softplus = torch.log(torch.expm1(torch.tensor(desired_gap)))
            self.raw_gaps = nn.Parameter(
                inverse_softplus.repeat(self.maximum_label - 2)
            )
        else:
            self.register_parameter("raw_gaps", None)

    def thresholds(self) -> torch.Tensor:
        first = self.first_threshold.reshape(1)
        if self.raw_gaps is None:
            return first
        positive_gaps = F.softplus(self.raw_gaps) + self.minimum_gap
        return torch.cat((first, first + torch.cumsum(positive_gaps, dim=0)))

    def exceedance_logits(self, latent_score: torch.Tensor) -> torch.Tensor:
        if latent_score.ndim != 1:
            raise ValueError("latent_score must be one-dimensional")
        return latent_score.unsqueeze(1) - self.thresholds().unsqueeze(0)

    def expected_label(self, latent_score: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.sigmoid(self.exceedance_logits(latent_score)).sum(dim=1)

    def loss(
        self,
        latent_score: torch.Tensor,
        observed_label: torch.Tensor,
    ) -> torch.Tensor:
        if observed_label.ndim != 1 or observed_label.shape != latent_score.shape:
            raise ValueError("observed_label must match latent_score")
        rounded = observed_label.round()
        if not torch.equal(rounded, observed_label):
            raise ValueError("ordinal labels must be integers")
        if bool(((rounded < 1) | (rounded > self.maximum_label)).any()):
            raise ValueError("ordinal label lies outside the configured scale")
        boundaries = torch.arange(
            1,
            self.maximum_label,
            device=observed_label.device,
            dtype=observed_label.dtype,
        )
        targets = (observed_label.unsqueeze(1) > boundaries.unsqueeze(0)).to(
            latent_score.dtype
        )
        return F.binary_cross_entropy_with_logits(
            self.exceedance_logits(latent_score), targets
        )


class BeatSynchronousSequenceEstimator(nn.Module):
    """Four-layer event Transformer with masked mean chart pooling."""

    def __init__(
        self,
        condition: str,
        maximum_observed_label: int,
        config: SequenceEstimatorConfig | None = None,
    ) -> None:
        super().__init__()
        if condition not in SEQUENCE_CONDITIONS:
            raise ValueError(
                f"unknown sequence condition {condition!r}; "
                f"expected one of {sorted(SEQUENCE_CONDITIONS)}"
            )
        self.condition = condition
        self.maximum_observed_label = int(maximum_observed_label)
        self.config = config or SequenceEstimatorConfig()
        self.instance_encoder = InstanceEncoder(
            d_model=self.config.d_model,
            n_heads=self.config.attention_heads,
            n_layers=self.config.encoder_layers,
            d_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_tokens_per_instance,
        )
        self.score_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden_dim, 1),
        )
        self.ordinal_head = (
            OrderedLogitHead(
                self.maximum_observed_label,
                minimum_gap=self.config.minimum_threshold_gap,
            )
            if condition == "seq_ordinal_logit"
            else None
        )

    @staticmethod
    def masked_mean_pool(
        instance_embeddings: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> torch.Tensor:
        if instance_embeddings.ndim != 3:
            raise ValueError("instance_embeddings must have shape [B, N, D]")
        if instance_mask.shape != instance_embeddings.shape[:2]:
            raise ValueError("instance_mask must have shape [B, N]")
        weights = instance_mask.to(instance_embeddings.dtype)
        counts = weights.sum(dim=1, keepdim=True)
        if bool((counts == 0).any()):
            raise ValueError("masked mean pooling received an empty chart")
        return (instance_embeddings * weights.unsqueeze(-1)).sum(dim=1) / counts

    def forward(
        self,
        instances: torch.Tensor,
        instance_masks: torch.Tensor,
        instance_counts: torch.Tensor,
    ) -> SequenceEstimatorOutput:
        if instances.ndim != 4 or instances.shape[-1] != 6:
            raise ValueError("instances must have shape [B, N, L, 6]")
        if instance_masks.shape != instances.shape[:3]:
            raise ValueError("instance_masks must have shape [B, N, L]")
        if instance_counts.shape != instances.shape[:1]:
            raise ValueError("instance_counts must have shape [B]")
        batch_size, instance_count, sequence_length, feature_count = instances.shape
        flat_instances = instances.reshape(
            batch_size * instance_count, sequence_length, feature_count
        )
        flat_masks = instance_masks.reshape(
            batch_size * instance_count, sequence_length
        )
        valid_instances = torch.arange(
            instance_count, device=instances.device
        ).unsqueeze(0) < instance_counts.unsqueeze(1)
        flat_valid = valid_instances.reshape(-1)
        if not bool(flat_valid.any()):
            raise ValueError("the batch contains no valid chart windows")
        encoded = torch.zeros(
            batch_size * instance_count,
            self.config.d_model,
            device=instances.device,
            dtype=instances.dtype,
        )
        encoded[flat_valid] = self.instance_encoder(
            flat_instances[flat_valid], flat_masks[flat_valid]
        )
        encoded = encoded.reshape(batch_size, instance_count, self.config.d_model)
        chart_embedding = self.masked_mean_pool(encoded, valid_instances)
        latent_score = self.score_head(chart_embedding).squeeze(-1)
        if self.ordinal_head is None:
            expected = latent_score
            thresholds = None
        else:
            expected = self.ordinal_head.expected_label(latent_score)
            thresholds = self.ordinal_head.thresholds()
        return SequenceEstimatorOutput(
            latent_score=latent_score,
            expected_observed_label=expected,
            thresholds=thresholds,
        )

    def observation_loss(
        self,
        output: SequenceEstimatorOutput,
        observed_label: torch.Tensor,
        *,
        huber_delta: float,
    ) -> torch.Tensor:
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if observed_label.shape != output.latent_score.shape:
            raise ValueError("observed_label must match the model score")
        if bool(
            (
                (observed_label < 1) | (observed_label > self.maximum_observed_label)
            ).any()
        ):
            raise ValueError("observed label lies outside the configured scale")
        if self.condition == "seq_ordinal_logit":
            assert self.ordinal_head is not None
            return self.ordinal_head.loss(output.latent_score, observed_label)
        if self.condition == "seq_ordinary_huber":
            return F.huber_loss(
                output.latent_score,
                observed_label,
                delta=huber_delta,
            )
        below_cap = observed_label < self.maximum_observed_label
        per_item = torch.zeros_like(output.latent_score)
        if bool(below_cap.any()):
            per_item[below_cap] = F.huber_loss(
                output.latent_score[below_cap],
                observed_label[below_cap],
                delta=huber_delta,
                reduction="none",
            )
        at_cap = ~below_cap
        if bool(at_cap.any()):
            shortfall = F.relu(observed_label[at_cap] - output.latent_score[at_cap])
            per_item[at_cap] = F.huber_loss(
                shortfall,
                torch.zeros_like(shortfall),
                delta=huber_delta,
                reduction="none",
            )
        return per_item.mean()


def standardize_from_training_scores(
    training_scores: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Standardize latent scores using outer-training statistics only."""

    if training_scores.ndim != 1 or scores.ndim != 1:
        raise ValueError("training_scores and scores must be one-dimensional")
    if training_scores.numel() < 2:
        raise ValueError("at least two training scores are required")
    if not bool(torch.isfinite(training_scores).all()) or not bool(
        torch.isfinite(scores).all()
    ):
        raise ValueError("scores must be finite")
    mean = training_scores.mean()
    standard_deviation = training_scores.std(unbiased=False)
    if float(standard_deviation) <= 0:
        raise ValueError("training latent scores must have nonzero variance")
    standardized = (scores - mean) / standard_deviation
    return standardized, float(mean), float(standard_deviation)
