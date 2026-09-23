from __future__ import annotations

import unittest

import torch

from TaikoChartEstimator.research.sequence import (
    BeatSynchronousSequenceEstimator,
    OrderedLogitHead,
    SequenceEstimatorConfig,
    SequenceEstimatorOutput,
    standardize_from_training_scores,
)


class ResearchSequenceTest(unittest.TestCase):
    def test_ordered_logit_thresholds_are_strict_by_construction(self) -> None:
        head = OrderedLogitHead(10, minimum_gap=1e-3)
        with torch.no_grad():
            head.raw_gaps.fill_(-100.0)
        thresholds = head.thresholds()
        self.assertEqual(thresholds.shape, (9,))
        self.assertTrue(torch.all(thresholds[1:] > thresholds[:-1]))

    def test_ordered_logit_uses_observed_integer_labels(self) -> None:
        head = OrderedLogitHead(8)
        scores = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        labels = torch.tensor([2.0, 5.0, 8.0])
        loss = head.loss(scores, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(scores.grad)
        with self.assertRaisesRegex(ValueError, "integers"):
            head.loss(scores.detach(), torch.tensor([2.0, 5.5, 8.0]))

    def test_topcoded_huber_does_not_penalize_predictions_above_cap(self) -> None:
        model = BeatSynchronousSequenceEstimator(
            "seq_topcoded_huber",
            8,
            SequenceEstimatorConfig(
                d_model=8,
                encoder_layers=1,
                attention_heads=2,
                feedforward_dim=16,
                head_hidden_dim=4,
                max_tokens_per_instance=4,
            ),
        )
        above = SequenceEstimatorOutput(
            latent_score=torch.tensor([9.0]),
            expected_observed_label=torch.tensor([9.0]),
            thresholds=None,
        )
        below = SequenceEstimatorOutput(
            latent_score=torch.tensor([7.0]),
            expected_observed_label=torch.tensor([7.0]),
            thresholds=None,
        )
        label = torch.tensor([8.0])
        self.assertEqual(
            model.observation_loss(above, label, huber_delta=1.0).item(), 0.0
        )
        self.assertGreater(
            model.observation_loss(below, label, huber_delta=1.0).item(), 0.0
        )

    def test_ordinary_huber_penalizes_above_cap_predictions(self) -> None:
        model = BeatSynchronousSequenceEstimator(
            "seq_ordinary_huber",
            8,
            SequenceEstimatorConfig(
                d_model=8,
                encoder_layers=1,
                attention_heads=2,
                feedforward_dim=16,
                head_hidden_dim=4,
                max_tokens_per_instance=4,
            ),
        )
        output = SequenceEstimatorOutput(
            latent_score=torch.tensor([9.0]),
            expected_observed_label=torch.tensor([9.0]),
            thresholds=None,
        )
        self.assertGreater(
            model.observation_loss(output, torch.tensor([8.0]), huber_delta=1.0).item(),
            0.0,
        )

    def test_masked_mean_excludes_padded_windows(self) -> None:
        embeddings = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
                [[2.0, 4.0], [100.0, 100.0], [100.0, 100.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        pooled = BeatSynchronousSequenceEstimator.masked_mean_pool(embeddings, mask)
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0], [2.0, 4.0]])))

    def test_forward_skips_fully_padded_windows(self) -> None:
        model = BeatSynchronousSequenceEstimator(
            "seq_ordinal_logit",
            8,
            SequenceEstimatorConfig(
                d_model=8,
                encoder_layers=1,
                attention_heads=2,
                feedforward_dim=16,
                head_hidden_dim=4,
                max_tokens_per_instance=4,
            ),
        )
        instances = torch.randn(2, 3, 4, 6)
        instances[..., 0] = torch.randint(0, 9, (2, 3, 4)).float()
        masks = torch.zeros(2, 3, 4)
        masks[0, :2] = 1
        masks[1, :1] = 1
        output = model(instances, masks, torch.tensor([2, 1]))
        self.assertTrue(torch.isfinite(output.latent_score).all())
        self.assertTrue(torch.isfinite(output.expected_observed_label).all())
        self.assertFalse(hasattr(model, "difficulty_classifier"))
        self.assertFalse(hasattr(model, "calibrator"))
        self.assertFalse(hasattr(model, "aggregator"))

    def test_training_only_standardization(self) -> None:
        standardized, mean, standard_deviation = standardize_from_training_scores(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([2.0, 4.0]),
        )
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(standard_deviation, (2.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(float(standardized[0]), 0.0)
        self.assertGreater(float(standardized[1]), 2.0)


if __name__ == "__main__":
    unittest.main()
