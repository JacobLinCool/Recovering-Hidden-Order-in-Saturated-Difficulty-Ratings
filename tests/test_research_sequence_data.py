from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from TaikoChartEstimator.data.v2.dataset import ChartBag, TaikoChartDataset
from TaikoChartEstimator.research.sequence_data import (
    ObservedLabelCollator,
    OniSequenceDataset,
    SequenceChartRecord,
    load_sequence_chart_records,
    partition_record_indices,
)

class ResearchSequenceDataTest(unittest.TestCase):
    def test_source_dataset_only_indexes_valid_oni_charts(self) -> None:
        class Rows:
            column_names = ("oni", "easy")

            def __init__(self, rows: list[dict]) -> None:
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict:
                return self.rows[index]

        chart = {
            "level": 10,
            "segments": [
                {
                    "timestamp": 0.0,
                    "measure_index": 0,
                    "notes": [
                        {"timestamp": 0.0, "note_type": "Don", "bpm": 120.0}
                    ],
                }
            ],
        }
        source = Rows([{"oni": chart, "easy": chart}, {"oni": None, "easy": chart}])
        dataset = TaikoChartDataset(
            hf_dataset=source,
            song_ids=["first", "second"],
            window_measures=[2, 4],
        )
        self.assertEqual(dataset.chart_index, [(0, "oni")])
        bag = dataset[0]
        self.assertEqual((bag.song_id, bag.difficulty, bag.star), ("first", "oni", 10))
        self.assertEqual(len(bag.instances), 2)

        invalid = Rows([{"oni": {**chart, "level": 11}}])
        invalid_dataset = TaikoChartDataset(
            hf_dataset=invalid,
            song_ids=["invalid"],
            window_measures=[2, 4],
        )
        with self.assertRaisesRegex(ValueError, "invalid Oni star rating"):
            invalid_dataset[0]

    @staticmethod
    def _synthetic_folds() -> tuple[SequenceChartRecord, ...]:
        rows = [
            {
                "chart_id": f"song-{index}:oni",
                "song_id": f"song-{index}",
                "normalized_title": f"group-{index}",
                "original_star": index + 6,
                "outer_test_fold": index,
                "inner_validation_for_outer_folds": [(index - 1) % 5],
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "assignments.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            return load_sequence_chart_records(path, cap=8, expected_count=5)
    def test_processed_chart_is_cached_within_a_worker(self) -> None:
        record = SequenceChartRecord(
            chart_id="song:oni",
            song_id="song",
            normalized_title="song",
            original_label=10,
            observed_label=8,
            outer_test_fold=0,
            inner_validation_for_outer_folds=(1,),
        )
        instance = torch.zeros(4, 6)
        instance[:, 0] = 1
        source_bag = ChartBag(
            song_id="song",
            difficulty="oni",
            star=10,
            instances=[instance],
            instance_masks=[torch.ones(4)],
            instance_measures=[(0, 3)],
        )

        class CountingDataset:
            def __init__(self) -> None:
                self.calls = 0

            def __getitem__(self, index: int) -> ChartBag:
                self.calls += 1
                return source_bag

        source = CountingDataset()
        dataset = OniSequenceDataset.__new__(OniSequenceDataset)
        dataset.records = (record,)
        dataset.record_by_song = {"song": record}
        dataset._items = ((source, 0),)
        dataset._processed_bag_cache = {}
        first = dataset[0]
        second = dataset[0]
        self.assertIs(first, second)
        self.assertEqual(source.calls, 1)
        self.assertEqual(first.star, 8)

    def test_frozen_panel_top_codes_without_changing_original_labels(self) -> None:
        records = self._synthetic_folds()
        self.assertEqual(len(records), 5)
        self.assertTrue(
            all(
                record.observed_label == min(record.original_label, 8)
                for record in records
            )
        )
        self.assertTrue(any(record.original_label == 10 for record in records))
        self.assertTrue(all(record.observed_label <= 8 for record in records))

    def test_every_outer_partition_has_zero_title_leakage(self) -> None:
        records = self._synthetic_folds()
        for outer_fold in range(5):
            with self.subTest(outer_fold=outer_fold):
                partitions = partition_record_indices(records, outer_fold=outer_fold)
                self.assertEqual(
                    set(partitions["refit"]) | set(partitions["test"]),
                    set(range(5)),
                )
                refit_titles = {
                    records[index].normalized_title for index in partitions["refit"]
                }
                test_titles = {
                    records[index].normalized_title for index in partitions["test"]
                }
                self.assertFalse(refit_titles & test_titles)

    def test_collator_exposes_observed_label_but_not_original_label(self) -> None:
        record = SequenceChartRecord(
            chart_id="song:oni",
            song_id="song",
            normalized_title="song",
            original_label=10,
            observed_label=8,
            outer_test_fold=0,
            inner_validation_for_outer_folds=(1,),
        )
        instance = torch.zeros(4, 6)
        instance[:, 0] = 1
        bag = ChartBag(
            song_id="song",
            difficulty="oni",
            star=8,
            instances=[instance],
            instance_masks=[torch.ones(4)],
            instance_measures=[(0, 3)],
        )
        batch = ObservedLabelCollator([record])([bag])
        self.assertEqual(batch["observed_label"].tolist(), [8.0])
        self.assertNotIn("original_label", batch)
        self.assertNotIn("star", batch)
        self.assertNotIn("is_right_censored", batch)


if __name__ == "__main__":
    unittest.main()
