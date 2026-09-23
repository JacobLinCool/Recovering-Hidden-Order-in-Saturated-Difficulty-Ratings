"""Frozen Oni panel and partition helpers for sequence OOF experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import DatasetDict
from torch.utils.data import Dataset

from TaikoChartEstimator.data.v2.dataset import (
    ChartBag,
    TaikoChartDataset,
    collate_chart_bags,
)
from TaikoChartEstimator.research.data import load_manifest_panel


@dataclass(frozen=True)
class SequenceChartRecord:
    """Frozen assignment and labels for one Oni chart."""

    chart_id: str
    song_id: str
    normalized_title: str
    original_label: int
    observed_label: int
    outer_test_fold: int
    inner_validation_for_outer_folds: tuple[int, ...]


def load_sequence_chart_records(
    path: str | Path,
    *,
    cap: int,
    expected_count: int = 1023,
) -> tuple[SequenceChartRecord, ...]:
    """Load the frozen fold table and derive scenario-observed labels."""

    if cap < 2:
        raise ValueError("cap must be at least 2")
    source = Path(path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} frozen Oni charts, found {len(rows)}"
        )
    records: list[SequenceChartRecord] = []
    seen_charts: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{source}:{index + 1}"
        chart_id = str(row.get("chart_id", ""))
        song_id = str(row.get("song_id", ""))
        normalized_title = str(row.get("normalized_title", ""))
        if not chart_id or not song_id or not normalized_title:
            raise ValueError(f"{context} has empty chart metadata")
        if chart_id != f"{song_id}:oni":
            raise ValueError(f"{context} is not the frozen Oni chart")
        if chart_id in seen_charts:
            raise ValueError(f"duplicate chart assignment: {chart_id}")
        seen_charts.add(chart_id)
        original = int(row["original_star"])
        outer_fold = int(row["outer_test_fold"])
        validation_folds = tuple(
            sorted(int(value) for value in row["inner_validation_for_outer_folds"])
        )
        if original < 1 or original > 10:
            raise ValueError(f"{context} has invalid original label {original}")
        if outer_fold not in range(5):
            raise ValueError(f"{context} has invalid outer fold {outer_fold}")
        if any(value not in range(5) for value in validation_folds):
            raise ValueError(f"{context} has an invalid validation fold")
        if outer_fold in validation_folds:
            raise ValueError(f"{context} puts a test chart in validation")
        records.append(
            SequenceChartRecord(
                chart_id=chart_id,
                song_id=song_id,
                normalized_title=normalized_title,
                original_label=original,
                observed_label=min(original, cap),
                outer_test_fold=outer_fold,
                inner_validation_for_outer_folds=validation_folds,
            )
        )
    records.sort(key=lambda record: record.chart_id)
    title_assignments: dict[str, tuple[int, tuple[int, ...]]] = {}
    for record in records:
        assignment = (
            record.outer_test_fold,
            record.inner_validation_for_outer_folds,
        )
        previous = title_assignments.setdefault(record.normalized_title, assignment)
        if previous != assignment:
            raise ValueError(
                f"frozen fold table splits normalized title {record.normalized_title!r}"
            )
    return tuple(records)


def partition_record_indices(
    records: Sequence[SequenceChartRecord],
    *,
    outer_fold: int,
) -> dict[str, tuple[int, ...]]:
    """Return disjoint train, validation, refit, and test chart indices."""

    if outer_fold not in range(5):
        raise ValueError("outer_fold must be in [0, 4]")
    test = tuple(
        index
        for index, record in enumerate(records)
        if record.outer_test_fold == outer_fold
    )
    validation = tuple(
        index
        for index, record in enumerate(records)
        if outer_fold in record.inner_validation_for_outer_folds
    )
    train = tuple(
        index
        for index, record in enumerate(records)
        if record.outer_test_fold != outer_fold
        and outer_fold not in record.inner_validation_for_outer_folds
    )
    refit = tuple(
        index
        for index, record in enumerate(records)
        if record.outer_test_fold != outer_fold
    )
    partitions = {
        "train": train,
        "validation": validation,
        "refit": refit,
        "test": test,
    }
    if not train or not validation or not test:
        raise ValueError(f"outer fold {outer_fold} has an empty partition")
    if set(train) & set(validation) or set(refit) & set(test):
        raise RuntimeError(f"outer fold {outer_fold} has overlapping partitions")
    if set(train) | set(validation) != set(refit):
        raise RuntimeError(f"outer fold {outer_fold} refit partition is inconsistent")
    if set(refit) | set(test) != set(range(len(records))):
        raise RuntimeError(f"outer fold {outer_fold} does not cover the panel")
    title_sets = {
        name: {records[index].normalized_title for index in indices}
        for name, indices in partitions.items()
    }
    if (
        title_sets["train"] & title_sets["validation"]
        or title_sets["refit"] & title_sets["test"]
    ):
        raise RuntimeError(f"outer fold {outer_fold} leaks normalized titles")
    return partitions


class OniSequenceDataset(Dataset[ChartBag]):
    """Lazy chart-token dataset whose exposed label is already top-coded."""

    def __init__(
        self,
        records: Sequence[SequenceChartRecord],
        manifest: Mapping[str, Any],
        source: DatasetDict,
        *,
        window_measures: Sequence[int],
        hop_measures: int,
        max_instances_per_chart: int,
        max_tokens_per_instance: int,
        cache_dir: str | None = None,
    ) -> None:
        self.records = tuple(records)
        self.record_by_song = {record.song_id: record for record in self.records}
        if len(self.record_by_song) != len(self.records):
            raise ValueError("the frozen Oni panel contains duplicate song IDs")
        located: dict[str, tuple[TaikoChartDataset, int]] = {}
        for panel in ("train", "validation", "test"):
            hf_dataset, song_ids = load_manifest_panel(
                manifest,
                panel,
                cache_dir=cache_dir,
                source=source,
            )
            dataset = TaikoChartDataset(
                hf_dataset=hf_dataset,
                song_ids=song_ids,
                window_measures=list(window_measures),
                hop_measures=int(hop_measures),
                max_instances_per_chart=int(max_instances_per_chart),
                max_tokens_per_instance=int(max_tokens_per_instance),
            )
            for chart_index, (song_index, difficulty) in enumerate(dataset.chart_index):
                if difficulty != "oni":
                    continue
                song_id = dataset.song_ids[song_index]
                if song_id not in self.record_by_song:
                    continue
                if song_id in located:
                    raise ValueError(f"duplicate source Oni chart for {song_id}")
                located[song_id] = (dataset, chart_index)
        missing = sorted(set(self.record_by_song).difference(located))
        extra = sorted(set(located).difference(self.record_by_song))
        if missing or extra:
            raise ValueError(
                "source snapshot disagrees with the frozen Oni panel: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        self._items = tuple(located[record.song_id] for record in self.records)
        self._processed_bag_cache: dict[int, ChartBag] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ChartBag:
        cached = self._processed_bag_cache.get(index)
        if cached is not None:
            return cached
        dataset, chart_index = self._items[index]
        bag = dataset[chart_index]
        record = self.records[index]
        if bag.song_id != record.song_id or bag.difficulty != "oni":
            raise RuntimeError("source chart order changed after panel construction")
        if int(bag.star) != record.original_label:
            raise RuntimeError(
                f"source label changed for {record.chart_id}: "
                f"{bag.star} != {record.original_label}"
            )
        observed_bag = replace(bag, star=record.observed_label)
        self._processed_bag_cache[index] = observed_bag
        return observed_bag


class ObservedLabelCollator:
    """Collate charts without exposing original evaluation labels."""

    def __init__(self, records: Sequence[SequenceChartRecord]) -> None:
        self._record_by_song = {record.song_id: record for record in records}

    def __call__(self, bags: list[ChartBag]) -> dict[str, Any]:
        batch = collate_chart_bags(bags)
        observed = batch.pop("star")
        records = [self._record_by_song[song_id] for song_id in batch["song_ids"]]
        expected = [record.observed_label for record in records]
        if observed.tolist() != [float(value) for value in expected]:
            raise RuntimeError("dataset and frozen observed labels disagree")
        batch.update(
            {
                "chart_ids": [record.chart_id for record in records],
                "normalized_titles": [record.normalized_title for record in records],
                "observed_label": observed,
            }
        )
        return batch
