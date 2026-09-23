"""Symbolic Oni chart windows for the paper's sequence experiment."""

from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import Audio
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from .tokenizer import EventTokenizer


@dataclass
class ChartBag:
    """One Oni chart with its measure-aligned symbolic event windows."""

    song_id: str
    difficulty: str
    star: int
    instances: list[torch.Tensor] = field(default_factory=list)
    instance_masks: list[torch.Tensor] = field(default_factory=list)
    instance_measures: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.instances)


class TaikoChartDataset(Dataset):
    """Read the frozen Oni panel as measure-aligned symbolic event windows."""

    def __init__(
        self,
        *,
        hf_dataset: HFDataset,
        song_ids: list[str],
        window_measures: list[int],
        hop_measures: int = 2,
        max_instances_per_chart: int = 64,
        max_tokens_per_instance: int = 128,
    ) -> None:
        if not window_measures or any(value <= 0 for value in window_measures):
            raise ValueError("window_measures must contain positive sizes")
        if hop_measures <= 0 or max_instances_per_chart <= 0 or max_tokens_per_instance <= 0:
            raise ValueError("window and instance limits must be positive")
        if len(song_ids) != len(hf_dataset):
            raise ValueError(
                "song_ids length must match the number of dataset rows: "
                f"{len(song_ids)} != {len(hf_dataset)}"
            )
        self.window_measures = list(window_measures)
        self.hop_measures = hop_measures
        self.max_instances_per_chart = max_instances_per_chart
        self.max_tokens_per_instance = max_tokens_per_instance
        self.tokenizer = EventTokenizer()
        if "audio" in hf_dataset.column_names:
            hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
        self.hf_dataset = hf_dataset
        self.song_ids = list(song_ids)
        self._build_chart_index()

    def _build_chart_index(self) -> None:
        """Index only Oni charts; the frozen record table checks completeness."""
        self.chart_index: list[tuple[int, str]] = []
        for song_idx in range(len(self.hf_dataset)):
            song = self.hf_dataset[song_idx]
            chart = song.get("oni")
            if chart is not None and chart.get("segments"):
                self.chart_index.append((song_idx, "oni"))

    def __len__(self) -> int:
        return len(self.chart_index)

    def _process_chart(
        self,
        song_data: dict,
        song_idx: int,
    ) -> ChartBag:
        """Tokenize a source Oni chart without substituting missing labels."""
        song_id = self.song_ids[song_idx]
        chart = song_data["oni"]
        star = chart["level"]
        if isinstance(star, bool) or not isinstance(star, int) or not 1 <= star <= 10:
            raise ValueError(f"invalid Oni star rating for {song_id}: {star!r}")
        segments = chart["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"missing Oni segments for {song_id}")
        tokens = self.tokenizer.tokenize_chart(segments)

        # Create multi-scale windows (measure-based in v2)
        all_instances = []
        all_masks = []
        all_measures = []

        for window_size in self.window_measures:
            windows = self.tokenizer.create_windows(
                tokens,
                window_measures=window_size,
                hop_measures=self.hop_measures,
            )

            for window_tokens in windows:
                if not window_tokens:
                    continue

                # Convert to tensor
                tensor, mask = self.tokenizer.tokens_to_tensor(
                    window_tokens,
                    max_length=self.max_tokens_per_instance,
                )

                # Pad to max length
                tensor, mask = self.tokenizer.pad_sequence(
                    tensor, mask, self.max_tokens_per_instance
                )

                # Record measure range
                start_measure = window_tokens[0].measure_index
                end_measure = window_tokens[-1].measure_index

                all_instances.append(tensor)
                all_masks.append(mask)
                all_measures.append((start_measure, end_measure))

        # Limit number of instances
        if len(all_instances) > self.max_instances_per_chart:
            # Sample uniformly
            indices = np.linspace(
                0, len(all_instances) - 1, self.max_instances_per_chart, dtype=int
            )
            all_instances = [all_instances[i] for i in indices]
            all_masks = [all_masks[i] for i in indices]
            all_measures = [all_measures[i] for i in indices]

        return ChartBag(
            song_id=song_id,
            difficulty="oni",
            star=star,
            instances=all_instances,
            instance_masks=all_masks,
            instance_measures=all_measures,
        )

    def __getitem__(self, idx: int) -> ChartBag:
        song_idx, _ = self.chart_index[idx]
        song_data = self.hf_dataset[song_idx]
        return self._process_chart(song_data, song_idx)


def collate_chart_bags(bags: list[ChartBag]) -> dict:
    """
    Collate function for ChartBag instances.

    Args:
        bags: List of ChartBag instances to collate

    Returns a dictionary suitable for model input.
    """
    if not bags:
        raise ValueError("cannot collate an empty chart-bag batch")
    empty = [
        f"{bag.song_id}:{bag.difficulty}"
        for bag in bags
        if not bag.instances
    ]
    if empty:
        raise ValueError(
            "every chart bag must contain at least one measure-aligned "
            f"instance; empty={empty}"
        )

    # Stack instances: need to handle variable numbers
    max_instances = max(len(b.instances) for b in bags)

    # Pad instances to same count
    batch_instances = []
    batch_masks = []
    instance_counts = []

    for bag in bags:
        instances = bag.instances
        masks = bag.instance_masks

        # Pad to max_instances
        n_pad = max_instances - len(instances)
        if n_pad > 0:
            pad_shape = instances[0].shape
            instances = instances + [torch.zeros(pad_shape) for _ in range(n_pad)]
            masks = masks + [torch.zeros(pad_shape[0]) for _ in range(n_pad)]

        batch_instances.append(torch.stack(instances))
        batch_masks.append(torch.stack(masks))
        instance_counts.append(len(bag.instances))

    return {
        "instances": torch.stack(batch_instances),  # [B, N, L, 6]
        "instance_masks": torch.stack(batch_masks),  # [B, N, L]
        "instance_counts": torch.tensor(instance_counts),  # [B]
        "star": torch.tensor([b.star for b in bags], dtype=torch.float32),  # [B]
        "song_ids": [b.song_id for b in bags],  # List[str]
    }
