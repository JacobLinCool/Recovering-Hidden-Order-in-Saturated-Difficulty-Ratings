#!/usr/bin/env python
"""Render the Dan-i Dojo external criterion for the ICASSP paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from TaikoChartEstimator.research.dojo import HIGH_DAN_ORDER
from TaikoChartEstimator.research.evidence import atomic_write_json

# Single-column layout: 86 mm = spconf column width ((178 mm - 6 mm) / 2).
FIG_WIDTH = 86.0 / 25.4
FIG_HEIGHT = 1.45
LEFT = 0.40
RIGHT = 0.07
BOTTOM = 0.22
TOP_MARGIN = 0.20
X_MARGIN = 0.36
FONT_SIZE = 9.0
TITLE_SIZE = 9.4
INK = "#17212b"
MUTED = "#66727d"
TIER_LABELS = ("10th Dan", "Kuroto", "Meijin", "Chojin", "Tatsujin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/icassp2027/artifact/public/dojo_criterion_summary.jsonl"),
    )
    parser.add_argument(
        "--matched",
        type=Path,
        default=Path("paper/icassp2027/artifact/public/dojo_public_matches.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/icassp2027/generated/dojo_criterion.pdf"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_row(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def deterministic_jitter(identity: str) -> float:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:4], "big") / 2**32
    return (fraction - 0.5) * 0.22


def centered_primary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = [
        dict(row)
        for row in rows
        if row["dan"] in HIGH_DAN_ORDER and int(row["official_star"]) == 10
    ]
    if any(
        int(row["expert_tier_order"]) != HIGH_DAN_ORDER[str(row["dan"])]
        for row in primary
    ):
        raise ValueError("public Dojo tier order is inconsistent")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        grouped[int(row["year"])].append(row)
    output: list[dict[str, Any]] = []
    for year, year_rows in sorted(grouped.items()):
        scores = np.asarray([float(row["latent_score"]) for row in year_rows])
        mean = float(np.mean(scores))
        standard_deviation = float(np.std(scores, ddof=0))
        if standard_deviation <= 0.0:
            raise ValueError(f"year {year} has no score variation")
        for row in year_rows:
            output.append(
                {
                    **row,
                    "centered_score": (float(row["latent_score"]) - mean)
                    / standard_deviation,
                }
            )
    return output


def main() -> None:
    args = parse_args()
    summaries = read_jsonl(args.summary)
    matches = read_jsonl(args.matched)
    global_summary = unique_row(
        summaries, family="global", condition="topcoded_huber"
    )
    sequence_summary = unique_row(
        summaries, family="sequence", condition="seq_ordinal_logit"
    )
    global_matches = [
        row
        for row in matches
        if row.get("family") == "global"
        and row.get("condition") == "topcoded_huber"
    ]
    centered = centered_primary_rows(global_matches)
    if len(centered) != 47:
        raise ValueError("Dojo figure requires the frozen 47-placement primary panel")
    for summary in (global_summary, sequence_summary):
        for metric in ("macro_year_spearman", "within_year_pair_accuracy"):
            if int(summary[metric]["valid_bootstrap_replicates"]) != 10_000:
                raise ValueError("Dojo figure requires 10,000 valid bootstraps")

    # Agreement estimates are reported in the text; the JSON keeps them so the
    # verifier can match the figure data against canonical evidence.
    systems = (
        (global_summary, "Global"),
        (sequence_summary, "Sequence"),
    )
    plot_data: list[dict[str, Any]] = []
    for metric in ("macro_year_spearman", "within_year_pair_accuracy"):
        for summary, _system_label in systems:
            record = summary[metric]
            point = float(record["point"])
            low = float(record["bootstrap_ci_low"])
            high = float(record["bootstrap_ci_high"])
            if not 0.0 <= low <= point <= high <= 1.0:
                raise ValueError(f"{metric} interval must lie inside 0-1")
            plot_data.append(
                {
                    "family": summary["family"],
                    "condition": summary["condition"],
                    "metric": metric,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "permutation_p_value": float(record["permutation_p_value"]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": FONT_SIZE,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    # Explicit inch-based layout at the printed single-column width.
    axis_scores = figure.add_axes(
        [
            LEFT / FIG_WIDTH,
            BOTTOM / FIG_HEIGHT,
            (FIG_WIDTH - LEFT - RIGHT) / FIG_WIDTH,
            (FIG_HEIGHT - BOTTOM - TOP_MARGIN) / FIG_HEIGHT,
        ]
    )

    years = sorted({int(row["year"]) for row in centered})
    colors = plt.cm.Blues(np.linspace(0.38, 0.88, len(years)))
    for color, year in zip(colors, years, strict=True):
        year_rows = [row for row in centered if int(row["year"]) == year]
        for row in year_rows:
            tier = int(row["expert_tier_order"])
            identity = (
                f"{row['year']}:{row['dan']}:{row['position']}:"
                f"{row['chart_key']}"
            )
            axis_scores.scatter(
                tier + deterministic_jitter(identity),
                float(row["centered_score"]),
                s=16,
                color=color,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.5,
                zorder=2,
            )
    tier_means = []
    for tier in range(len(HIGH_DAN_ORDER)):
        values = [
            float(row["centered_score"])
            for row in centered
            if int(row["expert_tier_order"]) == tier
        ]
        tier_means.append(float(np.mean(values)))
    axis_scores.plot(
        range(len(tier_means)),
        tier_means,
        color=INK,
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        zorder=3,
        label="tier mean",
    )
    axis_scores.axhline(0.0, color=MUTED, linestyle=(0, (3.0, 2.0)), linewidth=0.8, zorder=0)
    axis_scores.set_xticks(range(len(TIER_LABELS)), TIER_LABELS)
    axis_scores.set_xlim(-X_MARGIN, len(TIER_LABELS) - 1 + X_MARGIN)
    axis_scores.set_yticks([-1, 0, 1, 2])
    axis_scores.set_ylabel("Score (z)", labelpad=2)
    axis_scores.set_title(
        "Scores by course level",
        loc="left",
        fontweight="bold",
        fontsize=TITLE_SIZE,
        pad=3,
    )
    axis_scores.spines[["top", "right"]].set_visible(False)
    axis_scores.tick_params(colors=MUTED, length=2.5, pad=1)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".svg"))
    atomic_write_json(
        args.output.with_suffix(".json"),
        {
            "schema_version": "icassp2027_dojo_figure_v1",
            "placement_count": len(centered),
            "year_count": len(years),
            "tier_means_within_year_z": tier_means,
            "plot_data": plot_data,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
