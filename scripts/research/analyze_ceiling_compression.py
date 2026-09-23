#!/usr/bin/env python
"""Ceiling-compression diagnostic for the Hidden Order paper (Results 4.2).

Reads the frozen public out-of-fold scores and asks, for every artificial cap
c in {7, 8, 9} and every model, how the capped group (charts whose original
rating is >= c) is placed on the rating scale:

* ``near_fraction``: share of capped-group outputs within +/-delta of c
  (delta 0.02 / 0.05 / 0.10; 0.05 is primary);
* ``near_rho``: Spearman correlation between the original rating and the
  model's ranking score among those near-cap charts;
* ``median_by_original``: median rating-scale output by original rating.

Rating-scale output.  For GBDT, ridge, global Huber losses, and the extended
binary ordinal reduction the evaluated ``latent_score`` already is a rating.
Cumulative logit/probit and every Transformer condition expose a latent score
on another scale; for those the model's ``expected_observed_label`` (expected
observed rating, or the de-standardized Huber prediction for the Transformer
Huber losses) is used as the rating-scale output.  The ranking score used for
``near_rho`` is always the evaluated ``latent_score`` (the field behind the
paper's hidden-tail metrics).  For cumulative models the expected observed
rating is bounded by c, so their near-cap share measures saturation of a
bounded output, not loss of order in the latent score.

Uncertainty: paired title-cluster bootstrap (normalized titles among the
capped group, same resample for every model), primary cap 8 and delta 0.05.

Only frozen public scores and the canonical fold assignments are read; no
model is refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / "paper/manuscript/artifact/public"

GLOBAL_CONDITIONS = (
    "ordinary_huber",
    "topcoded_huber",
    "ridge",
    "gbdt",
    "ordinal_logit",
    "ordinal_probit",
    "ordinal_logit_reduction",
)
SEQUENCE_CONDITIONS = ("seq_ordinary_huber", "seq_topcoded_huber", "seq_ordinal_logit")
CONDITIONS = GLOBAL_CONDITIONS + SEQUENCE_CONDITIONS
LABELS = {
    "ordinary_huber": "Huber (ordinary)",
    "topcoded_huber": "Huber (one-sided)",
    "ridge": "Ridge",
    "gbdt": "GBDT",
    "ordinal_logit": "Cumulative logit",
    "ordinal_probit": "Cumulative probit",
    "ordinal_logit_reduction": "Ordinal reduction",
    "seq_ordinary_huber": "Transformer + ordinary Huber",
    "seq_topcoded_huber": "Transformer + one-sided Huber",
    "seq_ordinal_logit": "Transformer + cumulative logit",
}
# Which field is on the rating scale for the near-cap measurement.
RATING_FIELD = {
    "ordinary_huber": "latent_score",
    "topcoded_huber": "latent_score",
    "ridge": "latent_score",
    "gbdt": "latent_score",
    "ordinal_logit_reduction": "latent_score",
    "ordinal_logit": "expected_observed_label",
    "ordinal_probit": "expected_observed_label",
    "seq_ordinary_huber": "expected_observed_label",
    "seq_topcoded_huber": "expected_observed_label",
    "seq_ordinal_logit": "expected_observed_label",
}
TARGET_KIND = {
    "ordinary_huber": "exact_target",
    "ridge": "exact_target",
    "gbdt": "exact_target",
    "seq_ordinary_huber": "exact_target",
    "topcoded_huber": "lower_bound",
    "seq_topcoded_huber": "lower_bound",
    "ordinal_logit": "ordinal_cumulative",
    "ordinal_probit": "ordinal_cumulative",
    "seq_ordinal_logit": "ordinal_cumulative",
    "ordinal_logit_reduction": "ordinal_binary_reduction",
}
SCALE_NOTE = {
    "latent_score": "evaluated score is itself on the rating scale",
    "expected_observed_label": "evaluated latent score is on another scale; "
    "rating-scale output is the model's expected observed rating "
    "(cumulative: bounded by the cap; Transformer Huber: de-standardized "
    "prediction averaged over seeds, rank-consistent with the latent score "
    "up to seed averaging; see rating_vs_latent_spearman)",
}
MACRO_TAGS = {
    "ordinary_huber": "Huber",
    "topcoded_huber": "OneSided",
    "ridge": "Ridge",
    "gbdt": "Gbdt",
    "ordinal_logit": "Logit",
    "ordinal_probit": "Probit",
    "ordinal_logit_reduction": "Reduction",
    "seq_ordinary_huber": "SeqHuber",
    "seq_topcoded_huber": "SeqOneSided",
    "seq_ordinal_logit": "SeqLogit",
}
MACRO_CONDITIONS = (
    "gbdt",
    "ordinary_huber",
    "ridge",
    "topcoded_huber",
    "ordinal_logit_reduction",
    "seq_ordinary_huber",
    "seq_topcoded_huber",
    "seq_ordinal_logit",
)
MEDIAN_MACRO_CONDITIONS = ("gbdt", "ordinary_huber", "topcoded_huber")
RATING_WORDS = {8: "Eight", 9: "Nine", 10: "Ten"}
CAPS = {"cap7": 7, "cap8": 8, "cap9": 9}
DELTAS = (0.02, 0.05, 0.10)
PRIMARY_SCENARIO = "cap8"
PRIMARY_DELTA = 0.05
MIN_NEAR_FOR_RHO = 5
# A near-cap rho is emitted as a citable macro only when it rests on at least
# 30 charts and is defined in every bootstrap replicate.
MIN_NEAR_FOR_RHO_MACRO = 30

# Preliminary values computed independently by the lead (cap 8, delta 0.05,
# n = 699 capped charts).  (near count, near rho, medians for 8/9/10).
LEAD_PRELIMINARY = {
    "gbdt": (390, 0.028, (7.853, 7.994, 7.997)),
    "ordinary_huber": (113, 0.008, (7.675, 7.964, 8.088)),
    "ridge": (81, -0.164, None),
    "ordinal_logit_reduction": (421, 0.683, (7.854, 7.992, 8.000)),
    "topcoded_huber": (22, 0.224, (8.262, 9.269, 10.412)),
}

FIG_WIDTH = 3.386  # ICASSP column width: (178 mm - 6 mm) / 2
FIG_HEIGHT = 1.30
INK = "#17212b"
MUTED = "#66727d"
GUIDE = "#9aa4ad"
FIG_MODELS = (
    ("gbdt", "GBDT", "#d97706", "o"),
    ("ordinary_huber", "Ordinary Huber", "#66727d", "s"),
    ("topcoded_huber", "One-sided Huber", "#2b6cb0", "D"),
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    paper = here.parents[2] / "paper/manuscript"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--global-scores", type=Path, default=PUBLIC / "global_oof_scores.jsonl")
    parser.add_argument("--sequence-scores", type=Path, default=PUBLIC / "sequence_oof_scores.jsonl")
    parser.add_argument("--global-metrics", type=Path, default=PUBLIC / "global_hidden_tail_metrics.jsonl")
    parser.add_argument("--sequence-metrics", type=Path, default=PUBLIC / "sequence_hidden_tail_metrics.jsonl")
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=REPO / "experiments/icassp2027_topcoded_oof_v1/canonical/fold_assignments.jsonl",
    )
    parser.add_argument("--output-json", type=Path, default=paper / "generated/ceiling_compression.json")
    parser.add_argument("--output-macros", type=Path, default=paper / "generated/ceiling_compression_numbers.tex")
    parser.add_argument("--output-figure", type=Path, default=paper / "generated/ceiling_compression.pdf")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repo_path(path: Path) -> str:
    """Record a path relative to the repo root when it lies inside it."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chart_key(chart_id: str) -> str:
    """Same one-way key as scripts/research/build_public_artifact.py."""
    digest = hashlib.sha256(f"chart\0{chart_id}".encode("utf-8")).hexdigest()
    return f"chart-{digest[:20]}"


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def percentile_ci(values: list[float]) -> tuple[float, float] | None:
    finite = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if finite.size == 0:
        return None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def load_panel(args: argparse.Namespace) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], dict[str, str]]:
    titles = {chart_key(row["chart_id"]): str(row["normalized_title"]) for row in read_jsonl(args.fold_assignments)}
    index: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in (args.global_scores, args.sequence_scores):
        for row in read_jsonl(path):
            key = (str(row["scenario"]), str(row["condition"]))
            if row["chart_key"] in index[key]:
                raise ValueError(f"duplicate chart in {key}")
            index[key][row["chart_key"]] = row
    for key, rows in index.items():
        missing = set(rows) - set(titles)
        if missing:
            raise ValueError(f"{key}: {len(missing)} chart keys lack a title group")
    return index, titles


def arrays_for(
    index: dict[tuple[str, str], dict[str, dict[str, Any]]],
    scenario: str,
    condition: str,
    keys: list[str],
) -> dict[str, np.ndarray]:
    rows = index[(scenario, condition)]
    field = RATING_FIELD[condition]
    return {
        "original": np.asarray([float(rows[k]["evaluation_original_label"]) for k in keys]),
        "rank": np.asarray([float(rows[k]["latent_score"]) for k in keys]),
        "rating": np.asarray([float(rows[k][field]) for k in keys]),
    }


def point_stats(original: np.ndarray, rank: np.ndarray, rating: np.ndarray, cap: int, delta: float) -> dict[str, Any]:
    near = np.abs(rating - cap) <= delta + 1e-12
    count = int(near.sum())
    above = int((rating > cap + delta + 1e-12).sum())
    rho = spearman(original[near], rank[near]) if count >= MIN_NEAR_FOR_RHO else None
    return {
        "near_count": count,
        "near_fraction": count / len(rating),
        "above_count": above,
        "above_fraction": above / len(rating),
        "near_rho": rho,
        "near_rho_undefined_reason": None if rho is not None else "fewer than 5 near-cap charts or constant input",
        "near_original_counts": {str(int(r)): int(((original == r) & near).sum()) for r in range(cap, 11)},
    }


def medians(original: np.ndarray, rating: np.ndarray, cap: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for rating_value in range(cap, 11):
        mask = original == rating_value
        out[str(rating_value)] = float(np.median(rating[mask])) if mask.any() else None
    return out


def tie_fraction(rank: np.ndarray) -> float:
    _, counts = np.unique(rank, return_counts=True)
    n = len(rank)
    return float((counts * (counts - 1) / 2).sum() / (n * (n - 1) / 2))


def bootstrap_primary(
    data: dict[str, dict[str, np.ndarray]],
    clusters: list[np.ndarray],
    cap: int,
    delta: float,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float | None]]] = {
        c: defaultdict(list) for c in data
    }
    n_clusters = len(clusters)
    for _ in range(replicates):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([clusters[i] for i in chosen])
        for condition, arrays in data.items():
            original = arrays["original"][idx]
            rating = arrays["rating"][idx]
            rank = arrays["rank"][idx]
            near = np.abs(rating - cap) <= delta + 1e-12
            draws[condition]["near_fraction"].append(float(near.mean()))
            draws[condition]["near_rho"].append(
                spearman(original[near], rank[near]) if near.sum() >= MIN_NEAR_FOR_RHO else None
            )
            for rating_value in range(cap, 11):
                mask = original == rating_value
                draws[condition][f"median_{rating_value}"].append(
                    float(np.median(rating[mask])) if mask.any() else None
                )
    summary: dict[str, dict[str, Any]] = {}
    for condition, metrics in draws.items():
        summary[condition] = {}
        for name, values in metrics.items():
            ci = percentile_ci(values)
            summary[condition][name] = {
                "ci_low": None if ci is None else ci[0],
                "ci_high": None if ci is None else ci[1],
                "valid_replicates": int(sum(v is not None for v in values)),
            }
    return summary


def check_against_metrics(results: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Recompute full capped-group Spearman and compare with frozen metrics."""
    frozen: dict[tuple[str, str], float] = {}
    for row in read_jsonl(args.global_metrics):
        frozen[(row["scenario"], row["condition"])] = float(row["spearman_rho"])
    for row in read_jsonl(args.sequence_metrics):
        frozen[(row["scenario"], row["condition"])] = float(row["spearman_rho"])
    checked = []
    for scenario, per_condition in results.items():
        for condition, value in per_condition.items():
            key = (scenario, condition)
            if key not in frozen:
                continue
            if abs(frozen[key] - value["capped_group_spearman"]) > 1e-9:
                raise AssertionError(f"{key}: recomputed rho {value['capped_group_spearman']} != frozen {frozen[key]}")
            checked.append(f"{scenario}/{condition}")
    return checked


def check_lead(results: dict[str, Any]) -> list[str]:
    checked = []
    primary = results[PRIMARY_SCENARIO]
    for condition, (count, rho, meds) in LEAD_PRELIMINARY.items():
        row = primary[condition]
        stats = row["by_delta"][f"{PRIMARY_DELTA:.2f}"]
        assert row["capped_count"] == 699, row["capped_count"]
        assert stats["near_count"] == count, (condition, stats["near_count"], count)
        assert round(stats["near_fraction"] * 100) == round(count / 699 * 100)
        assert abs(stats["near_rho"] - rho) < 5e-4, (condition, stats["near_rho"], rho)
        if meds is not None:
            for rating_value, expected in zip((8, 9, 10), meds):
                got = row["median_by_original"][str(rating_value)]
                assert abs(got - expected) < 5e-4, (condition, rating_value, got, expected)
        checked.append(condition)
    return checked


def fmt_pct(value: float) -> str:
    return f"{100 * value:.0f}"


def fmt_rho(value: float) -> str:
    # Plain hyphen-minus, matching generated/hidden_order_numbers.tex.
    return f"{value:.3f}"


def fmt_med(value: float) -> str:
    return f"{value:.2f}"


def write_macros(path: Path, results: dict[str, Any], boot: dict[str, Any], replicates: int) -> list[str]:
    primary = results[PRIMARY_SCENARIO]
    lines = [
        "% Generated by scripts/research/analyze_ceiling_compression.py; do not edit.",
        "% Cap 8, capped group = charts originally rated 8--10; near cap = |output - 8| <= 0.05.",
        "% CIs: 95% paired title-cluster bootstrap. Pct = percent of the 699 capped charts;",
        "% Above = output > 8.05. Near-cap rho omitted when fewer than 30 near-cap charts.",
    ]
    names: list[str] = []

    def add(name: str, value: str) -> None:
        names.append(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    add("CeilDelta", f"{PRIMARY_DELTA:.2f}")
    for condition in MACRO_CONDITIONS:
        tag = MACRO_TAGS[condition]
        stats = primary[condition]["by_delta"][f"{PRIMARY_DELTA:.2f}"]
        b = boot[condition]
        add(f"Ceil{tag}NearPct", fmt_pct(stats["near_fraction"]))
        add(f"Ceil{tag}NearPctLow", fmt_pct(b["near_fraction"]["ci_low"]))
        add(f"Ceil{tag}NearPctHigh", fmt_pct(b["near_fraction"]["ci_high"]))
        add(f"Ceil{tag}NearCount", str(stats["near_count"]))
        add(f"Ceil{tag}AbovePct", fmt_pct(stats["above_fraction"]))
        if (
            stats["near_rho"] is not None
            and stats["near_count"] >= MIN_NEAR_FOR_RHO_MACRO
            and b["near_rho"]["valid_replicates"] == replicates
        ):
            add(f"Ceil{tag}NearRho", fmt_rho(stats["near_rho"]))
            add(f"Ceil{tag}NearRhoLow", fmt_rho(b["near_rho"]["ci_low"]))
            add(f"Ceil{tag}NearRhoHigh", fmt_rho(b["near_rho"]["ci_high"]))
    for condition in MEDIAN_MACRO_CONDITIONS:
        tag = MACRO_TAGS[condition]
        for rating_value, word in RATING_WORDS.items():
            add(f"Ceil{tag}Median{word}", fmt_med(primary[condition]["median_by_original"][str(rating_value)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return names


def make_figure(path: Path, data: dict[str, dict[str, np.ndarray]], cap: int) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "ceiling-compression",
        }
    )
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    left, right, bottom, top = 0.40, 0.05, 0.37, 0.05
    ax = fig.add_axes(
        (left / FIG_WIDTH, bottom / FIG_HEIGHT, 1 - (left + right) / FIG_WIDTH, 1 - (bottom + top) / FIG_HEIGHT)
    )
    ratings = list(range(cap, 11))
    offsets = (-0.22, 0.0, 0.22)
    plotted: dict[str, Any] = {}
    ax.axhline(cap, color=GUIDE, linewidth=0.6, linestyle=(0, (3.0, 2.0)), zorder=0)
    upper = cap
    for (condition, label, color, marker), offset in zip(FIG_MODELS, offsets):
        arrays = data[condition]
        rows = []
        for rating_value in ratings:
            values = arrays["rating"][arrays["original"] == rating_value]
            q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
            x = rating_value + offset
            ax.plot([x, x], [q10, q90], color=color, linewidth=0.6, solid_capstyle="butt", zorder=2)
            ax.plot([x, x], [q25, q75], color=color, linewidth=2.4, solid_capstyle="butt", zorder=3)
            ax.plot([x], [q50], marker=marker, markersize=3.6, color=color, markeredgewidth=0.6,
                    markeredgecolor="white", linestyle="none", zorder=4)
            upper = max(upper, q90)
            rows.append({"original": rating_value, "n": int(values.size), "q10": q10, "q25": q25,
                         "median": q50, "q75": q75, "q90": q90})
        plotted[condition] = rows
    ax.set_xlim(cap - 0.5, 10.5)
    ax.set_xticks(ratings)
    ax.set_xticklabels([str(r) for r in ratings])
    y_top = upper + 0.75
    ax.set_ylim(cap - 1.0, y_top)
    ax.set_yticks(list(range(cap, int(math.floor(y_top)) + 1, 2)))
    ax.set_xlabel("Original rating", labelpad=1.0)
    ax.set_ylabel("Cap-8 score", labelpad=2.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([0], [0], color=color, marker=marker, markersize=3.6, linewidth=2.4,
               markeredgewidth=0.6, markeredgecolor="white", label=label)
        for _, label, color, marker in FIG_MODELS
    ]
    ax.legend(handles=handles, loc="upper left", ncol=1, frameon=False, handlelength=1.2,
              handletextpad=0.4, labelspacing=0.15, borderaxespad=0.15, borderpad=0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)
    return {"width_in": FIG_WIDTH, "height_in": FIG_HEIGHT, "y_limits": [cap - 1.0, float(y_top)],
            "encoding": "median marker, thick bar = interquartile range, thin bar = 10th-90th percentile",
            "groups": plotted}


def main() -> None:
    args = parse_args()
    index, titles = load_panel(args)
    results: dict[str, Any] = {}
    primary_data: dict[str, dict[str, np.ndarray]] = {}
    primary_keys: list[str] = []
    for scenario, cap in CAPS.items():
        results[scenario] = {}
        for condition in CONDITIONS:
            if (scenario, condition) not in index:
                continue
            rows = index[(scenario, condition)]
            keys = sorted(k for k, r in rows.items() if float(r["evaluation_original_label"]) >= cap)
            arrays = arrays_for(index, scenario, condition, keys)
            entry = {
                "label": LABELS[condition],
                "target_kind": TARGET_KIND[condition],
                "rating_scale_field": RATING_FIELD[condition],
                "rating_scale_note": SCALE_NOTE[RATING_FIELD[condition]],
                "rank_field": "latent_score",
                "seeds": rows[keys[0]].get("seeds"),
                "capped_count": len(keys),
                "title_clusters": len({titles[k] for k in keys}),
                "capped_group_spearman": spearman(arrays["original"], arrays["rank"]),
                "rank_tie_pair_fraction": tie_fraction(arrays["rank"]),
                "by_delta": {
                    f"{delta:.2f}": point_stats(arrays["original"], arrays["rank"], arrays["rating"], cap, delta)
                    for delta in DELTAS
                },
                "median_by_original": medians(arrays["original"], arrays["rating"], cap),
                "rating_output_max": float(arrays["rating"].max()),
            }
            if RATING_FIELD[condition] != "latent_score":
                entry["rating_vs_latent_spearman"] = spearman(arrays["rank"], arrays["rating"])
            results[scenario][condition] = entry
            if scenario == PRIMARY_SCENARIO:
                if primary_keys and keys != primary_keys:
                    raise ValueError("cap-8 capped group differs across conditions")
                primary_keys = keys
                primary_data[condition] = arrays

    frozen_checked = check_against_metrics(results, args)
    lead_checked = check_lead(results)

    cluster_members: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(primary_keys):
        cluster_members[titles[key]].append(i)
    clusters = [np.asarray(v, dtype=int) for _, v in sorted(cluster_members.items())]
    boot = bootstrap_primary(primary_data, clusters, CAPS[PRIMARY_SCENARIO], PRIMARY_DELTA, args.replicates, args.seed)

    macro_names = write_macros(args.output_macros, results, boot, args.replicates)
    figure = make_figure(args.output_figure, primary_data, CAPS[PRIMARY_SCENARIO])

    payload = {
        "schema_version": "ceiling_compression_v1",
        "description": "Placement of capped-group outputs relative to the merged rating c; "
        "diagnostic of why exact-target models rank the capped group poorly.",
        "definitions": {
            "capped_group": "charts whose original rating is >= c (the paper's hidden-tail panel)",
            "near_fraction": "share of capped-group rating-scale outputs with |output - c| <= delta",
            "near_rho": "Spearman correlation between original rating and latent_score among near-cap charts "
            f"(undefined when fewer than {MIN_NEAR_FOR_RHO} charts)",
            "median_by_original": "median rating-scale output of capped charts by original rating",
            "above_fraction": "share of capped-group rating-scale outputs with output > c + delta",
            "near_rho_macro_rule": f"near_rho macros only when near_count >= {MIN_NEAR_FOR_RHO_MACRO} "
            "and every bootstrap replicate defines it",
            "rank_tie_pair_fraction": "share of capped-group chart pairs with exactly equal latent_score",
            "primary": {"scenario": PRIMARY_SCENARIO, "delta": PRIMARY_DELTA},
            "sequence_scores": "Transformer cap-8 scores are the frozen three-seed ensemble in the public artifact; "
            "caps 7/9 are single seed",
        },
        "inputs": {
            repo_path(p): sha256(p)
            for p in (args.global_scores, args.sequence_scores, args.fold_assignments,
                      args.global_metrics, args.sequence_metrics)
        },
        "bootstrap": {
            "method": "paired_title_cluster_resampling_within_capped_group",
            "replicates": args.replicates,
            "seed": args.seed,
            "confidence_level": 0.95,
            "scenario": PRIMARY_SCENARIO,
            "delta": PRIMARY_DELTA,
            "title_clusters": len(clusters),
            "intervals": boot,
        },
        "consistency_checks": {
            "capped_group_spearman_matches_frozen_metrics": frozen_checked,
            "matches_lead_preliminary_cap8_delta005": lead_checked,
        },
        "results": results,
        "figure": {"path": repo_path(args.output_figure), **figure},
        "macros": {"path": repo_path(args.output_macros), "names": macro_names},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    print(f"frozen spearman checks: {len(frozen_checked)}; lead checks: {lead_checked}")


if __name__ == "__main__":
    main()
