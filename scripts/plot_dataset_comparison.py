#!/usr/bin/env python3
"""Create security, recommendation-quality, and cost comparisons from two run tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFENSES = ["no_defense", "secagg_only", "dp_only", "dp_secagg"]
DEFENSE_LABELS = {
    "no_defense": "No defense",
    "secagg_only": "Secure aggregation",
    "dp_only": "Differential privacy",
    "dp_secagg": "DP + secure aggregation",
}
DEFENSE_COLORS = {
    "no_defense": "#53616f",
    "secagg_only": "#168a83",
    "dp_only": "#d98b22",
    "dp_secagg": "#bf4b47",
}
REQUIRED = {
    "defense",
    "epsilon",
    "mia_auc",
    "shadow_auc",
    "support_precision",
    "recall@5",
    "ndcg@5",
    "wall_time_sec",
    "dataset_n_users",
    "dataset_n_items",
    "dataset_n_interactions",
    "dataset_density",
}


def _load_results(path: Path, label: str, protocol: str) -> pd.DataFrame:
    results = pd.read_csv(path)
    missing = REQUIRED - set(results.columns)
    if missing:
        raise ValueError(f"{path} is missing result columns: {sorted(missing)}")
    results = results[results["defense"].isin(DEFENSES)].copy()
    if results.empty:
        raise ValueError(f"No comparable defense rows found in {path}.")
    results["dataset"] = label
    results["evaluation_protocol"] = protocol
    return results


def _x_values(group: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(group["epsilon"], errors="coerce").fillna(0.0)


def _plot_lines(axis: Any, results: pd.DataFrame, metric: str) -> None:
    for defense in DEFENSES:
        group = results[results["defense"] == defense].copy()
        if group.empty:
            continue
        group["plot_epsilon"] = _x_values(group)
        group = group.sort_values("plot_epsilon")
        axis.plot(
            group["plot_epsilon"],
            group[metric],
            color=DEFENSE_COLORS[defense],
            marker="o",
            linewidth=2,
            markersize=6,
            label=DEFENSE_LABELS[defense],
        )


def _finish_axis(axis: Any, epsilons: list[float]) -> None:
    axis.set_xticks([0.0] + epsilons, ["Baseline"] + [f"{epsilon:g}" for epsilon in epsilons])
    axis.set_xlabel("Target privacy budget ε")
    axis.grid(axis="y", color="#d8dee4", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _save_security_attacks(results: pd.DataFrame, output_dir: Path, epsilons: list[float]) -> None:
    datasets = list(dict.fromkeys(results["dataset"].tolist()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False, sharey=True)
    for axis, dataset in zip(axes[0], datasets):
        subset = results[results["dataset"] == dataset]
        _plot_lines(axis, subset, "mia_auc")
        axis.axhline(0.5, color="#252a31", linestyle="--", linewidth=1.2, label="Chance (0.50)")
        axis.set_title(f"{dataset}\nThreshold membership attack", loc="left", fontweight="bold")
        axis.set_ylabel("Attack ROC-AUC (lower is better; 0.50 = chance)")
        all_scores = pd.concat([subset["mia_auc"], subset["shadow_auc"]]).dropna()
        low = min(0.45, float(all_scores.min()) - 0.02) if not all_scores.empty else 0.45
        high = max(0.55, float(all_scores.max()) + 0.02) if not all_scores.empty else 0.55
        axis.set_ylim(max(0.0, low), min(1.0, high))
        _finish_axis(axis, epsilons)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=False)
    fig.suptitle("Membership inference under privacy defenses", y=1.16, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_dir / "membership_leakage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False, sharey=True)
    for axis, dataset in zip(axes[0], datasets):
        subset = results[results["dataset"] == dataset]
        _plot_lines(axis, subset, "shadow_auc")
        axis.axhline(0.5, color="#252a31", linestyle="--", linewidth=1.2, label="Chance (0.50)")
        axis.set_title(f"{dataset}\nShadow-model membership attack", loc="left", fontweight="bold")
        axis.set_ylabel("Attack ROC-AUC (lower is better; 0.50 = chance)")
        all_scores = subset["shadow_auc"].dropna()
        low = min(0.45, float(all_scores.min()) - 0.02) if not all_scores.empty else 0.45
        high = max(0.55, float(all_scores.max()) + 0.02) if not all_scores.empty else 0.55
        axis.set_ylim(max(0.0, low), min(1.0, high))
        _finish_axis(axis, epsilons)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=False)
    fig.suptitle("Shadow-model attack success under privacy defenses", y=1.16, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_dir / "shadow_attack.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_recommendation_quality(results: pd.DataFrame, output_dir: Path, epsilons: list[float]) -> None:
    datasets = list(dict.fromkeys(results["dataset"].tolist()))
    fig, axes = plt.subplots(2, len(datasets), figsize=(7 * len(datasets), 8), squeeze=False)
    metric_specs = [("recall@5", "Recall@5 (higher is better)"), ("ndcg@5", "NDCG@5 (higher is better)")]
    for column, dataset in enumerate(datasets):
        subset = results[results["dataset"] == dataset]
        for row, (metric, title) in enumerate(metric_specs):
            axis = axes[row][column]
            _plot_lines(axis, subset, metric)
            axis.set_title(f"{dataset}\n{title}", loc="left", fontweight="bold")
            axis.set_ylabel(title)
            axis.set_ylim(bottom=0)
            _finish_axis(axis, epsilons)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=False)
    fig.suptitle("Recommendation quality by dataset and defense", y=1.16, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "MovieLens uses chronological explicit-rating holdouts; Amazon uses random implicit-positive holdouts. Compare trends, not absolute dataset difficulty.",
        ha="center",
        color="#53616f",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.89))
    fig.savefig(output_dir / "recommendation_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    movie_lens = results[results["dataset"].str.contains("MovieLens", case=False)]
    if not movie_lens.empty and movie_lens["rmse"].notna().any():
        fig, axis = plt.subplots(figsize=(8, 5))
        _plot_lines(axis, movie_lens, "rmse")
        axis.set_title("MovieLens rating error by defense", loc="left", fontweight="bold")
        axis.set_ylabel("Validation RMSE (lower is better)")
        axis.set_yscale("log")
        _finish_axis(axis, epsilons)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "movielens_rmse.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _save_gradient_and_runtime(results: pd.DataFrame, output_dir: Path, epsilons: list[float]) -> None:
    datasets = list(dict.fromkeys(results["dataset"].tolist()))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False, sharey=True)
    for axis, dataset in zip(axes[0], datasets):
        subset = results[results["dataset"] == dataset]
        _plot_lines(axis, subset, "support_precision")
        axis.set_title(f"{dataset}\nGradient-inversion support", loc="left", fontweight="bold")
        axis.set_ylabel("Recovered-item precision (higher is better)")
        axis.set_ylim(0, 1)
        _finish_axis(axis, epsilons)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=False)
    fig.suptitle("Resistance to item-support recovery", y=1.16, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_dir / "gradient_inversion_resistance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False, sharey=True)
    for axis, dataset in zip(axes[0], datasets):
        subset = results[results["dataset"] == dataset]
        baseline = subset.loc[subset["defense"] == "no_defense", "wall_time_sec"].median()
        subset = subset.copy()
        subset["runtime_multiple"] = subset["wall_time_sec"] / max(float(baseline), 1e-9)
        _plot_lines(axis, subset.rename(columns={"runtime_multiple": "runtime_multiple"}), "runtime_multiple")
        axis.axhline(1.0, color="#252a31", linestyle="--", linewidth=1.0)
        axis.set_title(f"{dataset}\nRelative to its no-defense runtime", loc="left", fontweight="bold")
        axis.set_ylabel("Runtime multiple (1.0 = no defense)")
        axis.set_ylim(bottom=0)
        _finish_axis(axis, epsilons)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=False)
    fig.suptitle("Training and attack evaluation overhead", y=1.16, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_dir / "relative_runtime.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_comparison(
    movielens_csv: str | Path,
    amazon_csv: str | Path,
    output_dir: str | Path,
    movielens_label: str = "MovieLens-32M sample (3k users)",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    movielens = _load_results(Path(movielens_csv), movielens_label, "chronological explicit ratings")
    amazon = _load_results(Path(amazon_csv), "Amazon Electronics", "random implicit interactions")
    combined = pd.concat([movielens, amazon], ignore_index=True)
    epsilons = sorted(float(value) for value in combined["epsilon"].dropna().unique())
    _save_security_attacks(combined, output_dir, epsilons)
    _save_recommendation_quality(combined, output_dir, epsilons)
    _save_gradient_and_runtime(combined, output_dir, epsilons)
    combined.to_csv(output_dir / "combined_results.csv", index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movielens", type=Path, required=True, help="MovieLens results.csv")
    parser.add_argument("--amazon", type=Path, required=True, help="Amazon results.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_dataset"))
    parser.add_argument("--movielens-label", default="MovieLens-32M sample (3k users)")
    args = parser.parse_args()
    combined = make_comparison(args.movielens, args.amazon, args.output_dir, args.movielens_label)
    print(f"Compared {len(combined)} runs across {combined['dataset'].nunique()} datasets. Wrote charts and combined_results.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
