from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.movielens import load_movielens_data
from data.partition import compute_partition_stats


def main() -> None:
    config_path = ROOT / "configs" / "smoke_test.yaml"
    cfg = __import__("yaml").safe_load(open(config_path, "r", encoding="utf-8"))
    data = load_movielens_data(cfg)
    stats = compute_partition_stats(data.train_df, n_users=data.stats["n_users"], n_items=data.stats["n_items"])
    print(stats)

    _ = data.train_df["userId"].value_counts()
    hist_path = ROOT / "outputs" / "stats"
    hist_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data.train_df["rating"], bins=10)
    axes[0].set_title("Rating distribution")
    axes[1].hist(data.train_df["userId"].value_counts().to_numpy(), bins=20)
    axes[1].set_title("Ratings per user")
    fig.tight_layout()
    fig.savefig(hist_path / "ratings_stats.png", dpi=150)


if __name__ == "__main__":
    main()
