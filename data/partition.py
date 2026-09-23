from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a non-negative distribution."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n)


def entropy_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy of a categorical distribution."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def compute_partition_stats(df: pd.DataFrame, n_users: int | None = None, n_items: int | None = None) -> dict[str, Any]:
    """Aggregate client-level statistics used in the dataset section of the paper.

    These metrics quantify the cross-device non-IID skew in recommender datasets and
    directly motivate the use of both secure aggregation and DP in FL.
    """
    if df.empty:
        return {
            "sparsity_pct": 0.0,
            "ratings_per_user_mean": 0.0,
            "ratings_per_user_median": 0.0,
            "ratings_per_user_std": 0.0,
            "gini_ratings_per_user": 0.0,
            "genre_entropy_mean": 0.0,
            "num_users": 0,
            "num_items": 0,
        }

    user_counts = df["userId"].value_counts().sort_index()
    ratings_per_user = user_counts.to_numpy(dtype=float)
    n_users = int(n_users if n_users is not None else df["userId"].nunique())
    n_items = int(n_items if n_items is not None else df["movieId"].nunique())

    sparsity = 1.0 - (len(df) / max(1, n_users * n_items))
    genre_entropy = 0.0
    if "genre" in df.columns:
        grouped = df.groupby("userId")["genre"].apply(lambda g: g.value_counts(normalize=True).to_numpy())
        genre_entropy = float(np.mean([entropy_from_counts(np.asarray(v, dtype=float)) for v in grouped]))

    return {
        "sparsity_pct": float(sparsity * 100.0),
        "ratings_per_user_mean": float(ratings_per_user.mean()),
        "ratings_per_user_median": float(np.median(ratings_per_user)),
        "ratings_per_user_std": float(ratings_per_user.std(ddof=0)),
        "gini_ratings_per_user": float(gini_coefficient(ratings_per_user)),
        "genre_entropy_mean": float(genre_entropy),
        "num_users": n_users,
        "num_items": n_items,
    }


def build_client_train_data(train_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return per-client training data keyed by the client index."""
    return {int(user_id): group.reset_index(drop=True) for user_id, group in train_df.groupby("userId", sort=True)}
