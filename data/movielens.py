from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """Container for one train/validation/test split of the recommendation data.

    The bundle is intentionally simple: all clients see only their local training data,
    while the server keeps the global model. This matches the practical FL setup in
    McMahan et al. (2017) and the chronological split required for recommendation.
    """

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: dict[str, Any]


def _iterative_filter(df: pd.DataFrame, min_user_ratings: int, min_item_ratings: int) -> pd.DataFrame:
    """Filter sparse users/items until the matrix becomes stable."""
    if df.empty:
        return df
    for _ in range(20):
        user_counts = df["userId"].value_counts()
        item_counts = df["movieId"].value_counts()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        keep_items = item_counts[item_counts >= min_item_ratings].index
        filtered = df[df["userId"].isin(keep_users) & df["movieId"].isin(keep_items)].copy()
        if len(filtered) == len(df):
            return filtered
        df = filtered
    return df


def _normalize_ids(series: pd.Series) -> pd.Series:
    """Coerce raw IDs to a stable string form before remapping them to contiguous ints."""
    return series.map(lambda value: str(value).strip() if pd.notna(value) and str(value).strip() else None)


def _remap_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Remap user and item IDs to contiguous integer indices."""
    user_values = sorted(df["userId"].dropna().astype(str).unique())
    item_values = sorted(df["movieId"].dropna().astype(str).unique())
    user_map = {old: idx for idx, old in enumerate(user_values)}
    item_map = {old: idx for idx, old in enumerate(item_values)}
    out = df.copy()
    out["userId"] = out["userId"].map(user_map)
    out["movieId"] = out["movieId"].map(item_map)
    return out


def _syn_data(num_users: int, num_items: int, seed: int, popularity_alpha: float, cluster_count: int, user_cluster_pref: int) -> pd.DataFrame:
    """Create synthetic MovieLens-like data with clustered preferences and Zipf popularity.

    This is intentionally designed as a controlled fallback so the project can run in CI
    or on a machine without the 1GB MovieLens download. The data follows the same schema
    as real ratings.csv files and exhibits realistic non-IID structure.
    """
    rng = np.random.default_rng(seed)
    item_popularity = np.power(np.arange(1, num_items + 1), -popularity_alpha)
    item_popularity = item_popularity / item_popularity.sum()
    cluster_assignments = np.arange(num_items) % cluster_count

    rows: list[dict[str, float | int]] = []
    for user_id in range(num_users):
        clusters = rng.choice(cluster_count, size=user_cluster_pref, replace=False)
        preference_prob = rng.uniform(0.7, 1.4, size=num_items)
        preferred_mask = np.isin(cluster_assignments, clusters)
        counts = rng.poisson(lam=30) + 10
        for _ in range(max(1, counts)):
            item_id = int(rng.choice(num_items, p=item_popularity))
            is_preferred = bool(preferred_mask[item_id])
            if rng.random() < 0.85 and is_preferred:
                score = 0.4
            elif rng.random() < 0.15:
                score = 0.2
            else:
                score = -0.2
            rating = float(np.clip(3.5 + score + 0.5 * rng.normal() + 0.8 * (1 if is_preferred else 0), 0.5, 5.0))
            timestamp = int(rng.integers(1_600_000_000, 1_700_000_000))
            rows.append({"userId": user_id, "movieId": item_id, "rating": rating, "timestamp": timestamp})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Synthetic fallback produced no ratings.")
    return df


def _split_by_user(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological per-user split with a leave-one-out future-style evaluation."""
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for _, user_df in df.groupby("userId", sort=True):
        user_df = user_df.sort_values("timestamp").reset_index(drop=True)
        n = len(user_df)
        if n == 1:
            train_rows.append(user_df.iloc[0].to_dict())
            continue
        train_end = max(1, int(n * train_frac))
        val_end = min(n - 1, max(train_end + 1, int(n * (train_frac + val_frac))))

        train_rows.extend(user_df.iloc[:train_end].to_dict("records"))
        if val_end > train_end:
            val_rows.extend(user_df.iloc[train_end:val_end].to_dict("records"))
        if n > val_end:
            test_rows.extend(user_df.iloc[val_end:].to_dict("records"))

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, val_df, test_df


def _split_random_by_user(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible per-user holdouts when source data has no timestamps."""
    rng = np.random.default_rng(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for _, user_df in df.groupby("userId", sort=True):
        order = rng.permutation(len(user_df))
        shuffled = user_df.iloc[order]
        n = len(shuffled)
        if n < 3:
            train_rows.extend(shuffled.to_dict("records"))
            continue
        val_count = max(1, int(round(n * val_frac)))
        test_count = max(1, int(round(n * test_frac)))
        while val_count + test_count >= n:
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        train_count = n - val_count - test_count
        train_rows.extend(shuffled.iloc[:train_count].to_dict("records"))
        val_rows.extend(shuffled.iloc[train_count : train_count + val_count].to_dict("records"))
        test_rows.extend(shuffled.iloc[train_count + val_count :].to_dict("records"))

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def _add_training_negatives(
    train_df: pd.DataFrame,
    interactions: pd.DataFrame,
    n_items: int,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    """Sample unobserved user-item pairs as zero-label implicit-feedback examples."""
    if train_df.empty or negative_ratio <= 0:
        return train_df
    rng = np.random.default_rng(seed)
    observed = interactions.groupby("userId")["movieId"].agg(set).to_dict()
    negative_rows: list[dict[str, int | float]] = []
    item_ids = np.arange(n_items)
    for user_id, user_train in train_df.groupby("userId", sort=True):
        available = np.setdiff1d(item_ids, np.fromiter(observed.get(user_id, set()), dtype=np.int64))
        sample_count = min(len(available), len(user_train) * negative_ratio)
        if sample_count == 0:
            continue
        sampled = rng.choice(available, size=sample_count, replace=False)
        negative_rows.extend(
            {"userId": int(user_id), "movieId": int(item_id), "rating": 0.0}
            for item_id in sampled
        )
    if not negative_rows:
        return train_df
    return pd.concat([train_df, pd.DataFrame(negative_rows)], ignore_index=True)


def load_movielens_data(config: dict[str, Any]) -> DatasetBundle:
    """Load a ratings table from disk or synthesize a fallback if it does not exist.

    The implementation follows the data preprocessing pipeline typically used in
    recommendation papers: filter sparse entities, remap IDs to contiguous integers,
    and perform a user-wise chronological split to avoid future leakage in evaluation.
    """
    dataset_cfg = config.get("dataset", {})
    path = dataset_cfg.get("path")
    min_user_ratings = int(dataset_cfg.get("min_user_ratings", 5))
    min_item_ratings = int(dataset_cfg.get("min_item_ratings", 5))
    split_cfg = dataset_cfg.get("split_fractions", {"train": 0.7, "val": 0.15, "test": 0.15})
    split_strategy = str(dataset_cfg.get("split_strategy", "chronological"))
    implicit_feedback = bool(dataset_cfg.get("implicit_feedback", False))

    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        synth_cfg = dataset_cfg.get("synthetic", {})
        df = _syn_data(
            num_users=int(synth_cfg.get("num_users", 100)),
            num_items=int(synth_cfg.get("num_items", 150)),
            seed=int(synth_cfg.get("seed", 0)),
            popularity_alpha=float(synth_cfg.get("popularity_alpha", 1.0)),
            cluster_count=int(synth_cfg.get("cluster_count", 10)),
            user_cluster_pref=int(synth_cfg.get("user_cluster_pref", 2)),
        )

    required = {"userId", "movieId", "rating"}
    if split_strategy == "chronological":
        required.add("timestamp")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ratings data is missing columns: {sorted(missing)}")

    selected_columns = ["userId", "movieId", "rating"]
    if "timestamp" in df.columns:
        selected_columns.append("timestamp")
    df = df.loc[:, selected_columns].copy()
    df["userId"] = _normalize_ids(df["userId"])
    df["movieId"] = _normalize_ids(df["movieId"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    drop_columns = ["userId", "movieId", "rating"]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if split_strategy == "chronological":
            drop_columns.append("timestamp")
    df = df.dropna(subset=drop_columns).reset_index(drop=True)
    df = _iterative_filter(df, min_user_ratings, min_item_ratings)
    df = _remap_ids(df)

    train_frac = float(split_cfg.get("train", 0.7))
    val_frac = float(split_cfg.get("val", 0.15))
    test_frac = float(split_cfg.get("test", 0.15))
    if split_strategy == "random":
        train_df, val_df, test_df = _split_random_by_user(
            df,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=int(dataset_cfg.get("split_seed", 0)),
        )
    elif split_strategy == "chronological":
        train_df, val_df, test_df = _split_by_user(df, train_frac, val_frac, test_frac)
    else:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    if implicit_feedback:
        train_df = _add_training_negatives(
            train_df,
            interactions=df,
            n_items=int(df["movieId"].nunique()),
            negative_ratio=int(dataset_cfg.get("negative_ratio", 1)),
            seed=int(dataset_cfg.get("split_seed", 0)),
        )

    stats = {
        "n_users": int(df["userId"].nunique()),
        "n_items": int(df["movieId"].nunique()),
        "n_ratings": int(len(df)),
        "density": float(len(df) / (df["userId"].nunique() * df["movieId"].nunique())),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "rating_mean": float(df["rating"].mean()),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }
    return DatasetBundle(train_df=train_df, val_df=val_df, test_df=test_df, stats=stats)
