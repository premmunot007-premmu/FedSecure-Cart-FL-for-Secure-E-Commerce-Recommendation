from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch


def rmse_scores(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE on a vector of true and predicted ratings."""
    if y_true.size == 0:
        return 0.0
    err = y_true - y_pred
    return float(math.sqrt(np.mean(err ** 2)))


def recall_at_k(actual: set[int], ranked: list[int], k: int) -> float:
    """Compute recall@K for one user, where the held-out item is the relevant one."""
    if not actual:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in actual)
    return float(hits / len(actual))


def ndcg_at_k(actual: set[int], ranked: list[int], k: int) -> float:
    """NDCG@K for a single user."""
    if not actual:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in actual:
            dcg += 1.0 / np.log2(rank + 1)
    ideal = 1.0
    return float(dcg / ideal)


def evaluate_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    *,
    top_k: list[int] | None = None,
    max_users: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluate the ranking quality of the model on a holdout set using a future split.

    The metric logic is intentionally compact but faithful to the standard recommendation
    evaluation setup: user-wise chronological splits and held-out candidate ranking.
    """
    if df.empty:
        return {"rmse": 0.0, "recall": {str(k): 0.0 for k in (top_k or [5])}, "ndcg": {str(k): 0.0 for k in (top_k or [5])}}

    top_k = [int(k) for k in (top_k or [5])]
    all_items = sorted(int(item) for item in df["movieId"].unique())
    users = sorted(int(u) for u in df["userId"].unique())
    if max_users is not None:
        users = users[: int(max_users)]

    truths: list[float] = []
    preds: list[float] = []
    recall_scores: dict[str, list[float]] = {str(k): [] for k in top_k}
    ndcg_scores: dict[str, list[float]] = {str(k): [] for k in top_k}

    for user_id in users:
        user_df = df[df["userId"] == user_id].sort_values("timestamp").reset_index(drop=True)
        if len(user_df) < 2:
            continue
        held_out = user_df.iloc[-1]
        train_df = user_df.iloc[:-1]
        if train_df.empty:
            continue
        candidate_items = all_items
        user_tensor = torch.full((len(candidate_items),), user_id, dtype=torch.long, device=device)
        item_tensor = torch.as_tensor(candidate_items, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model(user_tensor, item_tensor).detach().cpu().numpy()
        ranked = [int(item) for _, item in sorted(zip(scores.tolist(), candidate_items), key=lambda pair: pair[0], reverse=True)]
        actual = {int(held_out["movieId"])}
        truth = float(held_out["rating"])
        pred = float(scores[candidate_items.index(int(held_out["movieId"]))]) if int(held_out["movieId"]) in candidate_items else 0.0
        truths.append(truth)
        preds.append(pred)
        for k in top_k:
            recall_scores[str(k)].append(recall_at_k(actual, ranked, k))
            ndcg_scores[str(k)].append(ndcg_at_k(actual, ranked, k))

    rmse_val = rmse_scores(np.asarray(truths, dtype=float), np.asarray(preds, dtype=float))
    return {
        "rmse": float(rmse_val),
        "recall": {str(k): float(np.mean(recall_scores[str(k)])) for k in top_k},
        "ndcg": {str(k): float(np.mean(ndcg_scores[str(k)])) for k in top_k},
    }
