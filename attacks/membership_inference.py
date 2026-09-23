from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.ncf import NCF


def _sample_loss(model: NCF, user_ids: torch.Tensor, item_ids: torch.Tensor, ratings: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return per-sample squared error, which is the core statistic for the Yeom attack."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(user_ids.to(device), item_ids.to(device)).cpu()
        errors = (preds - ratings.cpu()) ** 2
    return errors.numpy()


def loss_threshold_attack(
    model: NCF,
    member_df: pd.DataFrame,
    nonmember_df: pd.DataFrame,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Compute the Yeom et al. attack AUC using negative loss as the score."""
    if member_df.empty or nonmember_df.empty:
        return 0.5
    member_user = torch.as_tensor(member_df["userId"].to_numpy(), dtype=torch.long)
    member_item = torch.as_tensor(member_df["movieId"].to_numpy(), dtype=torch.long)
    member_rating = torch.as_tensor(member_df["rating"].to_numpy(), dtype=torch.float32)
    non_user = torch.as_tensor(nonmember_df["userId"].to_numpy(), dtype=torch.long)
    non_item = torch.as_tensor(nonmember_df["movieId"].to_numpy(), dtype=torch.long)
    non_rating = torch.as_tensor(nonmember_df["rating"].to_numpy(), dtype=torch.float32)

    member_loss = _sample_loss(model, member_user, member_item, member_rating, device)
    nonmember_loss = _sample_loss(model, non_user, non_item, non_rating, device)
    scores = np.concatenate([-member_loss, -nonmember_loss])
    labels = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    return float(roc_auc_score(labels, scores))


def _train_shadow_model(
    data_df: pd.DataFrame,
    *,
    embedding_dim: int,
    mlp_layers: tuple[int, ...],
    dropout: float,
    epochs: int,
    lr: float,
    device: torch.device | str = "cpu",
) -> NCF:
    num_users = int(data_df["userId"].max()) + 1
    num_items = int(data_df["movieId"].max()) + 1
    model = NCF(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim, mlp_layers=mlp_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for _ in range(max(1, int(epochs))):
        permutation = torch.randperm(len(data_df))
        for start in range(0, len(data_df), 64):
            batch_idx = permutation[start : start + 64]
            batch = data_df.iloc[batch_idx.cpu().numpy()]
            users = torch.as_tensor(batch["userId"].to_numpy(), dtype=torch.long, device=device)
            items = torch.as_tensor(batch["movieId"].to_numpy(), dtype=torch.long, device=device)
            ratings = torch.as_tensor(batch["rating"].to_numpy(), dtype=torch.float32, device=device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss = torch.nn.functional.mse_loss(preds, ratings)
            loss.backward()
            optimizer.step()
    return model


def shadow_model_attack(
    target_model: NCF,
    full_df: pd.DataFrame,
    *,
    shadow_count: int = 2,
    shadow_epochs: int = 2,
    embedding_dim: int = 8,
    mlp_layers: tuple[int, ...] = (16, 8),
    dropout: float = 0.2,
    lr: float = 0.01,
    device: torch.device | str = "cpu",
) -> float:
    """Train shadow models and fit a logistic-regression attack classifier on their losses.

    A clean member/non-member split is needed for ROC-AUC to be well-defined; otherwise the
    metric is undefined when only one class appears in the target sample.
    """
    if full_df.empty or len(full_df) < 2:
        return 0.5

    if len(full_df) < 4:
        member_df = full_df.iloc[: max(1, len(full_df) // 2)].copy()
        nonmember_df = full_df.iloc[max(1, len(full_df) // 2) :].copy()
    else:
        split = len(full_df) // 2
        member_df = full_df.iloc[:split].copy()
        nonmember_df = full_df.iloc[split:].copy()
    if member_df.empty or nonmember_df.empty:
        return 0.5

    shadow_features: list[np.ndarray] = []
    shadow_labels: list[int] = []
    for _ in range(max(1, int(shadow_count))):
        sample = member_df.sample(frac=0.7, replace=False, random_state=np.random.randint(0, 2**31 - 1))
        other = nonmember_df.sample(frac=0.7, replace=False, random_state=np.random.randint(0, 2**31 - 1))
        shadow_model = _train_shadow_model(
            pd.concat([sample, other], ignore_index=True),
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers,
            dropout=dropout,
            epochs=shadow_epochs,
            lr=lr,
            device=device,
        )
        for subset, label in [(sample, 1), (other, 0)]:
            users = torch.as_tensor(subset["userId"].to_numpy(), dtype=torch.long)
            items = torch.as_tensor(subset["movieId"].to_numpy(), dtype=torch.long)
            ratings = torch.as_tensor(subset["rating"].to_numpy(), dtype=torch.float32)
            with torch.no_grad():
                preds = shadow_model(users.to(device), items.to(device)).cpu()
            errs = (preds - ratings) ** 2
            abs_pred = preds.abs()
            scores = np.column_stack([errs.numpy(), abs_pred.numpy(), preds.numpy()])
            shadow_features.extend(scores)
            shadow_labels.extend([label] * len(subset))

    if len(set(shadow_labels)) < 2:
        return 0.5

    attack = LogisticRegression(max_iter=200, solver="liblinear")
    attack.fit(np.asarray(shadow_features), np.asarray(shadow_labels))

    target_users_m = torch.as_tensor(member_df["userId"].to_numpy(), dtype=torch.long)
    target_items_m = torch.as_tensor(member_df["movieId"].to_numpy(), dtype=torch.long)
    target_ratings_m = torch.as_tensor(member_df["rating"].to_numpy(), dtype=torch.float32)
    target_users_n = torch.as_tensor(nonmember_df["userId"].to_numpy(), dtype=torch.long)
    target_items_n = torch.as_tensor(nonmember_df["movieId"].to_numpy(), dtype=torch.long)
    target_ratings_n = torch.as_tensor(nonmember_df["rating"].to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        target_preds_m = target_model(target_users_m.to(device), target_items_m.to(device)).cpu()
        target_preds_n = target_model(target_users_n.to(device), target_items_n.to(device)).cpu()
    target_err = np.concatenate([(target_preds_m - target_ratings_m).pow(2).numpy(), (target_preds_n - target_ratings_n).pow(2).numpy()])
    target_abs = np.concatenate([target_preds_m.abs().numpy(), target_preds_n.abs().numpy()])
    target_pred = np.concatenate([target_preds_m.numpy(), target_preds_n.numpy()])
    target_features = np.column_stack([target_err, target_abs, target_pred])
    labels = np.concatenate([np.ones(len(member_df)), np.zeros(len(nonmember_df))])
    scores = attack.predict_proba(target_features)[:, 1]
    return float(np.clip(roc_auc_score(labels, scores), 0.0, 1.0))
