from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import torch

from models.ncf import NCF


def train_local_client(
    model: NCF,
    client_df: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> NCF:
    """Train the client's local model on that user's data without sending data externally."""
    if client_df.empty:
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)

    rows = client_df[["userId", "movieId", "rating"]].copy().reset_index(drop=True)
    for _ in range(max(1, int(epochs))):
        indices = torch.randperm(len(rows), device=device)
        for start in range(0, len(rows), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = rows.iloc[batch_idx.cpu().numpy()]
            user_ids = torch.as_tensor(batch["userId"].to_numpy(), dtype=torch.long, device=device)
            item_ids = torch.as_tensor(batch["movieId"].to_numpy(), dtype=torch.long, device=device)
            ratings = torch.as_tensor(batch["rating"].to_numpy(), dtype=torch.float32, device=device)
            optimizer.zero_grad()
            preds = model(user_ids, item_ids)
            loss = torch.nn.functional.mse_loss(preds, ratings)
            loss.backward()
            optimizer.step()
    return model


def compute_delta(
    base_state: dict[str, torch.Tensor],
    updated_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the client-local update as the state delta before any DP or SecAgg masking."""
    return {name: updated_state[name] - base_state[name] for name in base_state.keys()}


def weighted_delta(
    update: dict[str, torch.Tensor],
    client_sample_count: int,
    total_sample_count: int,
) -> dict[str, torch.Tensor]:
    """Apply the FedAvg client weight before SecAgg. This is required because SecAgg sums raw values."""
    if total_sample_count <= 0:
        raise ValueError("total_sample_count must be positive.")
    weight = client_sample_count / total_sample_count
    return {name: tensor * weight for name, tensor in update.items()}
