from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class NCF(nn.Module):
    """Neural Collaborative Filtering with a GMF branch and an MLP branch.

    The model follows the standard design of He et al. (2017): separate embeddings for
    general matrix factorization and a learned MLP interaction network, then a final
    fusion layer to predict a scalar rating. The output is clipped only at inference time,
    not during training, in line with the paper's requirement to keep the optimization
    objective unconstrained while preserving a valid rating interval for evaluation.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 16,
        mlp_layers: Sequence[int] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)

        self.gmf_user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.gmf_item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.mlp_user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.mlp_item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        layers: list[nn.Module] = []
        input_dim = 2 * self.embedding_dim
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim
        self.mlp_layers_module = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.embedding_dim + mlp_layers[-1], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_ids = user_ids.to(dtype=torch.long)
        item_ids = item_ids.to(dtype=torch.long)

        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_out = gmf_user * gmf_item

        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp_layers_module(mlp_input)

        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output_layer(fused).squeeze(-1)

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out.clamp(0.5, 5.0)
