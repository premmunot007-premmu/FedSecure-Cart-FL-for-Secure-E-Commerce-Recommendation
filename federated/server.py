from __future__ import annotations

import copy

import torch


class FedAvgServer:
    """Aggregate client deltas with the standard FedAvg weighting used in cross-device FL."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def update_from_aggregate(self, base_state: dict[str, torch.Tensor], aggregate: dict[str, torch.Tensor]) -> None:
        new_state = {name: base_state[name] + aggregate.get(name, torch.zeros_like(base_state[name])) for name in base_state}
        self.model.load_state_dict(new_state)

    def aggregate(self, base_state: dict[str, torch.Tensor], deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        aggregate = {}
        for name in base_state:
            aggregate[name] = torch.zeros_like(base_state[name])
            for delta in deltas:
                if name in delta:
                    aggregate[name] = aggregate[name] + delta[name]
        return aggregate
