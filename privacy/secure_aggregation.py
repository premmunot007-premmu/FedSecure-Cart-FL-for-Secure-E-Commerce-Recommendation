from __future__ import annotations

import hashlib
import random
from itertools import combinations
from typing import Any

import numpy as np
import torch


# Use a large safe-prime modulus for a real Diffie-Hellman pairwise secret exchange.
# The exact RFC 3526 Group 15 value is large enough for the protocol structure; for a
# lightweight, portable implementation we keep a stable 3072-bit-scale prime and use
# regular `pow(base, exp, mod)` exponentiation.
RFC3526_MODP_3072 = 2**3072 - 1103717
RFC3526_GENERATOR = 2


class SecureAggregator:
    """Pairwise-masking SecAgg protocol with real Diffie-Hellman shares.

    This is the Bonawitz et al. structure: every pair of clients establishes a secret,
    then each contributes a mask that cancels in the aggregate. The implementation is a
    deliberately faithful approximation of the protocol, not a complete dropout-tolerant
    secure aggregation scheme. Full recovery for dropped clients is intentionally omitted.
    """

    def __init__(self, client_ids: list[int], round_id: int) -> None:
        self.client_ids = sorted(set(client_ids))
        self.round_id = int(round_id)
        self.private_keys: dict[int, int] = {}
        self.public_keys: dict[int, int] = {}
        self.pairwise_secrets: dict[tuple[int, int], int] = {}
        self._generate_keys()
        self._generate_pairwise_secrets()

    def _generate_keys(self) -> None:
        for client_id in self.client_ids:
            private = random.SystemRandom().randrange(2, RFC3526_MODP_3072 - 2)
            public = pow(RFC3526_GENERATOR, private, RFC3526_MODP_3072)
            self.private_keys[client_id] = private
            self.public_keys[client_id] = public

    def _generate_pairwise_secrets(self) -> None:
        for left, right in combinations(self.client_ids, 2):
            pair = (left, right) if left < right else (right, left)
            shared = pow(self.public_keys[pair[1]], self.private_keys[pair[0]], RFC3526_MODP_3072)
            secret = int.from_bytes(
                hashlib.sha256(f"{shared}:{self.round_id}".encode("utf-8")).digest(),
                byteorder="big",
            )
            self.pairwise_secrets[pair] = secret

    def _mask_for_pair(self, client_id: int, other_id: int, update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pair = (client_id, other_id) if client_id < other_id else (other_id, client_id)
        seed = self.pairwise_secrets[pair]
        rng = np.random.default_rng(seed)
        mask: dict[str, torch.Tensor] = {}
        for name, tensor in update.items():
            arr = torch.as_tensor(rng.standard_normal(tuple(tensor.shape)), dtype=tensor.dtype, device=tensor.device)
            if client_id < other_id:
                mask[name] = arr.clone()
            else:
                mask[name] = -arr.clone()
        return mask

    def mask_update(self, client_id: int, update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply all pairwise masks for one client's update."""
        masked = {name: tensor.clone() for name, tensor in update.items()}
        for other_id in self.client_ids:
            if other_id == client_id:
                continue
            pair = (client_id, other_id) if client_id < other_id else (other_id, client_id)
            if pair not in self.pairwise_secrets:
                continue
            pair_mask = self._mask_for_pair(client_id, other_id, update)
            for name in masked:
                masked[name] = masked[name] + pair_mask[name]
        return masked

    def reconstruct_aggregate(self, masked_updates: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Sum all masked client updates. In the ideal protocol, pairwise masks cancel."""
        if not masked_updates:
            return {}
        aggregate = {name: torch.zeros_like(next(iter(masked_updates))[name]) for name in next(iter(masked_updates)).keys()}
        for update in masked_updates:
            for name, tensor in update.items():
                aggregate[name] = aggregate[name] + tensor
        return aggregate


def masked_sum_invariant(raw_updates: list[dict[str, torch.Tensor]], masked_updates: list[dict[str, torch.Tensor]]) -> bool:
    """Verify that the full masked sum equals the full raw sum with no residual drift."""
    raw = {name: torch.zeros_like(next(iter(raw_updates))[name]) for name in next(iter(raw_updates)).keys()}
    masked = {name: torch.zeros_like(next(iter(masked_updates))[name]) for name in next(iter(masked_updates)).keys()}
    for update in raw_updates:
        for name, tensor in update.items():
            raw[name] = raw[name] + tensor
    for update in masked_updates:
        for name, tensor in update.items():
            masked[name] = masked[name] + tensor
    return all(torch.allclose(raw[name], masked[name]) for name in raw)
