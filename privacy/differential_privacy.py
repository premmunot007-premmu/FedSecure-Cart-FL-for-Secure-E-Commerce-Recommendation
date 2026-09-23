from __future__ import annotations

import math
from typing import Any

import torch

if not hasattr(torch.nn, "RMSNorm"):
    class RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
            super().__init__()
            self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
            self.eps = float(eps)
            self.elementwise_affine = bool(elementwise_affine)
            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))
                self.bias = torch.nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
            rms = x.pow(2).mean(dim=dims, keepdim=True).add(self.eps).sqrt()
            out = x / rms
            if self.weight is not None:
                out = out * self.weight
            if self.bias is not None:
                out = out + self.bias
            return out

    torch.nn.RMSNorm = RMSNorm

from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def clip_update(update: dict[str, torch.Tensor], clip_norm: float) -> dict[str, torch.Tensor]:
    """Clip a client update to L2 norm C using the standard DP-FedAvg update-level bound.

    This is not per-example DP-SGD. Instead, the whole local model delta is treated as a
    single vectorized client update, which is exactly the setting described in the paper's
    FedAvg + DP formulation.
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")
    total_norm_sq = 0.0
    for tensor in update.values():
        total_norm_sq += float(torch.sum(tensor.detach().float() ** 2))
    total_norm = math.sqrt(total_norm_sq)
    if total_norm <= clip_norm:
        return {name: tensor.clone() for name, tensor in update.items()}
    scale = clip_norm / (total_norm + 1e-12)
    return {name: tensor * scale for name, tensor in update.items()}


def add_gaussian_noise(update: dict[str, torch.Tensor], clip_norm: float, noise_multiplier: float) -> dict[str, torch.Tensor]:
    """Add Gaussian noise calibrated to the update L2 sensitivity C."""
    return {
        name: tensor + torch.randn_like(tensor) * (noise_multiplier * clip_norm)
        for name, tensor in update.items()
    }


def apply_dp_to_update(
    update: dict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
) -> dict[str, torch.Tensor]:
    """Apply Layer 1: clipping followed by Gaussian noise."""
    clipped = clip_update(update, clip_norm)
    return add_gaussian_noise(clipped, clip_norm, noise_multiplier)


def relax_with_secagg(noise_multiplier: float, clients_per_round: int) -> float:
    """Apply the standard SecAgg relaxation used in the paper's empirical study.

    When K selected clients participate in a round, the aggregate noise variance is the
    sum of K independent client noises. The protocol can therefore reduce each client's
    per-round noise by sqrt(K) while preserving the same aggregate variance.
    """
    if clients_per_round <= 0:
        raise ValueError("clients_per_round must be positive.")
    return float(noise_multiplier / math.sqrt(clients_per_round))


def get_rdp_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    accountant: str = "rdp",
) -> float:
    """Solve for sigma using the standalone Opacus RDP accountant as requested."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant=accountant,
        )
    )


def make_accountant() -> RDPAccountant:
    """Create the standalone RDP accountant needed for update-level accounting."""
    return RDPAccountant()


def step_accountant(
    accountant: RDPAccountant,
    noise_multiplier: float,
    sample_rate: float,
    *,
    steps: int = 1,
) -> None:
    """Record one FL round as one accounting step."""
    for _ in range(max(1, int(steps))):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)


def accountant_epsilon(accountant: RDPAccountant, delta: float) -> float:
    """Return epsilon for the current accountant state."""
    return float(accountant.get_epsilon(delta=delta))
