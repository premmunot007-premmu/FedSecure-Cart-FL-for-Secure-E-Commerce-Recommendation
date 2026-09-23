from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    import tenseal as ts
except Exception:  # pragma: no cover
    ts = None


class HEKeyHolder:
    """Owns the secret key and exposes only the public context to the server.

    This directly follows the requirement that the server never sees the private key and
    only performs ciphertext addition under the public CKKS context. The attacker can only
    observe the encrypted aggregate, not the raw update tensor.
    """

    def __init__(self, poly_modulus_degree: int = 8192, coeff_mod_bit_sizes: list[int] | tuple[int, ...] = (60, 40, 40, 60), global_scale: int = 2**40) -> None:
        if ts is None:
            raise ImportError("tenseal is required for homomorphic encryption support.")
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, coeff_mod_bit_sizes=list(coeff_mod_bit_sizes))
        self.context.generate_galois_keys()
        self.context.global_scale = global_scale
        self.public_context_bytes = self.context.serialize(save_secret_key=False)

    def encrypt(self, tensor: torch.Tensor) -> bytes:
        values = tensor.detach().cpu().flatten().tolist()
        vector = ts.ckks_vector(self.context, values)
        return vector.serialize()

    def decrypt(self, encrypted: bytes) -> torch.Tensor:
        vector = ts.ckks_vector_from(self.context, encrypted)
        values = np.asarray(vector.decrypt())
        return torch.as_tensor(values, dtype=torch.float32)


def public_context_bytes_from_config(config: dict[str, Any]) -> bytes:
    """Build a public-only CKKS context from YAML config."""
    if ts is None:
        raise ImportError("tenseal is required for homomorphic encryption support.")
    poly_modulus_degree = int(config.get("poly_modulus_degree", 8192))
    coeff_bits = tuple(int(v) for v in config.get("coeff_mod_bit_sizes", [60, 40, 40, 60]))
    global_scale = int(config.get("global_scale", 2**40))
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, coeff_mod_bit_sizes=list(coeff_bits))
    context.generate_galois_keys()
    context.global_scale = global_scale
    return context.serialize(save_secret_key=False)


def aggregate_ciphertexts(public_context: bytes, encrypted_values: list[bytes]) -> bytes:
    """The server performs ciphertext addition only on the public context."""
    if ts is None:
        raise ImportError("tenseal is required for homomorphic encryption support.")
    if not encrypted_values:
        raise ValueError("No ciphertexts provided to aggregate.")
    context = ts.context_from(public_context)
    total = ts.ckks_vector_from(context, encrypted_values[0])
    for encrypted in encrypted_values[1:]:
        total = total + ts.ckks_vector_from(context, encrypted)
    return total.serialize()
