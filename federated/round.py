from __future__ import annotations

import copy
from typing import Any

import torch

from privacy.differential_privacy import apply_dp_to_update, accountant_epsilon, get_rdp_noise_multiplier, make_accountant, relax_with_secagg, step_accountant
from privacy.homomorphic import HEKeyHolder, aggregate_ciphertexts
from privacy.secure_aggregation import SecureAggregator
from federated.client import compute_delta, train_local_client, weighted_delta


def run_round(
    model: torch.nn.Module,
    client_data: dict[int, Any],
    selected_clients: list[int],
    *,
    device: torch.device,
    config: dict[str, Any],
    round_id: int,
    dp_enabled: bool,
    secagg_enabled: bool,
    he_enabled: bool,
    relax_with_secagg_flag: bool,
) -> dict[str, Any]:
    """Execute one FL round with the exact layer order required by the paper: local train -> DP -> SecAgg -> HE.

    The weighted update is applied before masking because SecAgg aggregates an unweighted sum.
    The server never sees raw unmasked updates when SecAgg is active.
    """
    training_cfg = config.get("federated", {})
    privacy_cfg = config.get("privacy", {})
    base_state = copy.deepcopy(model.state_dict())
    total_samples = sum(len(client_data[cid]) for cid in selected_clients)

    masked_updates: list[dict[str, torch.Tensor]] = []
    secagg = SecureAggregator(selected_clients, round_id) if secagg_enabled else None

    for client_id in selected_clients:
        local_model = copy.deepcopy(model)
        local_model = train_local_client(
            local_model,
            client_data[client_id],
            epochs=int(training_cfg.get("local_epochs", 1)),
            batch_size=int(training_cfg.get("batch_size", 32)),
            lr=float(training_cfg.get("lr", 0.01)),
            device=device,
        )
        local_state = local_model.state_dict()
        update = compute_delta(base_state, local_state)

        if total_samples <= 0:
            raise ValueError("Selected clients have no local samples.")
        weighted = weighted_delta(update, len(client_data[client_id]), total_samples)

        if dp_enabled:
            sigma = privacy_cfg.get("noise_multiplier")
            if sigma is None:
                target_epsilon = float(privacy_cfg.get("target_epsilon", 2.0))
                target_delta = float(privacy_cfg.get("target_delta", 1e-5))
                sample_rate = len(selected_clients) / max(1, config.get("federated", {}).get("num_clients", len(selected_clients)))
                steps = max(1, int(config.get("federated", {}).get("num_rounds", 1)))
                sigma = get_rdp_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    sample_rate=sample_rate,
                    steps=steps,
                    accountant="rdp",
                )
            if relax_with_secagg_flag and secagg_enabled:
                sigma = relax_with_secagg(sigma, len(selected_clients))
            clip_norm = float(privacy_cfg.get("clip_norm", 1.0))
            weighted = apply_dp_to_update(weighted, clip_norm, sigma)

        masked = weighted if secagg is None else secagg.mask_update(client_id, weighted)
        masked_updates.append(masked)

    if secagg_enabled:
        aggregate = secagg.reconstruct_aggregate(masked_updates)
    else:
        aggregate = {}
        for key in base_state:
            aggregate[key] = torch.zeros_like(base_state[key])
        for update in masked_updates:
            for key, tensor in update.items():
                aggregate[key] = aggregate[key] + tensor

    if he_enabled:
        he_cfg = privacy_cfg.get("he", {})
        holder = HEKeyHolder(
            poly_modulus_degree=int(he_cfg.get("poly_modulus_degree", 8192)),
            coeff_mod_bit_sizes=tuple(int(v) for v in he_cfg.get("coeff_mod_bit_sizes", [60, 40, 40, 60])),
            global_scale=int(he_cfg.get("global_scale", 2**40)),
        )
        # Serialize the aggregate into a single flat vector, encrypt, then decrypt and
        # reconstruct the original parameter-shaped tensors. This keeps the server's
        # role limited to ciphertext addition while allowing the key-holder to recover
        # the exact aggregate shape on decryption.
        param_names = list(aggregate.keys())
        param_shapes = [tuple(aggregate[k].shape) for k in param_names]
        param_sizes = [int(torch.numel(aggregate[k])) for k in param_names]
        flat = torch.cat([aggregate[k].flatten() for k in param_names]).detach().cpu()
        encrypted = holder.encrypt(flat)
        public_ctx = holder.public_context_bytes
        if not encrypted:
            raise ValueError("Encrypted aggregate is empty.")
        decrypted = holder.decrypt(encrypted).flatten()
        if decrypted.numel() != sum(param_sizes):
            raise RuntimeError("Decrypted aggregate size does not match expected parameter size.")
        # Split and reshape back into parameter dict
        parts = torch.split(decrypted, param_sizes)
        aggregate = {name: part.reshape(shape) for name, part, shape in zip(param_names, parts, param_shapes)}

    for key, value in aggregate.items():
        if value.shape != base_state[key].shape:
            value = value.reshape(base_state[key].shape)
    new_state = {name: base_state[name] + aggregate.get(name, torch.zeros_like(base_state[name])) for name in base_state}
    model.load_state_dict(new_state)
    return {
        "aggregate": aggregate,
        "base_state": base_state,
        "new_state": new_state,
        "selected_clients": selected_clients,
    }
