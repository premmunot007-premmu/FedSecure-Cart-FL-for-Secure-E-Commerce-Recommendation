from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from data.movielens import load_movielens_data
from data.partition import build_client_train_data
from evaluate.metrics import evaluate_model
from federated.round import run_round
from models.ncf import NCF
from privacy.differential_privacy import accountant_epsilon, get_rdp_noise_multiplier, make_accountant, step_accountant


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI override flags to the config object."""
    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"dp", "secagg", "he", "relax"}:
            config.setdefault("privacy", {})
            config["privacy"]["dp_enabled" if key == "dp" else "secagg_enabled" if key == "secagg" else "he_enabled" if key == "he" else "relax_with_secagg"] = bool(value)
        elif key == "epsilon":
            config.setdefault("privacy", {})
            config["privacy"]["target_epsilon"] = float(value)
    return config


def resolve_noise_multiplier(config: dict[str, Any], *, num_clients: int, clients_per_round: int, rounds: int) -> float:
    """Compute sigma either from the explicit YAML value or by solving the target epsilon."""
    privacy_cfg = config.get("privacy", {})
    explicit = privacy_cfg.get("noise_multiplier")
    if explicit is not None:
        return float(explicit)
    target_epsilon = float(privacy_cfg.get("target_epsilon", 2.0))
    target_delta = float(privacy_cfg.get("target_delta", 1e-5))
    sample_rate = clients_per_round / max(1, num_clients)
    return float(
        get_rdp_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=max(1, int(rounds)),
            accountant="rdp",
        )
    )


def run_training(
    config: dict[str, Any] | str | Path,
    *,
    dp_enabled: bool | None = None,
    secagg_enabled: bool | None = None,
    he_enabled: bool | None = None,
    relax_with_secagg: bool | None = None,
    epsilon: float | None = None,
) -> dict[str, Any]:
    """Train the federated recommender for the full round schedule and return the final model plus diagnostics."""
    if isinstance(config, (str, Path)):
        config = load_config(config)
    config = copy.deepcopy(config)
    if dp_enabled is not None:
        config.setdefault("privacy", {})["dp_enabled"] = bool(dp_enabled)
    if secagg_enabled is not None:
        config.setdefault("privacy", {})["secagg_enabled"] = bool(secagg_enabled)
    if he_enabled is not None:
        config.setdefault("privacy", {})["he_enabled"] = bool(he_enabled)
    if relax_with_secagg is not None:
        config.setdefault("privacy", {})["relax_with_secagg"] = bool(relax_with_secagg)
    if epsilon is not None:
        config.setdefault("privacy", {})["target_epsilon"] = float(epsilon)

    device = torch.device(config.get("experiment", {}).get("device", "cpu"))
    dataset_bundle = load_movielens_data(config)
    client_data = build_client_train_data(dataset_bundle.train_df)
    num_clients = int(config.get("federated", {}).get("num_clients", max(1, len(client_data))))
    clients_per_round = int(config.get("federated", {}).get("clients_per_round", min(num_clients, max(1, len(client_data)))))
    num_rounds = int(config.get("federated", {}).get("num_rounds", 1))
    seed = int(config.get("federated", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = NCF(
        num_users=int(dataset_bundle.stats["n_users"]),
        num_items=int(dataset_bundle.stats["n_items"]),
        embedding_dim=int(config.get("model", {}).get("embedding_dim", 16)),
        mlp_layers=tuple(int(v) for v in config.get("model", {}).get("mlp_layers", [32, 16])),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    )
    model.to(device)
    accountant = make_accountant()

    history: list[dict[str, Any]] = []
    epsilon_value: float | None = None
    for round_id in range(num_rounds):
        selected_clients = sorted(random.sample(sorted(client_data), min(len(client_data), clients_per_round)))
        round_result = run_round(
            model,
            client_data,
            selected_clients,
            device=device,
            config=config,
            round_id=round_id,
            dp_enabled=bool(config.get("privacy", {}).get("dp_enabled", False)),
            secagg_enabled=bool(config.get("privacy", {}).get("secagg_enabled", False)),
            he_enabled=bool(config.get("privacy", {}).get("he_enabled", False)),
            relax_with_secagg_flag=bool(config.get("privacy", {}).get("relax_with_secagg", False)),
        )

        if config.get("privacy", {}).get("dp_enabled", False):
            noise_multiplier = resolve_noise_multiplier(config, num_clients=num_clients, clients_per_round=clients_per_round, rounds=num_rounds)
            if config.get("privacy", {}).get("relax_with_secagg", False) and config.get("privacy", {}).get("secagg_enabled", False):
                noise_multiplier = noise_multiplier / np.sqrt(clients_per_round)
            sample_rate = len(selected_clients) / max(1, num_clients)
            step_accountant(accountant, noise_multiplier, sample_rate, steps=1)
            epsilon_value = accountant_epsilon(accountant, float(config.get("privacy", {}).get("target_delta", 1e-5)))

        if round_id % max(1, int(config.get("experiment", {}).get("log_every", 1))) == 0:
            metrics = evaluate_model(
                model,
                dataset_bundle.val_df,
                top_k=[int(k) for k in config.get("evaluation", {}).get("top_k", [5])],
                max_users=int(config.get("evaluation", {}).get("max_eval_users", len(dataset_bundle.val_df))),
                device=device,
            )
            history.append({"round": round_id, "epsilon": epsilon_value, **metrics})

    return {
        "model": model,
        "dataset": dataset_bundle,
        "client_data": client_data,
        "history": history,
        "epsilon": epsilon_value,
        "config": config,
        "accountant": accountant,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run one FL training job for FedSecure-Cart.")
    parser.add_argument("--config", type=str, required=True, help="YAML filepath to the configuration.")
    parser.add_argument("--dp", action="store_true", help="Enable DP.")
    parser.add_argument("--secagg", action="store_true", help="Enable secure aggregation.")
    parser.add_argument("--he", action="store_true", help="Enable homomorphic encryption.")
    parser.add_argument("--relax", action="store_true", help="Enable SecAgg privacy relaxation.")
    parser.add_argument("--epsilon", type=float, default=None, help="Target epsilon for DP.")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_training(
        config,
        dp_enabled=args.dp or config.get("privacy", {}).get("dp_enabled", False),
        secagg_enabled=args.secagg or config.get("privacy", {}).get("secagg_enabled", False),
        he_enabled=args.he or config.get("privacy", {}).get("he_enabled", False),
        relax_with_secagg=args.relax or config.get("privacy", {}).get("relax_with_secagg", False),
        epsilon=args.epsilon,
    )
    print({"epsilon": result["epsilon"], "history_len": len(result["history"])})


if __name__ == "__main__":
    _cli()
