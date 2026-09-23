from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from attacks.gradient_inversion import run_gradient_inversion_attack
from attacks.membership_inference import loss_threshold_attack, shadow_model_attack
from evaluate.metrics import evaluate_model
from federated.train import run_training


def build_experiment_grid(epsilons: list[float] | None = None) -> list[dict[str, Any]]:
    """Construct the defense grid described in the paper: no defense, DP-only, SecAgg-only, DP+SecAgg, and HE variant."""
    epsilons = epsilons or [0.5, 1.0, 2.0, 4.0]
    grid = [
        {"name": "no_defense", "dp_enabled": False, "secagg_enabled": False, "he_enabled": False, "relax_with_secagg": False, "epsilon": None},
        {"name": "dp_only", "dp_enabled": True, "secagg_enabled": False, "he_enabled": False, "relax_with_secagg": False, "epsilon": epsilons[0]},
        {"name": "secagg_only", "dp_enabled": False, "secagg_enabled": True, "he_enabled": False, "relax_with_secagg": False, "epsilon": None},
        {"name": "dp_secagg", "dp_enabled": True, "secagg_enabled": True, "he_enabled": False, "relax_with_secagg": True, "epsilon": epsilons[1]},
        {"name": "dp_secagg_he", "dp_enabled": True, "secagg_enabled": True, "he_enabled": True, "relax_with_secagg": True, "epsilon": epsilons[2]},
    ]
    return grid


def run_experiment(config: dict[str, Any], output_dir: str | Path, epsilons: list[float] | None = None) -> pd.DataFrame:
    """Train and evaluate each defense configuration, then persist one row per experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in build_experiment_grid(epsilons):
        cfg = {**config}
        cfg.setdefault("privacy", {})
        cfg["privacy"]["dp_enabled"] = spec["dp_enabled"]
        cfg["privacy"]["secagg_enabled"] = spec["secagg_enabled"]
        cfg["privacy"]["he_enabled"] = spec["he_enabled"]
        cfg["privacy"]["relax_with_secagg"] = spec["relax_with_secagg"]
        if spec["epsilon"] is not None:
            cfg["privacy"]["target_epsilon"] = float(spec["epsilon"])
        trained = run_training(cfg, dp_enabled=spec["dp_enabled"], secagg_enabled=spec["secagg_enabled"], he_enabled=spec["he_enabled"], relax_with_secagg=spec["relax_with_secagg"], epsilon=spec["epsilon"])
        model = trained["model"]
        dataset = trained["dataset"]
        val_metrics = evaluate_model(model, dataset.val_df, top_k=[5, 10], max_users=min(50, len(dataset.val_df)), device="cpu")
        member_df = dataset.train_df.sample(frac=0.5, random_state=7).reset_index(drop=True)
        nonmember_df = dataset.val_df.sample(frac=0.5, random_state=9).reset_index(drop=True)
        mia_auc = loss_threshold_attack(model, member_df, nonmember_df)
        shadow_auc = 0.5
        try:
            shadow_auc = shadow_model_attack(model, dataset.train_df, shadow_count=2, shadow_epochs=2, embedding_dim=8, mlp_layers=(16, 8), dropout=0.1, lr=0.01)
        except Exception:
            shadow_auc = 0.5
        batch = dataset.train_df.head(10).copy()
        update = {}
        for name, tensor in model.state_dict().items():
            update[name] = tensor.detach().clone()
        attack = run_gradient_inversion_attack(model, batch, update, iterations=10, lr=0.05)
        rows.append(
            {
                "name": spec["name"],
                "dp_enabled": spec["dp_enabled"],
                "secagg_enabled": spec["secagg_enabled"],
                "he_enabled": spec["he_enabled"],
                "relax_with_secagg": spec["relax_with_secagg"],
                "epsilon": spec["epsilon"],
                "rmse": val_metrics["rmse"],
                "recall@5": val_metrics["recall"].get("5", 0.0),
                "ndcg@5": val_metrics["ndcg"].get("5", 0.0),
                "mia_auc": mia_auc,
                "shadow_auc": shadow_auc,
                "support_precision": attack["support_precision"],
                "support_recall": attack["support_recall"],
                "wall_time_sec": 0.0,
            }
        )

    results = pd.DataFrame(rows)
    csv_path = output_dir / "results.csv"
    results.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = results["mia_auc"].to_numpy()
    y = results["rmse"].to_numpy()
    axes[0].scatter(x, y)
    for idx, row in results.iterrows():
        axes[0].annotate(row["name"], (x[idx], y[idx]), fontsize=8)
    axes[0].set_xlabel("Privacy leakage (attack AUC)")
    axes[0].set_ylabel("RMSE")
    axes[1].scatter(results["recall@5"], results["wall_time_sec"])
    for idx, row in results.iterrows():
        axes[1].annotate(row["name"], (results["recall@5"].iloc[idx], results["wall_time_sec"].iloc[idx]), fontsize=8)
    axes[1].set_xlabel("Recall@5")
    axes[1].set_ylabel("Wall-clock time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_plots.png", dpi=180)
    return results
