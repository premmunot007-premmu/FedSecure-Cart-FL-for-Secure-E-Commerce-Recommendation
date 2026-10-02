from __future__ import annotations

import copy
import time
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


def build_experiment_grid(
    epsilons: list[float] | None = None,
    defenses: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build two non-DP baselines and every DP defense at each requested epsilon."""
    epsilons = epsilons or [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    defense_specs = {
        "no_defense": (False, False, False, False),
        "secagg_only": (False, True, False, False),
        "dp_only": (True, False, False, False),
        "dp_secagg": (True, True, False, True),
        "dp_secagg_he": (True, True, True, True),
    }
    selected = defenses or list(defense_specs)
    unknown = set(selected) - set(defense_specs)
    if unknown:
        raise ValueError(f"Unsupported defenses: {sorted(unknown)}")

    grid: list[dict[str, Any]] = []
    for name in ("no_defense", "secagg_only"):
        if name in selected:
            dp_enabled, secagg_enabled, he_enabled, relax = defense_specs[name]
            grid.append(
                {
                    "name": name,
                    "defense": name,
                    "dp_enabled": dp_enabled,
                    "secagg_enabled": secagg_enabled,
                    "he_enabled": he_enabled,
                    "relax_with_secagg": relax,
                    "epsilon": None,
                }
            )

    dp_defenses = [name for name in ("dp_only", "dp_secagg", "dp_secagg_he") if name in selected]
    for epsilon in epsilons:
        for name in dp_defenses:
            dp_enabled, secagg_enabled, he_enabled, relax = defense_specs[name]
            grid.append(
                {
                    "name": f"{name}_eps_{epsilon:g}",
                    "defense": name,
                    "dp_enabled": dp_enabled,
                    "secagg_enabled": secagg_enabled,
                    "he_enabled": he_enabled,
                    "relax_with_secagg": relax,
                    "epsilon": float(epsilon),
                }
            )
    return grid


def _write_plots(results: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("rmse", "RMSE (lower is better)"),
        ("recall@5", "Recall@5 (higher is better)"),
        ("recall@10", "Recall@10 (higher is better)"),
        ("ndcg@5", "NDCG@5 (higher is better)"),
        ("ndcg@10", "NDCG@10 (higher is better)"),
        ("mia_auc", "Threshold MIA AUC (0.5 is chance)"),
        ("shadow_auc", "Shadow MIA AUC (0.5 is chance)"),
        ("support_precision", "Gradient support precision"),
        ("support_recall", "Gradient support recall"),
        ("wall_time_sec", "Run time (seconds)"),
    ]
    metrics = [(column, label) for column, label in metrics if column in results and results[column].notna().any()]
    rows = max(1, (len(metrics) + 3) // 4)
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.2 * rows), squeeze=False)
    for axis, (column, label) in zip(axes.flat, metrics):
        for defense, group in results.groupby("defense", sort=False):
            group = group.sort_values("epsilon", na_position="first")
            x_values = group["epsilon"].astype(float).fillna(0.0)
            axis.plot(x_values, group[column], marker="o", label=defense)
        axis.set_title(label)
        axis.set_xlabel("Target ε (0 denotes no-DP baseline)")
        axis.grid(True, alpha=0.25)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "metrics_by_epsilon.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for defense, group in results.groupby("defense", sort=False):
        axis.scatter(group["mia_auc"], group["recall@5"], label=defense, s=55)
        for _, row in group.iterrows():
            epsilon_label = "baseline" if pd.isna(row["epsilon"]) else f"ε={row['epsilon']:g}"
            axis.annotate(epsilon_label, (row["mia_auc"], row["recall@5"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Threshold membership-inference AUC (0.5 is chance)")
    axis.set_ylabel("Recall@5 (higher is better)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "privacy_utility.png", dpi=180)
    plt.close(fig)


def run_experiment(
    config: dict[str, Any],
    output_dir: str | Path,
    epsilons: list[float] | None = None,
    defenses: list[str] | None = None,
) -> pd.DataFrame:
    """Train and evaluate each defense configuration, then persist one row per experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    training_cfg = config.get("federated", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    attack_cfg = config.get("attacks", {})
    for spec in build_experiment_grid(epsilons, defenses):
        started_at = time.perf_counter()
        cfg = copy.deepcopy(config)
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
        implicit_feedback = bool(dataset_cfg.get("implicit_feedback", False))
        positives_train = dataset.train_df[dataset.train_df["rating"] > 0] if implicit_feedback else dataset.train_df
        positives_val = dataset.val_df[dataset.val_df["rating"] > 0] if implicit_feedback else dataset.val_df
        val_metrics = evaluate_model(
            model,
            dataset.val_df,
            top_k=[5, 10],
            max_users=min(int(config.get("evaluation", {}).get("max_eval_users", 50)), len(dataset.val_df)),
            device="cpu",
            implicit_feedback=implicit_feedback,
            candidate_items=range(int(dataset.stats["n_items"])),
            seen_interactions=dataset.train_df,
        )
        member_df = positives_train.sample(frac=0.5, random_state=7).reset_index(drop=True)
        nonmember_df = positives_val.sample(frac=0.5, random_state=9).reset_index(drop=True)
        mia_auc = loss_threshold_attack(model, member_df, nonmember_df)
        shadow_auc = 0.5
        try:
            shadow_size = min(len(positives_train), len(positives_val), 2000)
            shadow_data = pd.concat(
                [
                    positives_train.sample(n=shadow_size, random_state=17),
                    positives_val.sample(n=shadow_size, random_state=19),
                ],
                ignore_index=True,
            )
            shadow_auc = shadow_model_attack(
                model,
                shadow_data,
                shadow_count=int(attack_cfg.get("shadow_model_count", 2)),
                shadow_epochs=int(attack_cfg.get("shadow_epochs", 2)),
                embedding_dim=8,
                mlp_layers=(16, 8),
                dropout=0.1,
                lr=0.01,
            )
        except Exception:
            shadow_auc = 0.5
        last_round = trained["last_round_result"] or {}
        selected = last_round.get("selected_clients", [])
        selected_rows = [trained["client_data"][client_id] for client_id in selected if client_id in trained["client_data"]]
        attack_data = pd.concat(selected_rows, ignore_index=True) if selected_rows else positives_train
        batch = attack_data[attack_data["rating"] > 0].head(10).copy()
        update = last_round.get("aggregate", {})
        attack = run_gradient_inversion_attack(model, batch, update, iterations=10, lr=0.05)
        elapsed = time.perf_counter() - started_at
        rows.append(
            {
                "name": spec["name"],
                "defense": spec["defense"],
                "dp_enabled": spec["dp_enabled"],
                "secagg_enabled": spec["secagg_enabled"],
                "he_enabled": spec["he_enabled"],
                "relax_with_secagg": spec["relax_with_secagg"],
                "epsilon": spec["epsilon"],
                "rmse": val_metrics["rmse"],
                "recall@5": val_metrics["recall"].get("5", 0.0),
                "recall@10": val_metrics["recall"].get("10", 0.0),
                "ndcg@5": val_metrics["ndcg"].get("5", 0.0),
                "ndcg@10": val_metrics["ndcg"].get("10", 0.0),
                "mia_auc": mia_auc,
                "shadow_auc": shadow_auc,
                "support_precision": attack["support_precision"],
                "support_recall": attack["support_recall"],
                "wall_time_sec": elapsed,
                "split_strategy": dataset_cfg.get("split_strategy", "chronological"),
                "implicit_feedback": implicit_feedback,
                "negative_ratio": dataset_cfg.get("negative_ratio", 0),
                "dataset_n_users": dataset.stats["n_users"],
                "dataset_n_items": dataset.stats["n_items"],
                "dataset_n_interactions": dataset.stats["n_ratings"],
                "dataset_density": dataset.stats["density"],
                "num_clients": training_cfg.get("num_clients"),
                "clients_per_round": training_cfg.get("clients_per_round"),
                "num_rounds": training_cfg.get("num_rounds"),
                "local_epochs": training_cfg.get("local_epochs"),
                "batch_size": training_cfg.get("batch_size"),
                "learning_rate": training_cfg.get("lr"),
                "embedding_dim": model_cfg.get("embedding_dim"),
                "mlp_layers": ",".join(str(value) for value in model_cfg.get("mlp_layers", [])),
                "seed": training_cfg.get("seed"),
            }
        )

    results = pd.DataFrame(rows)
    csv_path = output_dir / "results.csv"
    results.to_csv(csv_path, index=False)

    _write_plots(results, output_dir)
    return results
