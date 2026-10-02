from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from attacks.gradient_inversion import run_gradient_inversion_attack
from attacks.membership_inference import loss_threshold_attack, shadow_model_attack
from data.movielens import load_movielens_data
from data.partition import compute_partition_stats
from evaluate.metrics import evaluate_model
from evaluate.run_experiment import build_experiment_grid
from federated.client import compute_delta, train_local_client
from federated.train import load_config, run_training
from models.ncf import NCF
from privacy.differential_privacy import clip_update
from privacy.secure_aggregation import SecureAggregator, masked_sum_invariant
from scripts.convert_amazon_catalog import convert_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_grid_sweeps_each_epsilon() -> None:
    epsilons = [0.5, 1.0, 2.0]
    defenses = ["no_defense", "secagg_only", "dp_only", "dp_secagg"]
    grid = build_experiment_grid(epsilons, defenses)

    assert len(grid) == 2 + 2 * len(epsilons)
    assert [spec["name"] for spec in grid[:2]] == ["no_defense", "secagg_only"]
    dp_specs = [spec for spec in grid if spec["dp_enabled"]]
    assert {spec["epsilon"] for spec in dp_specs} == set(epsilons)
    assert {spec["defense"] for spec in dp_specs} == {"dp_only", "dp_secagg"}


def test_amazon_catalog_implicit_feedback_pipeline(tmp_path: Path) -> None:
    catalog = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3", "p4", "p5", "other"],
            "category": ["Electronics|Audio"] * 5 + ["Home&Kitchen|Tools"],
            "user_id": ["u1,u2,u3", "u1,u2,u4", "u1,u3,u4", "u2,u3,u4", "u1,u2,u3,u4", "u1,u2,u3"],
        }
    )
    source_path = tmp_path / "amazon_catalog.csv"
    interactions_path = tmp_path / "amazon_interactions.csv"
    catalog.to_csv(source_path, index=False)
    convert_catalog(source_path, interactions_path)

    cfg = {
        "dataset": {
            "path": str(interactions_path),
            "implicit_feedback": True,
            "split_strategy": "random",
            "split_seed": 7,
            "min_user_ratings": 3,
            "min_item_ratings": 2,
            "negative_ratio": 1,
            "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        }
    }
    bundle = load_movielens_data(cfg)
    assert bundle.stats["n_users"] == 4
    assert bundle.stats["n_items"] == 5
    assert (bundle.train_df["rating"] == 0.0).any()
    assert (bundle.val_df["rating"] == 1.0).all()
    assert (bundle.test_df["rating"] == 1.0).all()

    positives = pd.concat(
        [
            bundle.train_df[bundle.train_df["rating"] > 0],
            bundle.val_df,
            bundle.test_df,
        ],
        ignore_index=True,
    )
    negative_pairs = set(zip(bundle.train_df.loc[bundle.train_df["rating"] == 0, "userId"], bundle.train_df.loc[bundle.train_df["rating"] == 0, "movieId"]))
    positive_pairs = set(zip(positives["userId"], positives["movieId"]))
    assert not negative_pairs & positive_pairs

    model = NCF(num_users=4, num_items=5, embedding_dim=4, mlp_layers=(8, 4), dropout=0.0)
    metrics = evaluate_model(
        model,
        bundle.val_df,
        top_k=[5],
        implicit_feedback=True,
        candidate_items=range(bundle.stats["n_items"]),
        seen_interactions=bundle.train_df,
    )
    assert metrics["rmse"] is None
    assert 0.0 <= metrics["recall"]["5"] <= 1.0
    assert 0.0 <= metrics["ndcg"]["5"] <= 1.0


def test_data_and_stats() -> None:
    cfg = load_config(ROOT / "configs" / "smoke_test.yaml")
    synthetic_cfg = {"dataset": {"path": "missing.csv", "synthetic": cfg["dataset"]["synthetic"], "min_user_ratings": 2, "min_item_ratings": 2, "split_fractions": cfg["dataset"]["split_fractions"]}}
    synthetic = load_movielens_data(synthetic_cfg)
    assert synthetic.stats["n_ratings"] == len(synthetic.train_df) + len(synthetic.val_df) + len(synthetic.test_df)
    assert synthetic.stats["n_ratings"] > 0
    stats = compute_partition_stats(synthetic.train_df, n_users=synthetic.stats["n_users"], n_items=synthetic.stats["n_items"])
    assert 0.0 <= stats["gini_ratings_per_user"] <= 1.0
    assert stats["sparsity_pct"] >= 0.0
    assert synthetic.stats["n_users"] > 0


def test_amazon_style_string_ids(tmp_path: Path) -> None:
    ratings = pd.DataFrame(
        {
            "userId": ["u1", "u1", "u2", "u2", "u3"],
            "movieId": ["m1", "m2", "m1", "m3", "m2"],
            "rating": [5.0, 4.0, 3.5, 2.0, 4.5],
            "timestamp": [100, 200, 150, 220, 300],
        }
    )
    path = tmp_path / "amazon_ratings.csv"
    ratings.to_csv(path, index=False)

    cfg = {
        "dataset": {
            "path": str(path),
            "min_user_ratings": 1,
            "min_item_ratings": 1,
            "split_fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        }
    }

    bundle = load_movielens_data(cfg)
    assert bundle.stats["n_users"] >= 3
    assert bundle.stats["n_items"] >= 3
    assert len(bundle.train_df) + len(bundle.val_df) + len(bundle.test_df) == bundle.stats["n_ratings"]


def test_secagg_invariant() -> None:
    raw_update = {"w": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([0.5])}
    client_ids = [1, 2, 3]
    secagg = SecureAggregator(client_ids, round_id=5)
    masked = [secagg.mask_update(cid, raw_update) for cid in client_ids]
    raw_sum = {name: sum((item[name] for item in [raw_update] * 3), start=torch.zeros_like(raw_update[name])) for name in raw_update}
    masked_sum = {name: sum((item[name] for item in masked), start=torch.zeros_like(raw_update[name])) for name in raw_update}
    assert masked_sum_invariant([raw_update, raw_update, raw_update], masked)
    for key in raw_sum:
        assert torch.allclose(raw_sum[key], masked_sum[key])


def test_dp_clipping() -> None:
    update = {"w": torch.randn(20, dtype=torch.float32), "b": torch.randn(5, dtype=torch.float32)}
    clip_norm = 0.75
    clipped = clip_update(update, clip_norm)
    total_norm_sq = sum(float(torch.sum(tensor ** 2)) for tensor in clipped.values())
    total_norm = float(np.sqrt(total_norm_sq))
    assert total_norm <= clip_norm + 1e-6


def test_federated_training_all_defense_modes() -> None:
    cfg = load_config(ROOT / "configs" / "smoke_test.yaml")
    combinations = [
        {"dp": False, "secagg": False, "he": False, "relax": False},
        {"dp": True, "secagg": False, "he": False, "relax": False},
        {"dp": False, "secagg": True, "he": False, "relax": False},
        {"dp": True, "secagg": True, "he": False, "relax": True},
    ]
    for combo in combinations:
        result = run_training(
            cfg,
            dp_enabled=combo["dp"],
            secagg_enabled=combo["secagg"],
            he_enabled=combo["he"],
            relax_with_secagg=combo["relax"],
        )
        assert len(result["history"]) == cfg["federated"]["num_rounds"]
        if combo["dp"]:
            assert result["epsilon"] is not None
            assert 0.0 <= result["epsilon"]
        else:
            assert result["epsilon"] is None


def test_membership_inference_attacks() -> None:
    cfg = load_config(ROOT / "configs" / "smoke_test.yaml")
    data = load_movielens_data(cfg)
    model = NCF(
        num_users=data.stats["n_users"],
        num_items=data.stats["n_items"],
        embedding_dim=8,
        mlp_layers=(16, 8),
        dropout=0.1,
    )
    model = train_local_client(model, data.train_df.head(20), epochs=1, batch_size=16, lr=0.01, device=torch.device("cpu"))
    auc1 = loss_threshold_attack(model, data.train_df.head(10), data.val_df.head(10))
    auc2 = shadow_model_attack(model, data.train_df.head(20), shadow_count=1, shadow_epochs=1, embedding_dim=8, mlp_layers=(16, 8), dropout=0.1, lr=0.01)
    assert 0.0 <= auc1 <= 1.0
    assert 0.0 <= auc2 <= 1.0


def test_gradient_inversion_support_recall() -> None:
    cfg = load_config(ROOT / "configs" / "smoke_test.yaml")
    data = load_movielens_data(cfg)
    base_model = NCF(
        num_users=data.stats["n_users"],
        num_items=data.stats["n_items"],
        embedding_dim=8,
        mlp_layers=(16, 8),
        dropout=0.1,
    )
    client_df = data.train_df.head(20)
    base_state = base_model.state_dict()
    trained = train_local_client(base_model, client_df, epochs=1, batch_size=16, lr=0.01, device=torch.device("cpu"))
    update = compute_delta(base_state, trained.state_dict())
    result = run_gradient_inversion_attack(trained, client_df, update, iterations=10, lr=0.05, threshold=1e-8)
    assert 0.0 <= result["support_precision"] <= 1.0
    assert 0.0 <= result["support_recall"] <= 1.0
    assert result["support_recall"] == 1.0
