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
from federated.client import compute_delta, train_local_client
from federated.train import load_config, run_training
from models.ncf import NCF
from privacy.differential_privacy import clip_update
from privacy.secure_aggregation import SecureAggregator, masked_sum_invariant


ROOT = Path(__file__).resolve().parents[1]


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
