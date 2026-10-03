# FedSecure-Cart

Federated Learning for Secure E-Commerce Recommendation.

FedSecure-Cart is a research-grade federated learning framework for movie recommendation that combines a neural collaborative filtering (NCF) model with privacy-preserving training. The project simulates non-IID client training, applies FedAvg, and evaluates the trade-off between utility, privacy, and communication overhead under differential privacy (DP), secure aggregation (SecAgg), and optional homomorphic encryption (HE).

## Highlights

- NCF-based recommendation model with GMF + MLP branches
- Federated training over non-IID user partitions
- Update-level DP with clipping and Gaussian noise
- Bonawitz-style secure aggregation with pairwise Diffie-Hellman masks
- Optional TenSEAL CKKS encryption for aggregated updates
- Attack evaluation for membership inference and gradient inversion
- YAML-driven experiment configuration and smoke-test reproduction

## Repository structure

- `attacks/` — privacy attack implementations and evaluation hooks
- `configs/` — default and smoke-test experiment configs
- `data/` — MovieLens loading and client partitioning utilities
- `evaluate/` — metrics and experiment-grid runner
- `federated/` — client/server orchestration and FL rounds
- `models/` — NCF recommendation model
- `privacy/` — DP, SecAgg, and HE implementations
- `scripts/` — CLI entry points for training and evaluation
- `tests/` — smoke tests for core functionality

## Quick start

1. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the smoke test suite:

```bash
pytest -q tests/test_smoke.py
```

3. Run a training pass using the smoke config:

```bash
python3 scripts/run_training.py --config configs/smoke_test.yaml
```

4. Run the evaluation grid:

```bash
python3 scripts/run_full_evaluation.py --config configs/smoke_test.yaml --output-dir outputs/smoke_grid
```

## Multi-Epsilon Dataset Experiments

Run the primary MovieLens-32M grid (six epsilon values and all five defense families):

```bash
python3 scripts/run_full_evaluation.py \
  --config configs/movielens_32m.yaml \
  --output-dir outputs/movielens_32m \
  --epsilons 0.25,0.5,1,2,4,8
```

The attached Amazon catalog contains aggregate product ratings and a limited list of reviewer IDs per product, but it has no individual review scores or timestamps. The Amazon workflow therefore treats listed reviewer-product pairs as implicit positives, uses a seeded random per-user holdout, and samples unobserved products as training negatives. Amazon results report ranking and attack metrics; RMSE and chronological claims are intentionally omitted. Its reduced run uses one FL round and five sampled clients per round.

For a matched pilot across both datasets, use the same core defenses and epsilon values. MovieLens uses a reproducible 3,000-user sample from the real 32M file; Amazon uses the Electronics implicit-feedback protocol above.

```bash
python3 scripts/run_full_evaluation.py \
  --config configs/movielens_32m_comparison.yaml \
  --output-dir outputs/movielens_32m_comparison \
  --epsilons 0.5,1,2 \
  --defenses no_defense,secagg_only,dp_only,dp_secagg

python3 scripts/plot_dataset_comparison.py \
  --movielens outputs/movielens_32m_comparison/results.csv \
  --amazon outputs/amazon_electronics_implicit/results.csv \
  --output-dir outputs/cross_dataset
```

The cross-dataset report separates explicit-rating RMSE from implicit-feedback ranking metrics. Membership attack AUC is centered on the 0.5 chance baseline, gradient-inversion support precision is shown separately, and runtime is normalized to each dataset's no-defense run.

The comparison artifacts are saved under `outputs/cross_dataset/`: `membership_leakage.png`, `shadow_attack.png`, `recommendation_quality.png`, `movielens_rmse.png`, `gradient_inversion_resistance.png`, `relative_runtime.png`, `combined_results.csv`, and `COMPARISON_NOTES.md`. The MovieLens comparison uses a deterministic sample, not the full 32M-run grid.

Convert the Electronics category and run the reduced core-defense grid:

```bash
python3 scripts/convert_amazon_catalog.py \
  --input /Users/mac/Desktop/TS_Material/amazon.csv \
  --output /Users/mac/Desktop/TS_Material/amazon_electronics_implicit.csv \
  --category Electronics

python3 scripts/run_full_evaluation.py \
  --config configs/amazon_electronics_implicit.yaml \
  --output-dir outputs/amazon_electronics_implicit \
  --epsilons 0.5,1,2 \
  --defenses no_defense,secagg_only,dp_only,dp_secagg
```

Each run writes `results.csv`, `metrics_by_epsilon.png`, and `privacy_utility.png`. The CSV contains defense flags, epsilon, data dimensions/density, federated and model parameters, attack scores, ranking metrics, and wall-clock time.

## Configuration

The project uses YAML configs in `configs/`.

- `configs/smoke_test.yaml` — lightweight validation run
- `configs/default.yaml` — larger reproducible experiment setup
- `configs/amazon_movies_tv.yaml` — Amazon review comparison setup
- `configs/movielens_32m.yaml` — MovieLens-32M primary experiment
- `configs/amazon_electronics_implicit.yaml` — reduced Amazon Electronics implicit-feedback experiment

Fields include FL settings, client counts, privacy parameters, defense toggles, and evaluation details.

## Outputs

The experiment pipeline writes outputs to `outputs/`.

- CSV results tables
- Pareto-style performance plots
- smoke-test artifacts for quick verification

## Research intent

The framework is designed for studying the privacy-utility trade-off in recommendation systems under federated training, with the baseline defenses and attacks directly wired into the same evaluation loop for reproducible comparison.

## Validation

The smoke suite verifies the main pipeline components, including model initialization, dataset loading, DP, SecAgg, HE, and metric generation.
