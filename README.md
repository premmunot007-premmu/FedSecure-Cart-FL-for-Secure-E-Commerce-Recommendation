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

## Amazon Movies and TV comparison setup

Use the real Amazon review dataset alongside MovieLens by converting the review JSONL to the project’s expected CSV schema.

```bash
python3 scripts/convert_amazon_reviews.py \
  --input /Users/mac/Downloads/Movies_and_TV.jsonl \
  --output /Users/mac/Downloads/amazon_movies_tv_ratings.csv \
  --max-rows 200000
```

Then point the project to the generated file by using the provided config:

```bash
python3 scripts/run_training.py --config configs/amazon_movies_tv.yaml
```

This config is tuned for a lighter Amazon comparison run and keeps the dataset size manageable while still producing a valid federated recommendation benchmark.

## Configuration

The project uses YAML configs in `configs/`.

- `configs/smoke_test.yaml` — lightweight validation run
- `configs/default.yaml` — larger reproducible experiment setup
- `configs/amazon_movies_tv.yaml` — Amazon review comparison setup

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
