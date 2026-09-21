# FedSecure-Cart

Federated Learning for Secure E-Commerce Recommendation.

FedSecure-Cart trains a Neural Collaborative Filtering (NCF) model across
simulated clients using **FedAvg**, protected by a tri-layer privacy pipeline:
**Differential Privacy** (gradient clipping + Gaussian noise), **Secure
Aggregation** (pairwise masking), and optional **Homomorphic Encryption**.
The framework is benchmarked on MovieLens-32M under non-IID client splits
and evaluated against gradient inversion and membership inference attacks
to quantify the privacy–utility–overhead tradeoff.
