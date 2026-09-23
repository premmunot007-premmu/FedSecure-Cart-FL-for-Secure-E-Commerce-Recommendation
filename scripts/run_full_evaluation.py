from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.run_experiment import run_experiment
from federated.train import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FedSecure-Cart defense grid and write the paper table.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "smoke_test.yaml"), help="Path to YAML config.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "grid"), help="Directory for CSV and plots.")
    args = parser.parse_args()
    config = load_config(args.config)
    results = run_experiment(config, output_dir=args.output_dir, epsilons=[0.5, 1.0, 2.0])
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
