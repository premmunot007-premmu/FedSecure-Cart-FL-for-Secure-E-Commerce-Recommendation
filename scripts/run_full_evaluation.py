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
    parser.add_argument("--epsilons", type=str, default="0.25,0.5,1,2,4,8", help="Comma-separated DP epsilon values.")
    parser.add_argument("--defenses", type=str, default="no_defense,secagg_only,dp_only,dp_secagg,dp_secagg_he", help="Comma-separated defense names.")
    args = parser.parse_args()
    config = load_config(args.config)
    epsilons = [float(value.strip()) for value in args.epsilons.split(",") if value.strip()]
    defenses = [value.strip() for value in args.defenses.split(",") if value.strip()]
    results = run_experiment(config, output_dir=args.output_dir, epsilons=epsilons, defenses=defenses)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
