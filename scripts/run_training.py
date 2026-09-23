from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.train import load_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single training run for FedSecure-Cart.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "smoke_test.yaml"), help="Path to YAML config.")
    parser.add_argument("--dp", action="store_true", help="Enable DP.")
    parser.add_argument("--secagg", action="store_true", help="Enable SecAgg.")
    parser.add_argument("--he", action="store_true", help="Enable HE.")
    parser.add_argument("--relax", action="store_true", help="Enable privacy relaxation.")
    parser.add_argument("--epsilon", type=float, default=None, help="Target epsilon.")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_training(
        config,
        dp_enabled=args.dp or bool(config.get("privacy", {}).get("dp_enabled", False)),
        secagg_enabled=args.secagg or bool(config.get("privacy", {}).get("secagg_enabled", False)),
        he_enabled=args.he or bool(config.get("privacy", {}).get("he_enabled", False)),
        relax_with_secagg=args.relax or bool(config.get("privacy", {}).get("relax_with_secagg", False)),
        epsilon=args.epsilon,
    )
    print(f"epsilon={result['epsilon']}")


if __name__ == "__main__":
    main()
