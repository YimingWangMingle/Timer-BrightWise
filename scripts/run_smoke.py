from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.config import TimerConfig
from tsfm.training import SmokeTrainingConfig, run_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CPU-only Timer smoke test")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model" / "tiny.json",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "smoke.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smoke",
    )
    parser.add_argument("--steps", type=int, help="Override the configured step count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = TimerConfig.from_json(args.model_config)
    training_values = json.loads(args.training_config.read_text(encoding="utf-8"))
    training_config = SmokeTrainingConfig(**training_values)
    if args.steps is not None:
        training_config = replace(training_config, steps=args.steps)

    result = run_smoke(model_config, training_config, args.output_dir)
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")
    print(f"checkpoint={result.checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
