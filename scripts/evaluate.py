#!/usr/bin/env python3
"""Evaluate trained Pac-Man AI agents without exploration."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.training.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Pac-Man AI agents")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory with checkpoints")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "config.yaml")
    evaluator = Evaluator(config, run_dir)
    evaluator.evaluate(num_episodes=args.episodes)


if __name__ == "__main__":
    main()
