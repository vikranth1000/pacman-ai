#!/usr/bin/env python3
"""Train Pac-Man AI agents — headless, maximum speed."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Pac-Man AI agents")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Specific run directory (for resume)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in run-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Trainer(config, run_dir=args.run_dir, resume=args.resume)

    try:
        trainer.train(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        from src.training.checkpoint import save_checkpoint
        save_checkpoint(trainer.run_dir, trainer.start_episode, trainer.agents, config)
        print(f"Checkpoint saved to {trainer.run_dir}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
