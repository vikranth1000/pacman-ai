"""Checkpoint save/load for all agents and training state."""

import json
from pathlib import Path
import torch

from src.agents.dqn_agent import DQNAgent


def save_checkpoint(run_dir: Path, episode: int, agents: dict[str, DQNAgent],
                    config: dict, save_replay: bool = False):
    """Save all agent checkpoints and training state."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save each agent
    for name, agent in agents.items():
        agent_path = ckpt_dir / f"{name}_ep{episode}.pt"
        agent.save(str(agent_path))

    # Save latest symlink-like reference
    state = {
        "episode": episode,
        "agent_names": list(agents.keys()),
        "checkpoint_pattern": "checkpoints/{name}_ep{episode}.pt",
    }
    with open(run_dir / "latest_checkpoint.json", "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint(run_dir: Path, agents: dict[str, DQNAgent],
                    episode: int | None = None) -> int:
    """Load agent checkpoints. Returns the episode number loaded.

    If episode is None, loads the latest checkpoint.
    """
    latest_path = run_dir / "latest_checkpoint.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    with open(latest_path) as f:
        state = json.load(f)

    if episode is None:
        episode = state["episode"]

    ckpt_dir = run_dir / "checkpoints"
    for name, agent in agents.items():
        agent_path = ckpt_dir / f"{name}_ep{episode}.pt"
        if agent_path.exists():
            agent.load(str(agent_path))
        else:
            raise FileNotFoundError(f"Checkpoint not found: {agent_path}")

    return episode


def find_latest_run(base_dir: Path) -> Path | None:
    """Find the most recent run directory."""
    if not base_dir.exists():
        return None
    runs = sorted(base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if r.is_dir() and (r / "latest_checkpoint.json").exists():
            return r
    return None
