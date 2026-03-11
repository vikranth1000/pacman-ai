#!/usr/bin/env python3
"""Watch AI agents play Pac-Man with Pygame GUI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seeding import set_seeds, get_device
from src.engine.game import GameState
from src.engine.constants import GhostID, GHOST_NAMES
from src.agents.dqn_agent import DQNAgent
from src.agents.observations import (
    build_pacman_observation, build_ghost_observation, get_observation_sizes,
)
from src.training.checkpoint import load_checkpoint
from src.gui.renderer import GameRenderer
from src.data.logger import DataLogger


def main():
    parser = argparse.ArgumentParser(description="Watch Pac-Man AI gameplay")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to run directory with checkpoints (random agents if not provided)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file (used when no run-dir)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Max episodes (0 = unlimited)")
    args = parser.parse_args()

    # Load config
    if args.run_dir:
        run_dir = Path(args.run_dir)
        config = load_config(run_dir / "config.yaml")
    else:
        config = load_config(args.config)
        run_dir = None

    device = get_device()
    set_seeds(config.get("seed", 42))

    # Initialize game
    game = GameState(config)
    pac_obs_size, ghost_obs_size = get_observation_sizes(config)

    # Create agents
    agents = {}
    agents["pacman"] = DQNAgent("pacman", pac_obs_size, config, device)
    for ghost_id in GhostID:
        name = GHOST_NAMES[ghost_id]
        agents[name] = DQNAgent(name, ghost_obs_size, config, device)

    # Load trained weights if available
    if run_dir:
        try:
            ep = load_checkpoint(run_dir, agents)
            print(f"Loaded checkpoint from episode {ep}")
            for agent in agents.values():
                agent.epsilon = 0.0  # no exploration in watch mode
        except FileNotFoundError:
            print("No checkpoint found, using random agents")

    # Initialize renderer
    gui_cfg = config.get("gui", {})
    tile_size = gui_cfg.get("tile_size", 20)
    renderer = GameRenderer(tile_size=tile_size)

    # Get win rate data if available
    win_rates = None
    if run_dir and (run_dir / "metrics.db").exists():
        logger = DataLogger(run_dir / "metrics.db")
        win_rates = logger.get_win_rates()
        logger.close()

    episode = 0
    running = True

    while running:
        game.reset()
        episode += 1
        print(f"Episode {episode}")

        while not game.done and running:
            # Get observations and actions
            pac_obs = build_pacman_observation(game)
            pac_legal = game.get_legal_actions_pacman()
            pac_action = agents["pacman"].act(pac_obs, pac_legal, training=False)

            ghost_actions = []
            for i in range(4):
                name = GHOST_NAMES[GhostID(i)]
                ghost_obs = build_ghost_observation(game, i)
                legal = game.get_legal_actions_ghost(i)
                action = agents[name].act(ghost_obs, legal, training=False)
                ghost_actions.append(action)

            # Step
            game.step(pac_action, ghost_actions)

            # Render
            running = renderer.render(game, episode, agents, win_rates)
            renderer.tick(args.fps)

        if game.done:
            print(f"  Winner: {game.winner} | Score: {game.pacman.score} | "
                  f"Steps: {game.step_count}")

        if args.episodes > 0 and episode >= args.episodes:
            break

    renderer.close()


if __name__ == "__main__":
    main()
