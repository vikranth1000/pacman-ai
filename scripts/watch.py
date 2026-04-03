# scripts/watch.py
"""Watch a trained Pac-Man agent play in Pygame."""
import argparse
from pathlib import Path

import numpy as np
import torch

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.env.pacman_env import PacmanEnv
from pacman.training.checkpoint import load_checkpoint
from pacman.training.trainer import get_device
from pacman.viz.renderer import GameRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)

    network = None
    if args.run_dir:
        network = ActorCritic()
        load_checkpoint(Path(args.run_dir) / "checkpoints", network, filename=args.checkpoint)
        network.to(device).eval()

    env = PacmanEnv(config, difficulty=2)
    renderer = GameRenderer(config)
    obs, _ = env.reset(seed=args.seed)
    paused = False

    while True:
        action_probs = None
        value = None

        if network is not None:
            with torch.no_grad():
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                mask = env.get_legal_mask()
                mask_t = torch.as_tensor(mask[None], device=device)
                logits, val = network(grid_t, scalars_t, mask_t)
                probs = torch.softmax(logits, dim=-1)
                action_probs = probs[0].cpu().numpy()
                value = val[0].item()
                action = logits.argmax(dim=-1).item()
        else:
            mask = env.get_legal_mask()
            action = np.random.choice(np.where(mask)[0])

        agent_info = {"action_probs": action_probs, "value": value}
        running, step_requested = renderer.render(env.state, agent_info)
        if not running:
            break

        if not paused or step_requested:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, _ = env.reset()

        renderer.tick(args.fps)

    renderer.close()


if __name__ == "__main__":
    main()
