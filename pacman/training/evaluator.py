# pacman/training/evaluator.py
"""Greedy policy evaluation."""
import numpy as np
import torch

from ..env.pacman_env import PacmanEnv
from ..agents.networks import ActorCritic


class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.env = PacmanEnv(config, difficulty=2)  # always eval at full difficulty

    @torch.no_grad()
    def evaluate(
        self,
        network: ActorCritic,
        num_episodes: int,
        device: torch.device,
    ) -> dict:
        network.eval()
        scores = []
        steps_list = []
        ghosts_eaten = []
        cleared = 0

        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=ep)
            episode_ghosts = 0
            while True:
                grid_t = torch.as_tensor(obs["grid"][None], device=device)
                scalars_t = torch.as_tensor(obs["scalars"][None], device=device)
                mask = self.env.get_legal_mask()
                mask_t = torch.as_tensor(mask[None], device=device)
                logits, _ = network(grid_t, scalars_t, mask_t)
                action = logits.argmax(dim=-1).item()
                obs, reward, terminated, truncated, info = self.env.step(action)
                for event in info.get("events", []):
                    if event.startswith("eat_ghost"):
                        episode_ghosts += 1
                if terminated or truncated:
                    scores.append(info["score"])
                    steps_list.append(self.env.state.step_count)
                    ghosts_eaten.append(episode_ghosts)
                    if info.get("winner") == "pacman":
                        cleared += 1
                    break

        network.train()
        return {
            "level_clear_rate": cleared / max(num_episodes, 1),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
            "mean_ghosts_eaten": float(np.mean(ghosts_eaten)) if ghosts_eaten else 0.0,
        }
