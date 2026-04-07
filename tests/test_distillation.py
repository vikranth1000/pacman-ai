"""Tests for PPO behavior distillation into dream agent."""
import torch
import numpy as np
import pytest
from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.agents.networks import ActorCritic
from pacman.training.dream_trainer import DreamPolicy


@pytest.fixture
def config():
    cfg = load_config()
    cfg["env"]["frame_stack"] = 1
    return cfg


class TestDistillationDataCollection:
    def test_collect_returns_latents_and_actions(self, config):
        """collect_distillation_data returns a dict with latents and actions tensors."""
        from pacman.training.distill_ppo import collect_distillation_data

        device = torch.device("cpu")

        ppo_net = ActorCritic(
            grid_channels=8 * 4,  # 4-frame stack
            num_scalars=5,
        )
        ppo_net.eval()

        wm = WorldModel()
        wm.eval()

        result = collect_distillation_data(
            ppo_network=ppo_net,
            world_model=wm,
            config=config,
            device=device,
            num_episodes=3,
            difficulty=0,
        )

        assert "latents" in result
        assert "actions" in result
        assert isinstance(result["latents"], torch.Tensor)
        assert isinstance(result["actions"], torch.Tensor)
        assert result["latents"].shape[1] == 2560
        assert result["actions"].shape[0] == result["latents"].shape[0]
        assert result["latents"].shape[0] > 0
        assert result["actions"].min() >= 0
        assert result["actions"].max() <= 3
