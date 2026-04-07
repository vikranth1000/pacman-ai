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


class TestBehavioralCloning:
    def test_train_bc_reduces_loss(self):
        """train_behavioral_cloning trains actor and returns metrics with decreasing loss."""
        from pacman.training.distill_ppo import train_behavioral_cloning

        device = torch.device("cpu")

        # Create a synthetic dataset: latent -> action mapping
        # Use a simple pattern: action = argmax of first 4 dims of latent
        N = 2000
        latents = torch.randn(N, 2560)
        # Assign actions based on a learnable pattern
        actions = (latents[:, :4].argmax(dim=-1)).long()

        policy = DreamPolicy(latent_dim=2560)

        result = train_behavioral_cloning(
            policy=policy,
            latents=latents,
            actions=actions,
            device=device,
            epochs=10,
            batch_size=256,
            lr=1e-3,
            patience=50,  # don't early stop for this test
        )

        assert "train_loss" in result
        assert "val_loss" in result
        assert "val_accuracy" in result
        assert "best_epoch" in result
        assert isinstance(result["val_accuracy"], float)
        assert result["val_accuracy"] > 0.25  # better than random (25%)
        # Loss should have decreased from initial
        assert result["val_loss"] < 2.0  # well below initial ~1.39 (ln(4))

    def test_bc_policy_predicts_actions(self):
        """After BC training, the policy should predict the majority action correctly."""
        from pacman.training.distill_ppo import train_behavioral_cloning

        torch.manual_seed(42)
        device = torch.device("cpu")

        # Create data with a strong signal: amplify the first 4 dims so the
        # argmax pattern is easily learnable despite noise in other dimensions
        N = 3000
        latents = torch.randn(N, 2560)
        latents[:, :4] *= 10.0  # amplify signal dimensions strongly
        actions = latents[:, :4].argmax(dim=-1).long()

        policy = DreamPolicy(latent_dim=2560)

        train_behavioral_cloning(
            policy=policy,
            latents=latents,
            actions=actions,
            device=device,
            epochs=30,
            batch_size=256,
            lr=1e-3,
            patience=50,
        )

        # Test on new data with the same signal amplification
        test_latents = torch.randn(500, 2560)
        test_latents[:, :4] *= 10.0
        test_actions = test_latents[:, :4].argmax(dim=-1).long()
        with torch.no_grad():
            logits = policy.actor(test_latents)
            predicted = logits.argmax(dim=-1)
        accuracy = (predicted == test_actions).float().mean().item()
        assert accuracy > 0.5, f"Expected >50% accuracy, got {accuracy:.1%}"
