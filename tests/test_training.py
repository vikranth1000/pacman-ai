# tests/test_training.py
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from pacman.utils.config import load_config
from pacman.agents.networks import ActorCritic
from pacman.agents.ppo import PPO
from pacman.training.trainer import Trainer, get_device
from pacman.training.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def config():
    c = load_config()
    # Override for fast tests
    c["env"]["num_envs"] = 4
    c["ppo"]["rollout_steps"] = 16
    c["ppo"]["minibatch_size"] = 16
    c["ppo"]["num_epochs"] = 2
    c["training"]["total_updates"] = 5
    c["training"]["eval_every"] = 3
    c["training"]["eval_episodes"] = 2
    c["training"]["checkpoint_every"] = 2
    c["game"]["max_steps"] = 200
    return c


class TestTrainer:
    def test_short_training_completes(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=5)
            # Should complete without errors
            assert (Path(tmpdir) / "checkpoints" / "latest.pt").exists()

    def test_checkpoint_save_load(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=3)

            # Load checkpoint into fresh network
            network2 = ActorCritic()
            meta = load_checkpoint(Path(tmpdir) / "checkpoints", network2)
            assert meta["update"] >= 0

    def test_curriculum_advances(self, config):
        config["curriculum"]["phase_thresholds"] = [0, 2, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, Path(tmpdir))
            trainer.train(total_updates=5)
            assert trainer.curriculum_phase >= 1


class TestCheckpoint:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            network = ActorCritic()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            save_checkpoint(
                Path(tmpdir), 100, network, optimizer,
                {"mean": 1.0, "var": 2.0, "count": 100},
                curriculum_phase=1, config={"test": True},
            )
            network2 = ActorCritic()
            optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-3)
            meta = load_checkpoint(Path(tmpdir), network2, optimizer2)
            assert meta["update"] == 100
            assert meta["curriculum_phase"] == 1
