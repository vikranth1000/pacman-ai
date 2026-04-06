"""Tests for the online dream loop components."""
import torch
import numpy as np
import pytest
from pathlib import Path
from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.world_model.world_model import WorldModel
from pacman.world_model.replay_buffer import EpisodeReplayBuffer
from pacman.training.dream_trainer import DreamTrainer, DreamPolicy


@pytest.fixture
def config():
    cfg = load_config()
    cfg["env"]["frame_stack"] = 1
    return cfg


@pytest.fixture
def small_wm():
    """A tiny world model for fast tests."""
    wm = WorldModel()
    return wm


class TestDreamTrainerReturnsResult:
    def test_train_returns_dict_with_best_score(self, small_wm, config, tmp_path):
        """DreamTrainer.train() should return a dict with best_score, best_update, best_path."""
        trainer = DreamTrainer(
            world_model=small_wm,
            config=config,
            device=torch.device("cpu"),
            imagination_horizon=3,
            num_imaginations=4,
            lr=1e-3,
            ppo_epochs=1,
        )
        result = trainer.train(
            total_updates=10,
            log_every=5,
            eval_every=5,
            eval_episodes=2,
            save_dir=tmp_path / "dream",
            patience=100,
        )
        assert isinstance(result, dict)
        assert "best_score" in result
        assert "best_update" in result
        assert "best_path" in result
        assert isinstance(result["best_score"], float)
        assert isinstance(result["best_update"], int)
        assert isinstance(result["best_path"], Path)
        assert result["best_path"].exists()
