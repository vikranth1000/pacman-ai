"""Tests for training loop, checkpointing, and data logging."""

import tempfile
import shutil
from pathlib import Path
import pytest

from src.engine.game import GameState
from src.engine.constants import GhostID, GHOST_NAMES
from src.agents.dqn_agent import DQNAgent
from src.agents.observations import get_observation_sizes
from src.training.trainer import Trainer
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.data.logger import DataLogger
from src.utils.config import load_config
from src.utils.seeding import get_device


@pytest.fixture
def config():
    cfg = load_config()
    # Speed up tests
    cfg["training"]["num_episodes"] = 3
    cfg["training"]["checkpoint_every"] = 2
    cfg["training"]["log_every"] = 1
    cfg["game"]["max_steps_per_episode"] = 100
    cfg["agent"]["min_replay_before_learn"] = 10
    cfg["agent"]["replay_buffer_size"] = 1000
    return cfg


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


class TestTrainer:
    def test_training_runs(self, config, tmp_dir):
        """Training loop completes without errors."""
        trainer = Trainer(config, run_dir=tmp_dir / "run_test")
        trainer.train(num_episodes=3)
        trainer.close()

    def test_metrics_logged(self, config, tmp_dir):
        """Metrics are written to SQLite after training."""
        run_dir = tmp_dir / "run_test"
        trainer = Trainer(config, run_dir=run_dir)
        trainer.train(num_episodes=3)
        trainer.close()

        # Check database exists and has data
        logger = DataLogger(run_dir / "metrics.db")
        episodes = logger.get_episodes(limit=10)
        assert len(episodes) == 3
        logger.close()

    def test_checkpoints_saved(self, config, tmp_dir):
        """Checkpoints are saved at configured intervals."""
        config["training"]["checkpoint_every"] = 2
        run_dir = tmp_dir / "run_test"
        trainer = Trainer(config, run_dir=run_dir)
        trainer.train(num_episodes=3)
        trainer.close()

        # Check checkpoint files exist
        assert (run_dir / "latest_checkpoint.json").exists()
        ckpt_dir = run_dir / "checkpoints"
        assert ckpt_dir.exists()

    def test_multi_game_auto_restart(self, config, tmp_dir):
        """Multiple games run consecutively without manual intervention."""
        run_dir = tmp_dir / "run_test"
        trainer = Trainer(config, run_dir=run_dir)
        trainer.train(num_episodes=5)
        trainer.close()

        logger = DataLogger(run_dir / "metrics.db")
        episodes = logger.get_episodes(limit=10)
        assert len(episodes) == 5
        logger.close()


class TestCheckpointing:
    def test_save_and_load(self, config, tmp_dir):
        """Agents can be saved and loaded."""
        device = get_device()
        pac_size, ghost_size = get_observation_sizes(config)

        agents = {}
        agents["pacman"] = DQNAgent("pacman", pac_size, config, device)
        for gid in GhostID:
            name = GHOST_NAMES[gid]
            agents[name] = DQNAgent(name, ghost_size, config, device)

        # Modify epsilon to verify it's saved
        agents["pacman"].epsilon = 0.42

        run_dir = tmp_dir / "ckpt_test"
        save_checkpoint(run_dir, 100, agents, config)

        # Create fresh agents and load
        new_agents = {}
        new_agents["pacman"] = DQNAgent("pacman", pac_size, config, device)
        for gid in GhostID:
            name = GHOST_NAMES[gid]
            new_agents[name] = DQNAgent(name, ghost_size, config, device)

        ep = load_checkpoint(run_dir, new_agents)
        assert ep == 100
        assert abs(new_agents["pacman"].epsilon - 0.42) < 1e-6


class TestDataLogger:
    def test_log_and_query(self, tmp_dir):
        db_path = tmp_dir / "test.db"
        logger = DataLogger(db_path)

        logger.log_episode(0, "pacman", 500, 200, 100, 2, 1, 2, True)
        logger.log_episode(1, "ghosts", 300, 150, 80, 0, 0, 0, False)

        episodes = logger.get_episodes()
        assert len(episodes) == 2

        win_rates = logger.get_win_rates()
        assert win_rates["pacman"] == 0.5
        assert win_rates["ghosts"] == 0.5

        logger.close()

    def test_agent_metrics(self, tmp_dir):
        db_path = tmp_dir / "test.db"
        logger = DataLogger(db_path)

        logger.log_episode(0, "pacman", 500, 200, 100, 2, 1, 2, True)
        logger.log_agent_metrics(0, "pacman", 15.0, 2.5, 0.5, 0.1, 200)

        metrics = logger.get_agent_metrics("pacman")
        assert len(metrics) == 1
        assert metrics[0]["total_reward"] == 15.0

        logger.close()
