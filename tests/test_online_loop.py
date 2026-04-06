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
from pacman.training.wm_trainer import WMTrainer


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


def _make_small_buffer(config, num_episodes=5, steps_per_ep=50):
    """Collect a few short episodes with random actions."""
    env = PacmanEnv(config, difficulty=0)
    buf = EpisodeReplayBuffer(max_episodes=100)
    for i in range(num_episodes):
        env.reset(seed=i)
        grids, scalars, actions, rewards, dones = [], [], [], [], []
        for step in range(steps_per_ep):
            raw = env._build_obs()
            grids.append(raw["grid"])
            scalars.append(raw["scalars"])
            mask = env.get_legal_mask()
            legal = np.where(mask)[0]
            action = int(np.random.choice(legal))
            _, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break
        buf.add_episode({
            "grid": torch.as_tensor(np.stack(grids)),
            "scalars": torch.as_tensor(np.stack(scalars)),
            "action": torch.tensor(actions, dtype=torch.long),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.bool),
        })
    return buf


class TestWMTrainerFineTune:
    def test_fine_tune_loads_checkpoint_and_trains(self, config, tmp_path):
        """fine_tune() should load existing WM checkpoint and train with lower LR."""
        device = torch.device("cpu")
        buf = _make_small_buffer(config)

        # 1. Train initial model for 2 steps to get a checkpoint
        wm = WorldModel().to(device)
        trainer = WMTrainer(wm, buf, device, lr=3e-4, seq_len=10, batch_size=2)
        save_dir = tmp_path / "wm"
        trainer.train(total_steps=2, log_every=1, save_every=2, save_dir=save_dir)

        checkpoint_path = save_dir / "world_model_latest.pt"
        assert checkpoint_path.exists()

        # 2. Fine-tune from checkpoint
        wm2, trainer2 = WMTrainer.fine_tune(
            checkpoint_path=checkpoint_path,
            replay_buffer=buf,
            device=device,
            lr=1e-4,
            seq_len=10,
            batch_size=2,
        )
        assert isinstance(wm2, WorldModel)
        assert isinstance(trainer2, WMTrainer)

        # 3. Run a few training steps — should not error
        trainer2.train(total_steps=2, log_every=1, save_every=2, save_dir=save_dir)
