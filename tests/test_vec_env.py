# tests/test_vec_env.py
import numpy as np
import pytest

from pacman.utils.config import load_config
from pacman.env.pacman_env import PacmanEnv
from pacman.env.vec_env import VecEnv


@pytest.fixture
def config():
    return load_config()


class TestVecEnv:
    def test_observation_shapes(self, config):
        env = VecEnv(4, config)
        obs = env.reset(seed=42)
        assert obs["grid"].shape == (4, 8, 31, 28)
        assert obs["scalars"].shape == (4, 5)
        assert obs["grid"].dtype == np.float32
        assert obs["scalars"].dtype == np.float32

    def test_observation_values_in_range(self, config):
        env = VecEnv(4, config)
        obs = env.reset(seed=42)
        assert obs["grid"].min() >= 0.0
        assert obs["grid"].max() <= 1.0
        assert obs["scalars"].min() >= 0.0
        assert obs["scalars"].max() <= 1.0

    def test_legal_masks_shape(self, config):
        env = VecEnv(4, config)
        env.reset(seed=42)
        masks = env.get_legal_masks()
        assert masks.shape == (4, 4)
        assert masks.dtype == bool
        assert masks.any(axis=1).all()  # every env has at least 1 legal action

    def test_step_returns_correct_shapes(self, config):
        env = VecEnv(4, config)
        env.reset(seed=42)
        masks = env.get_legal_masks()
        actions = np.array([np.random.choice(np.where(m)[0]) for m in masks])
        obs, rewards, dones, infos = env.step(actions)
        assert obs["grid"].shape == (4, 8, 31, 28)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_auto_reset(self, config):
        """Run until at least one env finishes, verify it auto-resets."""
        config = dict(config)
        config["game"] = dict(config["game"])
        config["game"]["max_steps"] = 50  # force quick termination
        env = VecEnv(4, config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        seen_done = False
        for _ in range(100):
            masks = env.get_legal_masks()
            actions = np.array([rng.choice(np.where(m)[0]) for m in masks])
            obs, rewards, dones, infos = env.step(actions)
            if dones.any():
                seen_done = True
                # After auto-reset, obs should still be valid
                assert obs["grid"].shape == (4, 8, 31, 28)
                break
        assert seen_done

    def test_parity_with_single_env(self, config):
        """VecEnv(N=1) should produce identical results to PacmanEnv with same seed."""
        single = PacmanEnv(config, difficulty=0)
        vec = VecEnv(1, config, difficulty=0)

        single_obs, _ = single.reset(seed=100)
        vec_obs = vec.reset(seed=100)

        np.testing.assert_array_equal(single_obs["grid"], vec_obs["grid"][0])
        np.testing.assert_array_equal(single_obs["scalars"], vec_obs["scalars"][0])

        rng_single = np.random.default_rng(200)
        rng_vec = np.random.default_rng(200)
        for _ in range(50):
            mask_s = single.get_legal_mask()
            mask_v = vec.get_legal_masks()[0]
            np.testing.assert_array_equal(mask_s, mask_v)

            action = rng_single.choice(np.where(mask_s)[0])
            obs_s, r_s, term_s, _, _ = single.step(int(action))
            obs_v, r_v, d_v, _ = vec.step(np.array([action]))

            np.testing.assert_array_almost_equal(r_s, r_v[0])
            if term_s:
                break

    def test_set_difficulty(self, config):
        env = VecEnv(2, config, difficulty=0)
        env.reset(seed=42)
        env.set_difficulty(2)
        assert env.difficulty == 2
