"""Tests for observation builders."""

import numpy as np
import pytest

from src.engine.game import GameState
from src.engine.constants import GhostMode
from src.agents.observations import (
    build_pacman_observation, build_ghost_observation, get_observation_sizes,
)
from src.utils.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def game(config):
    g = GameState(config)
    g.reset()
    return g


class TestPacmanObservation:
    def test_shape(self, game):
        obs = build_pacman_observation(game)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs.shape) == 1
        assert obs.shape[0] > 30  # should be ~50 features

    def test_values_normalized(self, game):
        obs = build_pacman_observation(game)
        # Most values should be in reasonable range
        assert np.all(obs >= -2.0)
        assert np.all(obs <= 2.0)

    def test_no_privileged_frightened_info(self, game):
        """Pac-Man observation should NOT contain explicit 'ghosts are edible' signal.
        Ghost mode is a raw one-hot — Pac-Man must learn what it means."""
        obs_normal = build_pacman_observation(game)

        # Set all ghosts to frightened
        for ghost in game.ghosts:
            ghost.mode = GhostMode.FRIGHTENED
            ghost.frightened_timer = 30
        obs_frightened = build_pacman_observation(game)

        # Observations should differ (mode changed) but there's no dedicated
        # "edible" flag — just the raw mode one-hot
        assert not np.array_equal(obs_normal, obs_frightened)

    def test_consistent_size(self, game):
        """Observation size should not change between states."""
        obs1 = build_pacman_observation(game)
        game.step(0, [0, 0, 0, 0])
        obs2 = build_pacman_observation(game)
        assert obs1.shape == obs2.shape


class TestGhostObservation:
    def test_shape(self, game):
        for i in range(4):
            obs = build_ghost_observation(game, i)
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert len(obs.shape) == 1

    def test_different_ghosts_different_obs(self, game):
        """Each ghost should see different observations (different positions/IDs)."""
        obs0 = build_ghost_observation(game, 0)
        obs1 = build_ghost_observation(game, 1)
        assert not np.array_equal(obs0, obs1)

    def test_consistent_size_across_ghosts(self, game):
        sizes = [len(build_ghost_observation(game, i)) for i in range(4)]
        assert len(set(sizes)) == 1  # all same size


class TestObservationSizes:
    def test_get_sizes(self, config):
        pac_size, ghost_size = get_observation_sizes(config)
        assert pac_size > 0
        assert ghost_size > 0
        assert isinstance(pac_size, int)
        assert isinstance(ghost_size, int)
