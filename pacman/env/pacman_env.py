# pacman/env/pacman_env.py
"""Single Pac-Man environment with Gymnasium-compatible interface."""
import numpy as np

from ..engine.constants import (
    Tile, GhostMode, MAZE_ROWS, MAZE_COLS, NUM_GHOSTS,
)
from ..engine.maze import load_initial_grid, compute_ghost_return_paths
from ..engine.maze_data import FRUIT_POSITION
from ..engine.entities import create_initial_state, GameState
from ..engine.game import step_game, get_legal_actions

NUM_CHANNELS = 8
NUM_SCALARS = 5


class PacmanEnv:
    """Single Pac-Man environment for evaluation and visualization."""

    def __init__(self, config: dict, difficulty: int = 0):
        self.config = config
        self.difficulty = difficulty
        self._initial_grid = load_initial_grid()
        self._return_paths = compute_ghost_return_paths(self._initial_grid)
        self._state: GameState | None = None
        self._rng = np.random.default_rng()

        # Frame stacking
        self.frame_stack = config["env"].get("frame_stack", 1)
        self._frame_buffer = None  # (frame_stack, C, H, W)

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = create_initial_state(self.config, self.difficulty)
        raw_obs = self._build_obs()

        if self.frame_stack > 1:
            self._frame_buffer = np.tile(
                raw_obs["grid"][np.newaxis],  # (1, C, H, W)
                (self.frame_stack, 1, 1, 1),
            )

        return self._stack_obs(raw_obs), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        state, events, reward = step_game(
            self._state, action, self.config, self._return_paths, self._rng,
        )
        self._state = state
        raw_obs = self._build_obs()

        if self.frame_stack > 1:
            self._frame_buffer[:-1] = self._frame_buffer[1:]
            self._frame_buffer[-1] = raw_obs["grid"]

        obs = self._stack_obs(raw_obs)
        terminated = state.done
        truncated = False
        info = {
            "score": state.score,
            "pellets_eaten": state.pellets_eaten,
            "lives": state.pac_lives,
            "winner": state.winner,
            "events": events,
        }
        return obs, reward, terminated, truncated, info

    def get_legal_mask(self) -> np.ndarray:
        return get_legal_actions(
            self._state.grid, self._state.pac_pos,
            prev_dir=int(self._state.pac_dir),
        )

    @property
    def state(self) -> GameState:
        return self._state

    def _stack_obs(self, raw_obs: dict) -> dict:
        """Stack frames if frame_stack > 1, otherwise return raw obs."""
        if self.frame_stack <= 1:
            return raw_obs
        stacked = self._frame_buffer.reshape(-1, MAZE_ROWS, MAZE_COLS)
        return {"grid": stacked.copy(), "scalars": raw_obs["scalars"]}

    def _build_obs(self) -> dict:
        """Build raw (unstacked) 8-channel grid + 5 scalars observation."""
        s = self._state
        grid = np.zeros((NUM_CHANNELS, MAZE_ROWS, MAZE_COLS), dtype=np.float32)

        # Ch 0: Walls
        grid[0] = (s.grid == Tile.WALL).astype(np.float32)
        # Ch 1: Pac-Man position
        grid[1, s.pac_pos[0], s.pac_pos[1]] = 1.0
        # Ch 2: Pellets
        grid[2] = (s.grid == Tile.PELLET).astype(np.float32)
        # Ch 3: Power pellets
        grid[3] = (s.grid == Tile.POWER_PELLET).astype(np.float32)
        # Ch 4: Dangerous ghosts (scatter/chase)
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] in (GhostMode.SCATTER, GhostMode.CHASE):
                grid[4, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0
        # Ch 5: Edible ghosts (frightened)
        for i in range(NUM_GHOSTS):
            if not s.ghost_in_house[i] and s.ghost_mode[i] == GhostMode.FRIGHTENED:
                grid[5, s.ghost_pos[i, 0], s.ghost_pos[i, 1]] = 1.0
        # Ch 6: Ghost house
        grid[6] = ((s.grid == Tile.GHOST_HOUSE) | (s.grid == Tile.GHOST_DOOR)).astype(np.float32)
        # Ch 7: Fruit
        if s.fruit_active:
            grid[7, FRUIT_POSITION[0], FRUIT_POSITION[1]] = 1.0

        # Scalars
        max_fright = self.config["game"]["frightened_duration"]
        scalars = np.array([
            s.pac_power_timer / max(max_fright, 1),
            s.pac_lives / self.config["game"]["lives"],
            s.pac_ghosts_eaten / 4.0,
            s.pellets_eaten / max(s.total_pellets, 1),
            s.pac_dir / 3.0,
        ], dtype=np.float32)

        return {"grid": grid, "scalars": scalars}
