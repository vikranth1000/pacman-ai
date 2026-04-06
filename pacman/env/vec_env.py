# pacman/env/vec_env.py
"""Vectorized Pac-Man environment — N games as batched NumPy operations."""
import numpy as np

from ..engine.constants import (
    Tile, GhostMode, GhostID, Direction, DIRECTION_DELTAS, OPPOSITE_DIRECTION,
    MAZE_ROWS, MAZE_COLS, NUM_GHOSTS, NUM_ACTIONS,
)
from ..engine.maze import load_initial_grid, compute_ghost_return_paths
from ..engine.maze_data import (
    PACMAN_START, GHOST_START_POSITIONS, FRUIT_POSITION,
)
from ..engine.entities import create_initial_state
from ..engine.game import step_game, get_legal_actions
from .pacman_env import NUM_CHANNELS, NUM_SCALARS


class VecEnv:
    """Vectorized Pac-Man environment stepping N games in parallel.

    Uses per-game step_game() internally with auto-reset.
    Supports frame stacking for temporal observations.
    """

    def __init__(self, num_envs: int, config: dict, difficulty: int = 0):
        self.num_envs = num_envs
        self.config = config
        self.difficulty = difficulty
        self._initial_grid = load_initial_grid()
        self._return_paths = compute_ghost_return_paths(self._initial_grid)
        self._states = []
        self._rngs = []

        # Frame stacking
        self.frame_stack = config["env"].get("frame_stack", 1)
        self._frame_buffer = None  # (N, frame_stack, C, H, W)

    def reset(self, seed: int | None = None) -> dict:
        base_seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._states = []
        self._rngs = []
        for i in range(self.num_envs):
            self._states.append(create_initial_state(self.config, self.difficulty))
            self._rngs.append(np.random.default_rng(base_seed + i))

        raw_obs = self._build_batch_obs()

        if self.frame_stack > 1:
            # Fill all frame slots with the initial observation
            self._frame_buffer = np.tile(
                raw_obs["grid"][:, np.newaxis],  # (N, 1, C, H, W)
                (1, self.frame_stack, 1, 1, 1),
            )

        return self._stack_obs(raw_obs)

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        """Step all environments. Auto-resets done envs.

        Returns: (obs_dict, rewards, dones, infos)
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = {
            "score": np.zeros(self.num_envs, dtype=np.int32),
            "pellets_eaten": np.zeros(self.num_envs, dtype=np.int32),
            "lives": np.zeros(self.num_envs, dtype=np.int32),
            "winner": [None] * self.num_envs,
            "level_cleared": np.zeros(self.num_envs, dtype=bool),
        }

        for i in range(self.num_envs):
            state, events, reward = step_game(
                self._states[i], int(actions[i]),
                self.config, self._return_paths, self._rngs[i],
            )
            rewards[i] = reward
            dones[i] = state.done
            infos["score"][i] = state.score
            infos["pellets_eaten"][i] = state.pellets_eaten
            infos["lives"][i] = state.pac_lives
            infos["winner"][i] = state.winner
            infos["level_cleared"][i] = state.winner == "pacman"

            # Auto-reset done environments
            if state.done:
                self._states[i] = create_initial_state(self.config, self.difficulty)

        raw_obs = self._build_batch_obs()

        # Update frame buffer
        if self.frame_stack > 1:
            self._frame_buffer[:, :-1] = self._frame_buffer[:, 1:]
            self._frame_buffer[:, -1] = raw_obs["grid"]
            # Reset frame buffer for done environments (new episode = no history)
            for i in range(self.num_envs):
                if dones[i]:
                    self._frame_buffer[i, :] = raw_obs["grid"][i]

        obs = self._stack_obs(raw_obs)
        return obs, rewards, dones, infos

    def get_legal_masks(self) -> np.ndarray:
        """Return (N, 4) bool mask of legal actions per env."""
        masks = np.zeros((self.num_envs, NUM_ACTIONS), dtype=bool)
        for i in range(self.num_envs):
            masks[i] = get_legal_actions(
                self._states[i].grid, self._states[i].pac_pos,
                prev_dir=int(self._states[i].pac_dir),
            )
        return masks

    def set_difficulty(self, difficulty: int) -> None:
        self.difficulty = difficulty
        for state in self._states:
            state.difficulty = difficulty

    def _stack_obs(self, raw_obs: dict) -> dict:
        """Stack frames if frame_stack > 1, otherwise return raw obs."""
        if self.frame_stack <= 1:
            return raw_obs
        stacked = self._frame_buffer.reshape(
            self.num_envs, -1, MAZE_ROWS, MAZE_COLS,
        )
        return {"grid": stacked.copy(), "scalars": raw_obs["scalars"]}

    def _build_batch_obs(self) -> dict:
        """Build raw (unstacked) batched observations: grid (N,8,31,28), scalars (N,5)."""
        grids = np.zeros((self.num_envs, NUM_CHANNELS, MAZE_ROWS, MAZE_COLS), dtype=np.float32)
        scalars = np.zeros((self.num_envs, NUM_SCALARS), dtype=np.float32)
        max_fright = self.config["game"]["frightened_duration"]

        for i, s in enumerate(self._states):
            grids[i, 0] = (s.grid == Tile.WALL)
            grids[i, 1, s.pac_pos[0], s.pac_pos[1]] = 1.0
            grids[i, 2] = (s.grid == Tile.PELLET)
            grids[i, 3] = (s.grid == Tile.POWER_PELLET)
            for g in range(NUM_GHOSTS):
                if not s.ghost_in_house[g] and s.ghost_mode[g] in (GhostMode.SCATTER, GhostMode.CHASE):
                    grids[i, 4, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0
                if not s.ghost_in_house[g] and s.ghost_mode[g] == GhostMode.FRIGHTENED:
                    grids[i, 5, s.ghost_pos[g, 0], s.ghost_pos[g, 1]] = 1.0
            grids[i, 6] = ((s.grid == Tile.GHOST_HOUSE) | (s.grid == Tile.GHOST_DOOR))
            if s.fruit_active:
                grids[i, 7, FRUIT_POSITION[0], FRUIT_POSITION[1]] = 1.0

            scalars[i] = [
                s.pac_power_timer / max(max_fright, 1),
                s.pac_lives / self.config["game"]["lives"],
                s.pac_ghosts_eaten / 4.0,
                s.pellets_eaten / max(s.total_pellets, 1),
                s.pac_dir / 3.0,
            ]

        return {"grid": grids, "scalars": scalars}
