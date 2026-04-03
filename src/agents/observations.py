"""Observation builders — convert game state to per-agent feature vectors."""

import numpy as np
from src.engine.constants import GhostMode, GhostID, MAZE_ROWS, MAZE_COLS
from src.engine.game import GameState


def _normalize_pos(row: int, col: int) -> tuple[float, float]:
    """Normalize position to [0, 1]."""
    return row / (MAZE_ROWS - 1), col / (MAZE_COLS - 1)


def _relative_direction(from_row: int, from_col: int,
                        to_row: int, to_col: int) -> tuple[float, float, float]:
    """Returns (dx_normalized, dy_normalized, manhattan_distance_normalized)."""
    dr = (to_row - from_row) / MAZE_ROWS
    dc = (to_col - from_col) / MAZE_COLS
    dist = (abs(to_row - from_row) + abs(to_col - from_col)) / (MAZE_ROWS + MAZE_COLS)
    return dr, dc, dist


def _ghost_mode_onehot(mode: GhostMode) -> list[float]:
    """One-hot encode ghost mode: [scatter, chase, frightened, eaten]."""
    vec = [0.0, 0.0, 0.0, 0.0]
    vec[int(mode)] = 1.0
    return vec


def build_pacman_observation(game: GameState) -> np.ndarray:
    """Build Pac-Man's observation vector (~50 features).

    Does NOT encode semantic meaning of ghost modes — only raw state.
    Pac-Man must learn through reward what modes mean.
    """
    pm = game.pacman
    features = []

    # Position (2)
    r, c = _normalize_pos(pm.row, pm.col)
    features.extend([r, c])

    # Wall sensors — legal moves in 4 directions (4)
    legal = game.get_legal_actions_pacman()
    for d in range(4):
        features.append(1.0 if d in legal else 0.0)

    # Per-ghost info: dx, dy, distance, mode one-hot (7 per ghost × 4 = 28)
    for ghost in game.ghosts:
        dr, dc, dist = _relative_direction(pm.row, pm.col, ghost.row, ghost.col)
        features.extend([dr, dc, dist])
        features.extend(_ghost_mode_onehot(ghost.mode))

    # Nearest pellet direction + distance (4)
    nearest_pellet = game.maze.find_nearest_pellet(pm.row, pm.col)
    if nearest_pellet is not None:
        dr, dc, dist = _relative_direction(pm.row, pm.col, nearest_pellet[0], nearest_pellet[1])
        features.extend([dr, dc, dist, 1.0])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # Nearest power pellet (4)
    nearest_power = game.maze.find_nearest_power_pellet(pm.row, pm.col)
    if nearest_power is not None:
        dr, dc, dist = _relative_direction(pm.row, pm.col, nearest_power[0], nearest_power[1])
        features.extend([dr, dc, dist, 1.0])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # Pellet density in 4 quadrants (4)
    density = game.maze.get_pellet_density(pm.row, pm.col)
    features.extend(density)

    # Game progress: pellets remaining fraction, lives normalized (2)
    total = game.maze.total_pellets + game.maze.total_power_pellets
    pellet_frac = game.maze.pellets_remaining / max(total, 1)
    lives_frac = pm.lives / pm.max_lives
    features.extend([pellet_frac, lives_frac])

    # Fruit info: active, direction (3)
    if game.fruit.active:
        dr, dc, dist = _relative_direction(pm.row, pm.col, game.fruit.row, game.fruit.col)
        features.extend([1.0, dr, dc])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def build_ghost_observation(game: GameState, ghost_idx: int) -> np.ndarray:
    """Build observation vector for a specific ghost (~40 features)."""
    ghost = game.ghosts[ghost_idx]
    pm = game.pacman
    features = []

    # Own position (2)
    r, c = _normalize_pos(ghost.row, ghost.col)
    features.extend([r, c])

    # Own mode one-hot (4)
    features.extend(_ghost_mode_onehot(ghost.mode))

    # Wall sensors (4)
    legal = game.get_legal_actions_ghost(ghost_idx)
    for d in range(4):
        features.append(1.0 if d in legal else 0.0)

    # Pac-Man info: dx, dy, distance, powered_up flag (4)
    dr, dc, dist = _relative_direction(ghost.row, ghost.col, pm.row, pm.col)
    features.extend([dr, dc, dist, 1.0 if pm.powered_up else 0.0])

    # Other ghosts info: dx, dy, distance, mode (7 per ghost × 3 = 21)
    for i, other in enumerate(game.ghosts):
        if i == ghost_idx:
            continue
        dr, dc, dist = _relative_direction(ghost.row, ghost.col, other.row, other.col)
        features.extend([dr, dc, dist])
        features.extend(_ghost_mode_onehot(other.mode))

    # Scatter target direction (2)
    sr, sc = ghost.scatter_target
    dr = (sr - ghost.row) / MAZE_ROWS
    dc = (sc - ghost.col) / MAZE_COLS
    features.extend([dr, dc])

    # Game state: pellets remaining fraction, frightened timer normalized (2)
    total = game.maze.total_pellets + game.maze.total_power_pellets
    pellet_frac = game.maze.pellets_remaining / max(total, 1)
    fright_frac = ghost.frightened_timer / max(game.frightened_duration, 1)
    features.extend([pellet_frac, fright_frac])

    return np.array(features, dtype=np.float32)


# Observation sizes (for network initialization)
PACMAN_OBS_SIZE = 49  # 2 + 4 + 28 + 4 + 4 + 4 + 2 + 3 = 51... let's compute dynamically

def get_observation_sizes(config: dict) -> tuple[int, int]:
    """Get actual observation sizes by building dummy observations."""
    game = GameState(config)
    game.reset()
    pac_obs = build_pacman_observation(game)
    ghost_obs = build_ghost_observation(game, 0)
    return len(pac_obs), len(ghost_obs)
