"""Observation builders — convert game state to per-agent feature vectors.

Enhanced with corridor look-ahead, direction awareness, and spatial reasoning.
"""

import numpy as np
from src.engine.constants import Direction, Tile, GhostMode, GhostID, DIRECTION_DELTAS, MAZE_ROWS, MAZE_COLS
from src.engine.game import GameState

LOOK_AHEAD_DIST = 8


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


def _direction_onehot(direction: Direction) -> list[float]:
    """One-hot encode direction: [up, down, left, right]."""
    vec = [0.0, 0.0, 0.0, 0.0]
    vec[int(direction)] = 1.0
    return vec


def _scan_corridor_pacman(game: GameState, row: int, col: int, direction: int) -> list[float]:
    """Scan straight corridor from Pac-Man's position.

    Returns 5 features:
        - wall_dist: normalized distance to nearest wall
        - pellet_density: pellet count in corridor normalized
        - ghost_danger: proximity of nearest non-frightened ghost (1=adjacent, 0=none)
        - ghost_opportunity: proximity of nearest frightened ghost (1=adjacent, 0=none)
        - has_junction: whether an escape junction exists in corridor
    """
    dr, dc = DIRECTION_DELTAS[Direction(direction)]
    max_dist = LOOK_AHEAD_DIST
    wall_dist = max_dist
    pellet_count = 0
    ghost_danger = 0.0
    ghost_opportunity = 0.0
    has_junction = 0.0

    for i in range(1, max_dist + 1):
        nr, nc = game.maze.wrap_position(row + dr * i, col + dc * i)
        if not game.maze.is_walkable_for_pacman(nr, nc):
            wall_dist = i
            break

        tile = game.maze.get_tile(nr, nc)
        if tile in (Tile.PELLET, Tile.POWER_PELLET):
            pellet_count += 1

        # Check for ghosts (only record first one found)
        if ghost_danger == 0.0 and ghost_opportunity == 0.0:
            for g in game.ghosts:
                if not g.in_ghost_house and not g.is_eaten and g.row == nr and g.col == nc:
                    proximity = 1.0 - (i - 1) / max_dist
                    if g.is_frightened:
                        ghost_opportunity = proximity
                    else:
                        ghost_danger = proximity
                    break

        # Check for junction (3+ walkable neighbors = escape route)
        if has_junction == 0.0:
            walkable = 0
            for d in Direction:
                ddr, ddc = DIRECTION_DELTAS[d]
                wr, wc = game.maze.wrap_position(nr + ddr, nc + ddc)
                if game.maze.is_walkable_for_pacman(wr, wc):
                    walkable += 1
            if walkable >= 3:
                has_junction = 1.0

    return [
        wall_dist / max_dist,
        pellet_count / max_dist,
        ghost_danger,
        ghost_opportunity,
        has_junction,
    ]


def _scan_corridor_ghost(game: GameState, row: int, col: int, direction: int) -> list[float]:
    """Scan straight corridor from ghost's position.

    Returns 3 features:
        - wall_dist: normalized distance to nearest wall
        - pacman_proximity: proximity of Pac-Man if visible (1=adjacent, 0=not visible)
        - has_junction: whether a junction exists in corridor
    """
    dr, dc = DIRECTION_DELTAS[Direction(direction)]
    max_dist = LOOK_AHEAD_DIST
    wall_dist = max_dist
    pacman_proximity = 0.0
    has_junction = 0.0

    pm = game.pacman
    for i in range(1, max_dist + 1):
        nr, nc = game.maze.wrap_position(row + dr * i, col + dc * i)
        if not game.maze.is_walkable(nr, nc):
            wall_dist = i
            break

        if pacman_proximity == 0.0 and pm.row == nr and pm.col == nc:
            pacman_proximity = 1.0 - (i - 1) / max_dist

        if has_junction == 0.0:
            walkable = 0
            for d in Direction:
                ddr, ddc = DIRECTION_DELTAS[d]
                wr, wc = game.maze.wrap_position(nr + ddr, nc + ddc)
                if game.maze.is_walkable(wr, wc):
                    walkable += 1
            if walkable >= 3:
                has_junction = 1.0

    return [
        wall_dist / max_dist,
        pacman_proximity,
        has_junction,
    ]


def build_pacman_observation(game: GameState) -> np.ndarray:
    """Build Pac-Man's enhanced observation vector.

    Features:
        Position (2), direction (4), wall sensors (4), corridor look-ahead (20),
        per-ghost info (28), nearest pellet (4), nearest power pellet (4),
        pellet density (4), game progress (2), fruit (3), escape routes (1).

    Does NOT encode semantic meaning of ghost modes — only raw state.
    Pac-Man must learn through reward what modes mean.
    """
    pm = game.pacman
    features = []

    # === Spatial Awareness ===

    # Position (2)
    r, c = _normalize_pos(pm.row, pm.col)
    features.extend([r, c])

    # Direction of travel one-hot (4)
    features.extend(_direction_onehot(pm.direction))

    # Wall sensors — legal moves in 4 directions (4)
    legal = game.get_legal_actions_pacman()
    for d in range(4):
        features.append(1.0 if d in legal else 0.0)

    # Corridor look-ahead — 4 directions × 5 features = 20
    # (wall distance, pellet density, ghost danger, ghost opportunity, junction)
    for d in range(4):
        features.extend(_scan_corridor_pacman(game, pm.row, pm.col, d))

    # === Ghost Information ===

    # Per-ghost: dx, dy, distance, mode one-hot (7 per ghost × 4 = 28)
    for ghost in game.ghosts:
        dr, dc, dist = _relative_direction(pm.row, pm.col, ghost.row, ghost.col)
        features.extend([dr, dc, dist])
        features.extend(_ghost_mode_onehot(ghost.mode))

    # === Navigation Targets ===

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

    # === Game State ===

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

    # Escape routes: number of legal moves normalized (1)
    features.append(len(legal) / 4.0)

    return np.array(features, dtype=np.float32)


def build_ghost_observation(game: GameState, ghost_idx: int) -> np.ndarray:
    """Build enhanced observation vector for a specific ghost.

    Features:
        Position (2), direction (4), mode (4), wall sensors (4),
        corridor look-ahead (12), Pac-Man info (8), other ghosts (21),
        scatter target (2), game state (2), Pac-Man escape routes (1).
    """
    ghost = game.ghosts[ghost_idx]
    pm = game.pacman
    features = []

    # === Own State ===

    # Own position (2)
    r, c = _normalize_pos(ghost.row, ghost.col)
    features.extend([r, c])

    # Own direction one-hot (4)
    features.extend(_direction_onehot(ghost.direction))

    # Own mode one-hot (4)
    features.extend(_ghost_mode_onehot(ghost.mode))

    # Wall sensors (4)
    legal = game.get_legal_actions_ghost(ghost_idx)
    for d in range(4):
        features.append(1.0 if d in legal else 0.0)

    # Corridor look-ahead — 4 directions × 3 features = 12
    # (wall distance, Pac-Man proximity, junction)
    for d in range(4):
        features.extend(_scan_corridor_ghost(game, ghost.row, ghost.col, d))

    # === Target Information ===

    # Pac-Man info: dx, dy, distance, powered_up, direction one-hot (8)
    dr, dc, dist = _relative_direction(ghost.row, ghost.col, pm.row, pm.col)
    features.extend([dr, dc, dist, 1.0 if pm.powered_up else 0.0])
    features.extend(_direction_onehot(pm.direction))

    # === Team Information ===

    # Other ghosts info: dx, dy, distance, mode (7 per ghost × 3 = 21)
    for i, other in enumerate(game.ghosts):
        if i == ghost_idx:
            continue
        dr, dc, dist = _relative_direction(ghost.row, ghost.col, other.row, other.col)
        features.extend([dr, dc, dist])
        features.extend(_ghost_mode_onehot(other.mode))

    # === Strategic State ===

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

    # Pac-Man escape routes: how cornered is Pac-Man? (1)
    pac_legal = game.get_legal_actions_pacman()
    features.append(len(pac_legal) / 4.0)

    return np.array(features, dtype=np.float32)


def get_observation_sizes(config: dict) -> tuple[int, int]:
    """Get actual observation sizes by building dummy observations."""
    game = GameState(config)
    game.reset()
    pac_obs = build_pacman_observation(game)
    ghost_obs = build_ghost_observation(game, 0)
    return len(pac_obs), len(ghost_obs)
