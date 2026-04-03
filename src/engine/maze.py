"""Maze management — loading, querying walls/pellets, tunnel wrapping."""

import copy
import numpy as np
from src.engine.constants import Tile, Direction, DIRECTION_DELTAS, MAZE_ROWS, MAZE_COLS
from src.engine.maze_data import CLASSIC_MAZE, TUNNEL_POSITIONS


class Maze:
    """Manages the maze grid, pellet state, and spatial queries."""

    def __init__(self):
        self.grid = np.array(copy.deepcopy(CLASSIC_MAZE), dtype=np.int8)
        self.rows = MAZE_ROWS
        self.cols = MAZE_COLS
        self.total_pellets = 0
        self.total_power_pellets = 0
        self.pellets_remaining = 0
        self._count_pellets()

    def _count_pellets(self):
        self.total_pellets = int(np.sum(self.grid == Tile.PELLET))
        self.total_power_pellets = int(np.sum(self.grid == Tile.POWER_PELLET))
        self.pellets_remaining = self.total_pellets + self.total_power_pellets

    def reset(self):
        """Reset maze to initial state with all pellets."""
        self.grid = np.array(copy.deepcopy(CLASSIC_MAZE), dtype=np.int8)
        self._count_pellets()

    def is_wall(self, row: int, col: int) -> bool:
        """Check if position is a wall or out of bounds."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True
        return self.grid[row, col] == Tile.WALL

    def is_walkable(self, row: int, col: int) -> bool:
        """Check if a position can be walked on (not wall, not ghost house interior for Pac-Man)."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        tile = self.grid[row, col]
        return tile != Tile.WALL

    def is_walkable_for_pacman(self, row: int, col: int) -> bool:
        """Pac-Man cannot enter ghost house or ghost door."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        tile = self.grid[row, col]
        return tile not in (Tile.WALL, Tile.GHOST_HOUSE, Tile.GHOST_DOOR)

    def is_walkable_for_ghost(self, row: int, col: int, is_eaten: bool = False) -> bool:
        """Ghosts can walk on most tiles. Only eaten ghosts can re-enter ghost door."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        tile = self.grid[row, col]
        if tile == Tile.WALL:
            return False
        if tile == Tile.GHOST_DOOR and not is_eaten:
            return True  # ghosts can pass through door when exiting
        return True

    def get_tile(self, row: int, col: int) -> Tile:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return Tile.WALL
        return Tile(self.grid[row, col])

    def eat_pellet(self, row: int, col: int) -> Tile | None:
        """Try to eat pellet at position. Returns tile type if eaten, None otherwise."""
        tile = self.get_tile(row, col)
        if tile in (Tile.PELLET, Tile.POWER_PELLET):
            self.grid[row, col] = Tile.EMPTY
            self.pellets_remaining -= 1
            return tile
        return None

    def get_legal_directions(self, row: int, col: int, for_pacman: bool = True) -> list[Direction]:
        """Get list of directions that lead to walkable tiles."""
        legal = []
        for d in Direction:
            dr, dc = DIRECTION_DELTAS[d]
            nr, nc = row + dr, col + dc
            # Handle tunnel wrapping
            nr, nc = self.wrap_position(nr, nc)
            if for_pacman:
                if self.is_walkable_for_pacman(nr, nc):
                    legal.append(d)
            else:
                if self.is_walkable(nr, nc):
                    legal.append(d)
        return legal

    def wrap_position(self, row: int, col: int) -> tuple[int, int]:
        """Handle tunnel wrapping."""
        if col < 0:
            col = self.cols - 1
        elif col >= self.cols:
            col = 0
        return row, col

    def find_nearest_pellet(self, row: int, col: int) -> tuple[int, int] | None:
        """Find nearest pellet by manhattan distance (fast approximation)."""
        return self._nearest_by_manhattan(row, col, {Tile.PELLET, Tile.POWER_PELLET})

    def find_nearest_power_pellet(self, row: int, col: int) -> tuple[int, int] | None:
        """Find nearest power pellet by manhattan distance."""
        return self._nearest_by_manhattan(row, col, {Tile.POWER_PELLET})

    def _nearest_by_manhattan(self, row: int, col: int, target_tiles: set) -> tuple[int, int] | None:
        """Find nearest target tile by manhattan distance. Fast O(n) scan."""
        mask = np.isin(self.grid, list(target_tiles))
        positions = np.argwhere(mask)
        if len(positions) == 0:
            return None
        # Manhattan distance with tunnel wrapping consideration
        dr = np.abs(positions[:, 0] - row)
        dc_raw = np.abs(positions[:, 1] - col)
        dc = np.minimum(dc_raw, self.cols - dc_raw)  # tunnel wrap
        dists = dr + dc
        idx = np.argmin(dists)
        return (int(positions[idx, 0]), int(positions[idx, 1]))

    def get_pellet_density(self, row: int, col: int) -> tuple[float, float, float, float]:
        """Count pellets in 4 quadrants relative to position. Vectorized."""
        total = max(self.pellets_remaining, 1)
        pellet_mask = (self.grid == Tile.PELLET) | (self.grid == Tile.POWER_PELLET)
        positions = np.argwhere(pellet_mask)
        if len(positions) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        rows, cols = positions[:, 0], positions[:, 1]
        up = rows <= row
        down = ~up
        left = cols <= col
        right = ~left

        up_left = int(np.sum(up & left))
        up_right = int(np.sum(up & right))
        down_left = int(np.sum(down & left))
        down_right = int(np.sum(down & right))

        return (up_left / total, up_right / total, down_left / total, down_right / total)
