"""Tests for maze loading, walls, pellets, and spatial queries."""

import pytest
from src.engine.maze import Maze
from src.engine.constants import Tile, Direction


class TestMaze:
    def setup_method(self):
        self.maze = Maze()

    def test_maze_dimensions(self):
        assert self.maze.rows == 31
        assert self.maze.cols == 28

    def test_corners_are_walls(self):
        assert self.maze.is_wall(0, 0)
        assert self.maze.is_wall(0, 27)
        assert self.maze.is_wall(30, 0)
        assert self.maze.is_wall(30, 27)

    def test_pellet_count(self):
        assert self.maze.total_pellets > 0
        assert self.maze.total_power_pellets == 4

    def test_eat_pellet(self):
        # Find a pellet
        initial_remaining = self.maze.pellets_remaining
        # Row 1, col 1 should be a pellet in the classic maze
        result = self.maze.eat_pellet(1, 1)
        assert result == Tile.PELLET
        assert self.maze.pellets_remaining == initial_remaining - 1
        # Eating again returns None
        result = self.maze.eat_pellet(1, 1)
        assert result is None

    def test_eat_power_pellet(self):
        # Row 3, col 1 should be a power pellet
        result = self.maze.eat_pellet(3, 1)
        assert result == Tile.POWER_PELLET

    def test_wall_detection(self):
        assert self.maze.is_wall(0, 0)
        assert not self.maze.is_wall(1, 1)

    def test_out_of_bounds_is_wall(self):
        assert self.maze.is_wall(-1, 0)
        assert self.maze.is_wall(0, -1)
        assert self.maze.is_wall(31, 0)
        assert self.maze.is_wall(0, 28)

    def test_legal_directions(self):
        # Center of a corridor should have at least 2 legal directions
        legal = self.maze.get_legal_directions(1, 1, for_pacman=True)
        assert len(legal) >= 1

    def test_tunnel_wrapping(self):
        r, c = self.maze.wrap_position(14, -1)
        assert c == 27
        r, c = self.maze.wrap_position(14, 28)
        assert c == 0

    def test_reset(self):
        self.maze.eat_pellet(1, 1)
        self.maze.reset()
        assert self.maze.get_tile(1, 1) == Tile.PELLET

    def test_find_nearest_pellet(self):
        result = self.maze.find_nearest_pellet(1, 1)
        assert result is not None

    def test_pellet_density(self):
        density = self.maze.get_pellet_density(15, 14)
        assert len(density) == 4
        assert sum(density) > 0
        assert all(0 <= d <= 1 for d in density)

    def test_pacman_cannot_enter_ghost_house(self):
        assert not self.maze.is_walkable_for_pacman(13, 14)  # ghost house interior
