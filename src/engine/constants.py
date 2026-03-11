"""Game constants — directions, tile types, ghost modes, ghost identities."""

from enum import IntEnum


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# Movement deltas: (row_delta, col_delta)
DIRECTION_DELTAS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

OPPOSITE_DIRECTION = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


class Tile(IntEnum):
    WALL = 0
    EMPTY = 1
    PELLET = 2
    POWER_PELLET = 3
    GHOST_HOUSE = 4
    GHOST_DOOR = 5
    TUNNEL = 6


class GhostMode(IntEnum):
    SCATTER = 0
    CHASE = 1
    FRIGHTENED = 2
    EATEN = 3


class GhostID(IntEnum):
    BLINKY = 0
    PINKY = 1
    INKY = 2
    CLYDE = 3


GHOST_NAMES = {
    GhostID.BLINKY: "blinky",
    GhostID.PINKY: "pinky",
    GhostID.INKY: "inky",
    GhostID.CLYDE: "clyde",
}

# Scatter target corners (row, col) — classic Pac-Man
SCATTER_TARGETS = {
    GhostID.BLINKY: (0, 25),     # top-right
    GhostID.PINKY: (0, 2),       # top-left
    GhostID.INKY: (30, 27),      # bottom-right
    GhostID.CLYDE: (30, 0),      # bottom-left
}

# Ghost colors for rendering
GHOST_COLORS = {
    GhostID.BLINKY: (255, 0, 0),       # red
    GhostID.PINKY: (255, 184, 255),     # pink
    GhostID.INKY: (0, 255, 255),        # cyan
    GhostID.CLYDE: (255, 184, 82),      # orange
}

FRIGHTENED_COLOR = (33, 33, 222)        # blue
FRIGHTENED_BLINK_COLOR = (255, 255, 255)  # white

# Maze dimensions
MAZE_ROWS = 31
MAZE_COLS = 28

# Score values
PELLET_SCORE = 10
POWER_PELLET_SCORE = 50
GHOST_EAT_SCORES = [200, 400, 800, 1600]

NUM_ACTIONS = 4
