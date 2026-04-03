"""Game entities — Pac-Man, Ghost, Fruit."""

from src.engine.constants import (
    Direction, GhostMode, GhostID, GHOST_NAMES,
    SCATTER_TARGETS, DIRECTION_DELTAS, OPPOSITE_DIRECTION,
)
from src.engine.maze_data import (
    PACMAN_START, GHOST_START_POSITIONS, GHOST_HOME_POSITIONS, GHOST_DOOR_POS,
)


class PacMan:
    """Pac-Man entity with position, direction, lives, score, and power-up state."""

    def __init__(self, lives: int = 3):
        self.start_row, self.start_col = PACMAN_START
        self.row = self.start_row
        self.col = self.start_col
        self.direction = Direction.LEFT
        self.lives = lives
        self.max_lives = lives
        self.score = 0
        self.powered_up = False
        self.power_timer = 0
        self.ghosts_eaten_this_power = 0  # for escalating score

    def reset_position(self):
        """Reset to start position after death."""
        self.row = self.start_row
        self.col = self.start_col
        self.direction = Direction.LEFT
        self.powered_up = False
        self.power_timer = 0
        self.ghosts_eaten_this_power = 0

    def full_reset(self, lives: int = 3):
        """Full reset for new episode."""
        self.reset_position()
        self.lives = lives
        self.max_lives = lives
        self.score = 0


class Ghost:
    """Ghost entity with identity, position, mode, and navigation state."""

    def __init__(self, ghost_id: GhostID):
        self.ghost_id = ghost_id
        self.name = GHOST_NAMES[ghost_id]
        self.scatter_target = SCATTER_TARGETS[ghost_id]

        start = GHOST_START_POSITIONS[ghost_id]
        self.start_row, self.start_col = start
        self.home_row, self.home_col = GHOST_HOME_POSITIONS[ghost_id]

        self.row = self.start_row
        self.col = self.start_col
        self.direction = Direction.UP if ghost_id == GhostID.BLINKY else Direction.DOWN
        self.mode = GhostMode.SCATTER
        self.previous_mode = GhostMode.SCATTER  # for mode transitions
        self.frightened_timer = 0
        self.in_ghost_house = ghost_id != GhostID.BLINKY  # Blinky starts outside
        self.exiting_house = False

    def reset_position(self):
        """Reset to start position after Pac-Man death."""
        self.row = self.start_row
        self.col = self.start_col
        self.direction = Direction.UP if self.ghost_id == GhostID.BLINKY else Direction.DOWN
        self.mode = GhostMode.SCATTER
        self.previous_mode = GhostMode.SCATTER
        self.frightened_timer = 0
        self.in_ghost_house = self.ghost_id != GhostID.BLINKY
        self.exiting_house = False

    def full_reset(self):
        """Full reset for new episode."""
        self.reset_position()

    def enter_frightened(self, duration: int):
        """Enter frightened mode."""
        if self.mode == GhostMode.EATEN:
            return  # eaten ghosts don't become frightened
        self.previous_mode = self.mode
        self.mode = GhostMode.FRIGHTENED
        self.frightened_timer = duration
        # Ghosts reverse direction when entering frightened
        self.direction = OPPOSITE_DIRECTION[self.direction]

    def enter_eaten(self):
        """Enter eaten mode — return to ghost house."""
        self.mode = GhostMode.EATEN
        self.frightened_timer = 0

    def reach_home(self):
        """Ghost reached home after being eaten — respawn."""
        self.mode = self.previous_mode
        self.in_ghost_house = True
        self.exiting_house = True

    @property
    def is_frightened(self) -> bool:
        return self.mode == GhostMode.FRIGHTENED

    @property
    def is_eaten(self) -> bool:
        return self.mode == GhostMode.EATEN


class Fruit:
    """Bonus fruit that appears temporarily."""

    def __init__(self):
        self.active = False
        self.row = 0
        self.col = 0
        self.timer = 0
        self.score_value = 100

    def spawn(self, row: int, col: int, duration: int, score: int = 100):
        self.active = True
        self.row = row
        self.col = col
        self.timer = duration
        self.score_value = score

    def tick(self):
        """Decrease timer. Deactivate if expired."""
        if self.active:
            self.timer -= 1
            if self.timer <= 0:
                self.active = False

    def collect(self) -> int:
        """Collect fruit. Returns score value."""
        self.active = False
        return self.score_value

    def reset(self):
        self.active = False
        self.timer = 0
