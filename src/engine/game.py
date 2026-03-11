"""Game engine — main game state, step(), reset(), collision detection, ghost mode management."""

from collections import deque
from src.engine.constants import (
    Direction, Tile, GhostMode, GhostID, DIRECTION_DELTAS,
    OPPOSITE_DIRECTION, MAZE_ROWS, MAZE_COLS,
    PELLET_SCORE, POWER_PELLET_SCORE, GHOST_EAT_SCORES, NUM_ACTIONS,
)
from src.engine.maze import Maze
from src.engine.entities import PacMan, Ghost, Fruit
from src.engine.maze_data import GHOST_DOOR_POS, FRUIT_POSITION


class GameState:
    """Complete game state with step-based simulation."""

    def __init__(self, config: dict):
        self.config = config
        game_cfg = config.get("game", {})

        self.maze = Maze()
        self.pacman = PacMan(lives=game_cfg.get("lives", 3))
        self.ghosts = [Ghost(GhostID(i)) for i in range(4)]
        self.fruit = Fruit()

        # Game state
        self.step_count = 0
        self.max_steps = game_cfg.get("max_steps_per_episode", 3000)
        self.done = False
        self.winner = None  # "pacman" or "ghosts"
        self.pellets_eaten = 0
        self.ghosts_eaten_total = 0
        self.fruits_eaten = 0

        # Mode schedule
        self.mode_schedule = game_cfg.get("mode_schedule", [42, 120, 42, 120, 30, 120, 30, -1])
        self.mode_index = 0
        self.mode_timer = self.mode_schedule[0] if self.mode_schedule else 120
        self.global_mode = GhostMode.SCATTER

        # Frightened duration
        self.frightened_duration = game_cfg.get("frightened_duration", 36)

        # Fruit config
        self.fruit_spawn_pellets = game_cfg.get("fruit_spawn_pellets", [70, 170])
        self.fruit_score = game_cfg.get("fruit_score", 100)
        self.fruit_duration = game_cfg.get("fruit_duration", 60)
        self.fruit_spawned_at = set()  # track which thresholds already spawned

        # Ghost house exit thresholds
        self.ghost_exit_pellets = game_cfg.get("ghost_exit_pellets", [0, 30, 60, 80])

        # Events from the last step (for reward computation)
        self.events = []

    def reset(self):
        """Reset for a new episode."""
        self.maze.reset()
        self.pacman.full_reset(lives=self.config.get("game", {}).get("lives", 3))
        for g in self.ghosts:
            g.full_reset()
        self.fruit.reset()

        self.step_count = 0
        self.done = False
        self.winner = None
        self.pellets_eaten = 0
        self.ghosts_eaten_total = 0
        self.fruits_eaten = 0
        self.events = []

        self.mode_index = 0
        self.mode_timer = self.mode_schedule[0] if self.mode_schedule else 120
        self.global_mode = GhostMode.SCATTER
        self.fruit_spawned_at = set()

    def step(self, pacman_action: int, ghost_actions: list[int]) -> dict:
        """Advance one tick. Returns dict of events that occurred.

        Args:
            pacman_action: Direction index (0-3) for Pac-Man
            ghost_actions: List of 4 Direction indices for each ghost
        """
        if self.done:
            return {"done": True, "events": []}

        self.step_count += 1
        self.events = []

        # 1. Move Pac-Man
        self._move_pacman(Direction(pacman_action))

        # 2. Check Pac-Man collisions (pellets, fruit) after Pac-Man moves
        self._check_pacman_pickups()

        # 3. Check Pac-Man-ghost collision (mid-step)
        if self._check_ghost_collision():
            return self._build_step_result()

        # 4. Move ghosts
        for i, ghost in enumerate(self.ghosts):
            self._move_ghost(ghost, Direction(ghost_actions[i]))

        # 5. Check Pac-Man-ghost collision again (after ghost moves)
        if self._check_ghost_collision():
            return self._build_step_result()

        # 6. Update ghost mode timers
        self._update_mode_timers()

        # 7. Update ghost house exits
        self._update_ghost_house()

        # 8. Update fruit
        self.fruit.tick()
        self._check_fruit_spawn()

        # 9. Update Pac-Man power timer
        if self.pacman.powered_up:
            self.pacman.power_timer -= 1
            if self.pacman.power_timer <= 0:
                self.pacman.powered_up = False
                self.pacman.ghosts_eaten_this_power = 0
                self.events.append("power_end")

        # 10. Check win condition
        if self.maze.pellets_remaining <= 0:
            self.done = True
            self.winner = "pacman"
            self.events.append("level_clear")

        # 11. Check max steps
        if self.step_count >= self.max_steps:
            self.done = True
            if self.winner is None:
                self.winner = "ghosts"  # timeout = ghosts win
            self.events.append("timeout")

        return self._build_step_result()

    def _move_pacman(self, desired_direction: Direction):
        """Move Pac-Man in desired direction if legal, else continue current direction."""
        dr, dc = DIRECTION_DELTAS[desired_direction]
        nr, nc = self.maze.wrap_position(self.pacman.row + dr, self.pacman.col + dc)

        if self.maze.is_walkable_for_pacman(nr, nc):
            self.pacman.row = nr
            self.pacman.col = nc
            self.pacman.direction = desired_direction
        else:
            # Try continuing current direction
            dr, dc = DIRECTION_DELTAS[self.pacman.direction]
            nr, nc = self.maze.wrap_position(self.pacman.row + dr, self.pacman.col + dc)
            if self.maze.is_walkable_for_pacman(nr, nc):
                self.pacman.row = nr
                self.pacman.col = nc

    def _move_ghost(self, ghost: Ghost, desired_direction: Direction):
        """Move ghost. Handles ghost house exit and eaten return."""
        if ghost.in_ghost_house and not ghost.exiting_house:
            return  # waiting in ghost house

        if ghost.exiting_house:
            # Move toward ghost door exit
            target_row, target_col = GHOST_DOOR_POS
            # Simple: move toward door
            if ghost.row > target_row:
                ghost.row -= 1
            elif ghost.row < target_row:
                ghost.row += 1
            elif ghost.col > target_col:
                ghost.col -= 1
            elif ghost.col < target_col:
                ghost.col += 1
            else:
                # Reached door, move up one more to exit
                ghost.row -= 1
                ghost.in_ghost_house = False
                ghost.exiting_house = False
            return

        if ghost.is_eaten:
            # Move toward ghost house
            self._move_ghost_toward_target(ghost, GHOST_DOOR_POS[0], GHOST_DOOR_POS[1])
            if ghost.row == GHOST_DOOR_POS[0] and ghost.col == GHOST_DOOR_POS[1]:
                ghost.reach_home()
            return

        # Normal movement — cannot reverse direction
        dr, dc = DIRECTION_DELTAS[desired_direction]
        nr, nc = self.maze.wrap_position(ghost.row + dr, ghost.col + dc)

        # Check if desired direction is opposite (forbidden unless mode change)
        is_reverse = desired_direction == OPPOSITE_DIRECTION.get(ghost.direction)

        if not is_reverse and self.maze.is_walkable(nr, nc):
            ghost.row = nr
            ghost.col = nc
            ghost.direction = desired_direction
        else:
            # Try other legal directions (prefer current direction, then others)
            moved = False
            # First try current direction
            dr, dc = DIRECTION_DELTAS[ghost.direction]
            nr, nc = self.maze.wrap_position(ghost.row + dr, ghost.col + dc)
            if self.maze.is_walkable(nr, nc):
                ghost.row = nr
                ghost.col = nc
                moved = True

            if not moved:
                # Try all non-reverse directions
                for d in Direction:
                    if d == OPPOSITE_DIRECTION.get(ghost.direction):
                        continue
                    dr, dc = DIRECTION_DELTAS[d]
                    nr, nc = self.maze.wrap_position(ghost.row + dr, ghost.col + dc)
                    if self.maze.is_walkable(nr, nc):
                        ghost.row = nr
                        ghost.col = nc
                        ghost.direction = d
                        moved = True
                        break

                if not moved:
                    # Last resort: reverse
                    rev = OPPOSITE_DIRECTION[ghost.direction]
                    dr, dc = DIRECTION_DELTAS[rev]
                    nr, nc = self.maze.wrap_position(ghost.row + dr, ghost.col + dc)
                    if self.maze.is_walkable(nr, nc):
                        ghost.row = nr
                        ghost.col = nc
                        ghost.direction = rev

    def _move_ghost_toward_target(self, ghost: Ghost, target_row: int, target_col: int):
        """Simple pathfinding: move ghost one step toward target using BFS."""
        # BFS to find first step toward target
        visited = set()
        queue = deque()
        queue.append((ghost.row, ghost.col, None))  # (row, col, first_direction)
        visited.add((ghost.row, ghost.col))

        while queue:
            r, c, first_dir = queue.popleft()
            if r == target_row and c == target_col:
                if first_dir is not None:
                    dr, dc = DIRECTION_DELTAS[first_dir]
                    nr, nc = self.maze.wrap_position(ghost.row + dr, ghost.col + dc)
                    ghost.row = nr
                    ghost.col = nc
                    ghost.direction = first_dir
                return

            for d in Direction:
                dr, dc = DIRECTION_DELTAS[d]
                nr, nc = self.maze.wrap_position(r + dr, c + dc)
                if (nr, nc) not in visited and self.maze.is_walkable(nr, nc):
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_dir if first_dir is not None else d))

    def _check_pacman_pickups(self):
        """Check if Pac-Man picked up pellets or fruit."""
        eaten = self.maze.eat_pellet(self.pacman.row, self.pacman.col)

        if eaten == Tile.PELLET:
            self.pacman.score += PELLET_SCORE
            self.pellets_eaten += 1
            self.events.append("eat_pellet")

        elif eaten == Tile.POWER_PELLET:
            self.pacman.score += POWER_PELLET_SCORE
            self.pellets_eaten += 1
            self.pacman.powered_up = True
            self.pacman.power_timer = self.frightened_duration
            self.pacman.ghosts_eaten_this_power = 0
            self.events.append("eat_power_pellet")
            # All non-eaten ghosts enter frightened mode
            for g in self.ghosts:
                g.enter_frightened(self.frightened_duration)

        # Check fruit
        if self.fruit.active and self.pacman.row == self.fruit.row and self.pacman.col == self.fruit.col:
            score = self.fruit.collect()
            self.pacman.score += score
            self.fruits_eaten += 1
            self.events.append("eat_fruit")

    def _check_ghost_collision(self) -> bool:
        """Check collision between Pac-Man and ghosts. Returns True if Pac-Man died (episode might end)."""
        for ghost in self.ghosts:
            if ghost.in_ghost_house or ghost.is_eaten:
                continue
            if ghost.row == self.pacman.row and ghost.col == self.pacman.col:
                if ghost.is_frightened:
                    # Pac-Man eats ghost
                    ghost.enter_eaten()
                    eat_index = min(self.pacman.ghosts_eaten_this_power, len(GHOST_EAT_SCORES) - 1)
                    score = GHOST_EAT_SCORES[eat_index]
                    self.pacman.score += score
                    self.pacman.ghosts_eaten_this_power += 1
                    self.ghosts_eaten_total += 1
                    self.events.append(f"eat_ghost_{ghost.name}")
                else:
                    # Ghost catches Pac-Man
                    self.pacman.lives -= 1
                    self.events.append(f"caught_by_{ghost.name}")

                    if self.pacman.lives <= 0:
                        self.done = True
                        self.winner = "ghosts"
                        self.events.append("game_over")
                        return True
                    else:
                        # Reset positions, not the whole game
                        self.pacman.reset_position()
                        for g in self.ghosts:
                            g.reset_position()
                        self.events.append("life_lost")
                        return True  # positions reset, skip remaining movement
        return False

    def _update_mode_timers(self):
        """Update scatter/chase mode schedule and frightened timers."""
        # Update individual ghost frightened timers
        for ghost in self.ghosts:
            if ghost.is_frightened:
                ghost.frightened_timer -= 1
                if ghost.frightened_timer <= 0:
                    ghost.mode = self.global_mode
                    ghost.frightened_timer = 0

        # Update global scatter/chase timer
        if self.mode_timer > 0:
            self.mode_timer -= 1
            if self.mode_timer <= 0:
                self.mode_index += 1
                if self.mode_index < len(self.mode_schedule):
                    duration = self.mode_schedule[self.mode_index]
                    if duration == -1:
                        # Permanent chase
                        self.global_mode = GhostMode.CHASE
                        self.mode_timer = 999999
                    else:
                        # Alternate between scatter and chase
                        self.global_mode = GhostMode.CHASE if self.mode_index % 2 == 1 else GhostMode.SCATTER
                        self.mode_timer = duration

                    # Apply to non-frightened, non-eaten ghosts
                    for ghost in self.ghosts:
                        if ghost.mode not in (GhostMode.FRIGHTENED, GhostMode.EATEN):
                            old_mode = ghost.mode
                            ghost.mode = self.global_mode
                            if old_mode != self.global_mode:
                                ghost.direction = OPPOSITE_DIRECTION[ghost.direction]

    def _update_ghost_house(self):
        """Check if any ghost should exit the ghost house."""
        for i, ghost in enumerate(self.ghosts):
            if ghost.in_ghost_house and not ghost.exiting_house:
                threshold = self.ghost_exit_pellets[i] if i < len(self.ghost_exit_pellets) else 0
                if self.pellets_eaten >= threshold:
                    ghost.exiting_house = True

    def _check_fruit_spawn(self):
        """Check if fruit should spawn based on pellets eaten."""
        for threshold in self.fruit_spawn_pellets:
            if self.pellets_eaten >= threshold and threshold not in self.fruit_spawned_at:
                self.fruit_spawned_at.add(threshold)
                if not self.fruit.active:
                    self.fruit.spawn(
                        FRUIT_POSITION[0], FRUIT_POSITION[1],
                        self.fruit_duration, self.fruit_score,
                    )
                    self.events.append("fruit_spawn")

    def _build_step_result(self) -> dict:
        """Build step result dictionary."""
        return {
            "done": self.done,
            "winner": self.winner,
            "score": self.pacman.score,
            "step": self.step_count,
            "events": list(self.events),
            "pellets_remaining": self.maze.pellets_remaining,
        }

    def get_legal_actions_pacman(self) -> list[int]:
        """Get legal action indices for Pac-Man."""
        directions = self.maze.get_legal_directions(self.pacman.row, self.pacman.col, for_pacman=True)
        return [int(d) for d in directions]

    def get_legal_actions_ghost(self, ghost_idx: int) -> list[int]:
        """Get legal action indices for a ghost."""
        ghost = self.ghosts[ghost_idx]
        if ghost.in_ghost_house or ghost.exiting_house or ghost.is_eaten:
            return list(range(NUM_ACTIONS))  # any action, movement is overridden
        directions = self.maze.get_legal_directions(ghost.row, ghost.col, for_pacman=False)
        # Filter out reverse direction
        current_opposite = OPPOSITE_DIRECTION.get(ghost.direction)
        filtered = [int(d) for d in directions if d != current_opposite]
        return filtered if filtered else [int(d) for d in directions]
