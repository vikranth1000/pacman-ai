"""Pygame game renderer — draws the maze, entities, and HUD."""

import pygame
from src.engine.constants import Tile, GhostMode, MAZE_ROWS, MAZE_COLS
from src.engine.game import GameState
from src.gui.sprites import (
    draw_pacman, draw_ghost, draw_pellet, draw_power_pellet, draw_fruit,
)


class GameRenderer:
    """Renders the Pac-Man game state using Pygame."""

    # Colors
    BG_COLOR = (0, 0, 0)
    WALL_COLOR = (33, 33, 222)
    GHOST_HOUSE_COLOR = (40, 40, 40)
    GHOST_DOOR_COLOR = (255, 184, 174)
    TUNNEL_COLOR = (0, 0, 0)
    SIDEBAR_BG = (20, 20, 30)
    TEXT_COLOR = (255, 255, 255)
    LABEL_COLOR = (180, 180, 180)

    def __init__(self, tile_size: int = 20, sidebar_width: int = 200):
        self.tile_size = tile_size
        self.sidebar_width = sidebar_width
        self.maze_width = MAZE_COLS * tile_size
        self.maze_height = MAZE_ROWS * tile_size
        self.width = self.maze_width + sidebar_width
        self.height = self.maze_height
        self.frame = 0

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pac-Man AI Simulator")
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.clock = pygame.time.Clock()

    def render(self, game: GameState, episode: int = 0,
               agents: dict | None = None, win_rates: dict | None = None) -> bool:
        """Render one frame. Returns False if window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.frame += 1
        self.screen.fill(self.BG_COLOR)

        self._draw_maze(game)
        self._draw_entities(game)
        self._draw_sidebar(game, episode, agents, win_rates)

        pygame.display.flip()
        return True

    def _draw_maze(self, game: GameState):
        """Draw maze walls and pellets."""
        ts = self.tile_size
        for r in range(MAZE_ROWS):
            for c in range(MAZE_COLS):
                x = c * ts
                y = r * ts
                tile = game.maze.get_tile(r, c)

                if tile == Tile.WALL:
                    pygame.draw.rect(self.screen, self.WALL_COLOR, (x, y, ts, ts))
                    # Draw darker border for 3D effect
                    pygame.draw.rect(self.screen, (20, 20, 150), (x, y, ts, ts), 1)
                elif tile == Tile.PELLET:
                    draw_pellet(self.screen, x, y, ts)
                elif tile == Tile.POWER_PELLET:
                    draw_power_pellet(self.screen, x, y, ts, self.frame)
                elif tile == Tile.GHOST_HOUSE:
                    pygame.draw.rect(self.screen, self.GHOST_HOUSE_COLOR, (x, y, ts, ts))
                elif tile == Tile.GHOST_DOOR:
                    pygame.draw.rect(self.screen, self.GHOST_DOOR_COLOR,
                                     (x, y + ts // 2 - 1, ts, 3))

    def _draw_entities(self, game: GameState):
        """Draw Pac-Man, ghosts, and fruit."""
        ts = self.tile_size

        # Fruit
        if game.fruit.active:
            draw_fruit(self.screen, game.fruit.col * ts, game.fruit.row * ts, ts)

        # Ghosts
        for ghost in game.ghosts:
            if not ghost.in_ghost_house or ghost.exiting_house:
                draw_ghost(
                    self.screen,
                    ghost.col * ts, ghost.row * ts, ts,
                    int(ghost.ghost_id), ghost.mode,
                    ghost.frightened_timer, self.frame,
                )

        # Pac-Man
        draw_pacman(
            self.screen,
            game.pacman.col * ts, game.pacman.row * ts, ts,
            int(game.pacman.direction), self.frame,
        )

    def _draw_sidebar(self, game: GameState, episode: int,
                      agents: dict | None, win_rates: dict | None):
        """Draw info sidebar with metrics."""
        x = self.maze_width + 5
        w = self.sidebar_width - 10

        # Sidebar background
        pygame.draw.rect(self.screen, self.SIDEBAR_BG,
                         (self.maze_width, 0, self.sidebar_width, self.height))

        y = 10

        # Title
        self._draw_text("PAC-MAN AI", x, y, self.font_large, (255, 255, 0))
        y += 30

        # Episode
        self._draw_text(f"Episode: {episode}", x, y, self.font, self.LABEL_COLOR)
        y += 20

        # Score
        self._draw_text(f"Score: {game.pacman.score}", x, y, self.font, self.TEXT_COLOR)
        y += 20

        # Lives
        self._draw_text(f"Lives: {game.pacman.lives}", x, y, self.font, self.TEXT_COLOR)
        y += 20

        # Steps
        self._draw_text(f"Step: {game.step_count}", x, y, self.font, self.LABEL_COLOR)
        y += 20

        # Pellets
        self._draw_text(f"Pellets: {game.pellets_eaten}", x, y, self.font, self.LABEL_COLOR)
        y += 20

        # Ghosts eaten
        self._draw_text(f"Ghosts eaten: {game.ghosts_eaten_total}", x, y, self.font, self.LABEL_COLOR)
        y += 30

        # Win rates
        if win_rates:
            self._draw_text("WIN RATES (100)", x, y, self.font_large, (100, 200, 255))
            y += 22
            pac_wr = win_rates.get("pacman", 0)
            ghost_wr = win_rates.get("ghosts", 0)
            self._draw_text(f"Pac-Man: {pac_wr:.1%}", x, y, self.font, (255, 255, 0))
            y += 18
            self._draw_text(f"Ghosts:  {ghost_wr:.1%}", x, y, self.font, (255, 50, 50))
            y += 25

            # Win rate bar
            bar_w = w - 10
            bar_h = 12
            pygame.draw.rect(self.screen, (50, 50, 50), (x, y, bar_w, bar_h))
            pac_bar = int(bar_w * pac_wr)
            if pac_bar > 0:
                pygame.draw.rect(self.screen, (255, 255, 0), (x, y, pac_bar, bar_h))
            y += 25

        # Ghost modes
        self._draw_text("GHOST STATUS", x, y, self.font_large, (100, 200, 255))
        y += 22
        mode_names = {GhostMode.SCATTER: "SCT", GhostMode.CHASE: "CHS",
                      GhostMode.FRIGHTENED: "FRT", GhostMode.EATEN: "EAT"}
        for ghost in game.ghosts:
            mode_str = mode_names.get(ghost.mode, "???")
            color = (255, 50, 50) if ghost.mode == GhostMode.FRIGHTENED else self.LABEL_COLOR
            self._draw_text(f"{ghost.name:>6}: {mode_str}", x, y, self.font, color)
            y += 16
        y += 15

        # Agent stats
        if agents:
            self._draw_text("AGENT STATS", x, y, self.font_large, (100, 200, 255))
            y += 22
            pac = agents.get("pacman")
            if pac:
                self._draw_text(f"PM ε: {pac.epsilon:.3f}", x, y, self.font, (255, 255, 0))
                y += 16
                self._draw_text(f"PM reward: {pac.episode_reward:.1f}", x, y, self.font, self.LABEL_COLOR)
                y += 20

            for ghost in game.ghosts:
                agent = agents.get(ghost.name)
                if agent:
                    self._draw_text(
                        f"{ghost.name[:3]} rwd: {agent.episode_reward:.1f}",
                        x, y, self.font, self.LABEL_COLOR
                    )
                    y += 16

        # Power-up indicator
        if game.pacman.powered_up:
            y = self.height - 40
            self._draw_text("POWER UP!", x, y, self.font_large, (0, 255, 255))
            self._draw_text(f"Timer: {game.pacman.power_timer}", x, y + 20, self.font, self.LABEL_COLOR)

    def _draw_text(self, text: str, x: int, y: int,
                   font: pygame.font.Font, color: tuple):
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def tick(self, fps: int = 10):
        """Limit frame rate."""
        if fps > 0:
            self.clock.tick(fps)

    def close(self):
        pygame.quit()
