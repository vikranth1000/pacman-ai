"""Main training loop — orchestrates episodes, agents, logging, and checkpointing."""

import time
from pathlib import Path
from datetime import datetime

from src.engine.game import GameState
from src.engine.constants import GhostID, GHOST_NAMES
from src.agents.dqn_agent import DQNAgent
from src.agents.observations import build_pacman_observation, build_ghost_observation, get_observation_sizes
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.data.logger import DataLogger
from src.utils.config import save_config
from src.utils.seeding import set_seeds, get_device


class Trainer:
    """Orchestrates continuous self-play training."""

    def __init__(self, config: dict, run_dir: Path | None = None, resume: bool = False):
        self.config = config
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Set seeds
        seed = config.get("seed", 42)
        set_seeds(seed)

        # Create run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path("runs") / f"run_{timestamp}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        save_config(config, self.run_dir / "config.yaml")

        # Initialize game
        self.game = GameState(config)

        # Get observation sizes dynamically
        pac_obs_size, ghost_obs_size = get_observation_sizes(config)
        print(f"Observation sizes — Pac-Man: {pac_obs_size}, Ghost: {ghost_obs_size}")

        # Create agents
        self.agents: dict[str, DQNAgent] = {}
        self.agents["pacman"] = DQNAgent("pacman", pac_obs_size, config, self.device)
        for ghost_id in GhostID:
            name = GHOST_NAMES[ghost_id]
            self.agents[name] = DQNAgent(name, ghost_obs_size, config, self.device)

        # Logger
        self.logger = DataLogger(self.run_dir / "metrics.db")
        self.logger.log_config(config)

        # Training state
        self.start_episode = 0
        self.reward_cfg = config.get("rewards", {})
        self.train_cfg = config.get("training", {})

        # Resume from checkpoint if requested
        if resume:
            try:
                ep = load_checkpoint(self.run_dir, self.agents)
                self.start_episode = ep
                print(f"Resumed from episode {ep}")
            except FileNotFoundError:
                print("No checkpoint found, starting fresh")

    def train(self, num_episodes: int | None = None, callback=None):
        """Run training loop.

        Args:
            num_episodes: Override config num_episodes if provided.
            callback: Optional function called each episode with (episode, step_result, agents).
                      If callback returns False, training stops.
        """
        if num_episodes is None:
            num_episodes = self.train_cfg.get("num_episodes", 5000)

        checkpoint_every = self.train_cfg.get("checkpoint_every", 100)
        log_every = self.train_cfg.get("log_every", 10)

        total_start = time.time()
        print(f"Starting training: {num_episodes} episodes from episode {self.start_episode}")

        for episode in range(self.start_episode, self.start_episode + num_episodes):
            ep_start = time.time()

            # Reset
            self.game.reset()
            for agent in self.agents.values():
                agent.reset_episode_stats()
                agent.update_epsilon(episode)

            # Run episode
            step_result = self._run_episode(training=True)

            # Log metrics
            self._log_episode(episode, step_result)

            # Checkpoint
            if (episode + 1) % checkpoint_every == 0:
                save_checkpoint(self.run_dir, episode + 1, self.agents, self.config)

            # Print progress
            if (episode + 1) % log_every == 0:
                ep_time = time.time() - ep_start
                elapsed = time.time() - total_start
                eps_per_sec = (episode - self.start_episode + 1) / elapsed
                win_rates = self.logger.get_win_rates(window=100)
                print(
                    f"Episode {episode + 1} | "
                    f"Score: {step_result['score']:>5} | "
                    f"Steps: {step_result['step']:>4} | "
                    f"Winner: {step_result['winner']:>6} | "
                    f"ε: {self.agents['pacman'].epsilon:.3f} | "
                    f"Pac-Man WR: {win_rates['pacman']:.1%} | "
                    f"Ghost WR: {win_rates['ghosts']:.1%} | "
                    f"{eps_per_sec:.1f} ep/s"
                )

            # Callback
            if callback is not None:
                if callback(episode, step_result, self.agents) is False:
                    print("Training stopped by callback")
                    break

        # Final checkpoint
        save_checkpoint(self.run_dir, self.start_episode + num_episodes, self.agents, self.config)
        elapsed = time.time() - total_start
        print(f"Training complete: {num_episodes} episodes in {elapsed:.1f}s "
              f"({num_episodes / elapsed:.1f} ep/s)")

    def _run_episode(self, training: bool = True) -> dict:
        """Run a single episode. Returns the final step result."""
        pac_rewards = self.reward_cfg.get("pacman", {})
        ghost_rewards = self.reward_cfg.get("ghost", {})

        step_result = {"done": False}
        action_counts = {name: 0 for name in self.agents}

        while not self.game.done:
            # Get observations
            pac_obs = build_pacman_observation(self.game)
            ghost_obs = [build_ghost_observation(self.game, i) for i in range(4)]

            # Get actions
            pac_legal = self.game.get_legal_actions_pacman()
            pac_action = self.agents["pacman"].act(pac_obs, pac_legal, training)
            action_counts["pacman"] += 1

            ghost_actions = []
            ghost_names = [GHOST_NAMES[GhostID(i)] for i in range(4)]
            for i in range(4):
                legal = self.game.get_legal_actions_ghost(i)
                action = self.agents[ghost_names[i]].act(ghost_obs[i], legal, training)
                ghost_actions.append(action)
                action_counts[ghost_names[i]] += 1

            # Step game
            step_result = self.game.step(pac_action, ghost_actions)

            # Compute rewards
            pac_reward = self._compute_pacman_reward(step_result, pac_rewards)
            ghost_reward_list = self._compute_ghost_rewards(step_result, ghost_rewards)

            # Get next observations
            next_pac_obs = build_pacman_observation(self.game)
            next_ghost_obs = [build_ghost_observation(self.game, i) for i in range(4)]
            done = step_result["done"]

            # Store transitions and learn
            self.agents["pacman"].store_transition(pac_obs, pac_action, pac_reward, next_pac_obs, done)
            if training:
                self.agents["pacman"].learn()

            for i in range(4):
                name = ghost_names[i]
                self.agents[name].store_transition(
                    ghost_obs[i], ghost_actions[i], ghost_reward_list[i], next_ghost_obs[i], done
                )
                if training:
                    self.agents[name].learn()

        step_result["action_counts"] = action_counts
        return step_result

    def _compute_pacman_reward(self, step_result: dict, reward_cfg: dict) -> float:
        """Compute Pac-Man's reward from step events."""
        reward = reward_cfg.get("time_step", -0.01)

        # Pellet proximity shaping: reward for being close to nearest pellet
        nearest = self.game.maze.find_nearest_pellet(self.game.pacman.row, self.game.pacman.col)
        if nearest is not None:
            dist = abs(nearest[0] - self.game.pacman.row) + abs(nearest[1] - self.game.pacman.col)
            if dist > 0:
                reward += reward_cfg.get("pellet_proximity_scale", 0.05) / dist

        for event in step_result["events"]:
            if event == "eat_pellet":
                reward += reward_cfg.get("eat_pellet", 1.0)
            elif event == "eat_power_pellet":
                reward += reward_cfg.get("eat_power_pellet", 2.0)
            elif event.startswith("eat_ghost_"):
                reward += reward_cfg.get("eat_ghost", 5.0)
            elif event == "eat_fruit":
                reward += reward_cfg.get("eat_fruit", 3.0)
            elif event == "level_clear":
                reward += reward_cfg.get("clear_level", 20.0)
            elif event.startswith("caught_by_"):
                reward += reward_cfg.get("caught_by_ghost", -10.0)
            elif event == "game_over":
                reward += reward_cfg.get("game_over", -20.0)

        return reward

    def _compute_ghost_rewards(self, step_result: dict, reward_cfg: dict) -> list[float]:
        """Compute per-ghost rewards from step events."""
        rewards = [0.0, 0.0, 0.0, 0.0]
        events = step_result["events"]

        for i, ghost in enumerate(self.game.ghosts):
            # Proximity shaping
            pm = self.game.pacman
            dist = abs(ghost.row - pm.row) + abs(ghost.col - pm.col)
            if dist > 0 and not ghost.is_frightened and not ghost.is_eaten:
                rewards[i] += reward_cfg.get("proximity_scale", 0.1) / dist

        for event in events:
            if event == "eat_pellet":
                for i in range(4):
                    rewards[i] += reward_cfg.get("pacman_eats_pellet", -0.5)
            elif event == "level_clear":
                for i in range(4):
                    rewards[i] += reward_cfg.get("pacman_clears_level", -15.0)
            elif event.startswith("caught_by_"):
                # Find which ghost caught Pac-Man
                ghost_name = event.replace("caught_by_", "")
                for i in range(4):
                    name = GHOST_NAMES[GhostID(i)]
                    if name == ghost_name:
                        rewards[i] += reward_cfg.get("catch_pacman_self", 10.0)
                    else:
                        rewards[i] += reward_cfg.get("catch_pacman_team", 3.0)
            elif event == "game_over":
                for i in range(4):
                    rewards[i] += reward_cfg.get("game_over_win", 15.0)
            elif event.startswith("eat_ghost_"):
                ghost_name = event.replace("eat_ghost_", "")
                for i in range(4):
                    name = GHOST_NAMES[GhostID(i)]
                    if name == ghost_name:
                        rewards[i] += reward_cfg.get("got_eaten", -5.0)

        return rewards

    def _log_episode(self, episode: int, step_result: dict):
        """Log episode metrics to database."""
        self.logger.log_episode(
            episode_id=episode,
            winner=step_result.get("winner"),
            score=step_result.get("score", 0),
            steps=step_result.get("step", 0),
            pellets_eaten=self.game.pellets_eaten,
            ghosts_eaten=self.game.ghosts_eaten_total,
            fruits_eaten=self.game.fruits_eaten,
            lives_remaining=self.game.pacman.lives,
            level_cleared=step_result.get("winner") == "pacman",
        )

        action_counts = step_result.get("action_counts", {})
        for name, agent in self.agents.items():
            self.logger.log_agent_metrics(
                episode_id=episode,
                agent_name=name,
                total_reward=agent.episode_reward,
                avg_q_value=agent.avg_q_value,
                epsilon=agent.epsilon,
                loss=agent.last_loss,
                actions_taken=action_counts.get(name, 0),
            )

    def close(self):
        """Cleanup resources."""
        self.logger.close()
