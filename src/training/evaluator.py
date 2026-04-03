"""Evaluation mode — run trained agents without exploration or learning."""

from pathlib import Path

from src.engine.game import GameState
from src.engine.constants import GhostID, GHOST_NAMES
from src.agents.dqn_agent import DQNAgent
from src.agents.observations import build_pacman_observation, build_ghost_observation, get_observation_sizes
from src.training.checkpoint import load_checkpoint
from src.data.logger import DataLogger
from src.utils.seeding import get_device


class Evaluator:
    """Runs trained agents in evaluation mode (no exploration, no learning)."""

    def __init__(self, config: dict, run_dir: Path):
        self.config = config
        self.device = get_device()
        self.run_dir = Path(run_dir)

        self.game = GameState(config)
        pac_obs_size, ghost_obs_size = get_observation_sizes(config)

        # Create agents and load checkpoints
        self.agents: dict[str, DQNAgent] = {}
        self.agents["pacman"] = DQNAgent("pacman", pac_obs_size, config, self.device)
        for ghost_id in GhostID:
            name = GHOST_NAMES[ghost_id]
            self.agents[name] = DQNAgent(name, ghost_obs_size, config, self.device)

        ep = load_checkpoint(self.run_dir, self.agents)
        print(f"Loaded checkpoint from episode {ep}")

        # Set all epsilons to 0 (no exploration)
        for agent in self.agents.values():
            agent.epsilon = 0.0

    def evaluate(self, num_episodes: int = 100) -> dict:
        """Run evaluation episodes and return summary statistics."""
        results = []

        for ep in range(num_episodes):
            self.game.reset()
            step_result = self._run_episode()
            results.append({
                "episode": ep,
                "winner": step_result["winner"],
                "score": step_result["score"],
                "steps": step_result["step"],
                "pellets_eaten": self.game.pellets_eaten,
                "ghosts_eaten": self.game.ghosts_eaten_total,
            })

        # Compute stats
        total = len(results)
        pac_wins = sum(1 for r in results if r["winner"] == "pacman")
        avg_score = sum(r["score"] for r in results) / total
        avg_steps = sum(r["steps"] for r in results) / total
        avg_pellets = sum(r["pellets_eaten"] for r in results) / total

        summary = {
            "total_episodes": total,
            "pacman_win_rate": pac_wins / total,
            "ghost_win_rate": 1 - pac_wins / total,
            "avg_score": avg_score,
            "avg_steps": avg_steps,
            "avg_pellets_eaten": avg_pellets,
            "results": results,
        }

        print(f"\nEvaluation Results ({total} episodes):")
        print(f"  Pac-Man win rate: {summary['pacman_win_rate']:.1%}")
        print(f"  Ghost win rate:   {summary['ghost_win_rate']:.1%}")
        print(f"  Avg score:        {avg_score:.1f}")
        print(f"  Avg steps:        {avg_steps:.1f}")
        print(f"  Avg pellets:      {avg_pellets:.1f}")

        return summary

    def _run_episode(self) -> dict:
        """Run one episode with no learning."""
        step_result = {"done": False}
        while not self.game.done:
            pac_obs = build_pacman_observation(self.game)
            ghost_obs = [build_ghost_observation(self.game, i) for i in range(4)]

            pac_legal = self.game.get_legal_actions_pacman()
            pac_action = self.agents["pacman"].act(pac_obs, pac_legal, training=False)

            ghost_actions = []
            for i in range(4):
                name = GHOST_NAMES[GhostID(i)]
                legal = self.game.get_legal_actions_ghost(i)
                action = self.agents[name].act(ghost_obs[i], legal, training=False)
                ghost_actions.append(action)

            step_result = self.game.step(pac_action, ghost_actions)

        return step_result
