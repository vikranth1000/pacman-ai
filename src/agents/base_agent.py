"""Abstract base class for all agents."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Interface that all agents must implement."""

    @abstractmethod
    def act(self, state: np.ndarray, legal_actions: list[int], training: bool = True) -> int:
        """Select an action given current state and legal actions."""
        ...

    @abstractmethod
    def learn(self) -> float | None:
        """Perform one learning step. Returns loss or None if not enough data."""
        ...

    @abstractmethod
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        ...

    def end_episode(self) -> float | None:
        """Called at end of episode. PPO uses this for batch updates. DQN no-ops."""
        return None

    @abstractmethod
    def save(self, path: str):
        """Save agent state to disk."""
        ...

    @abstractmethod
    def load(self, path: str):
        """Load agent state from disk."""
        ...

    @abstractmethod
    def update_epsilon(self, episode: int):
        """Update exploration rate based on episode number."""
        ...
