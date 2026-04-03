"""Experience replay buffer for DQN agents."""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer using pre-allocated numpy arrays for fast sampling."""

    def __init__(self, capacity: int = 100000, state_size: int = 0):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self._initialized = False
        self._state_size = state_size

    def _init_arrays(self, state_size: int):
        """Lazily initialize arrays on first push."""
        self._state_size = state_size
        self.states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=bool)
        self._initialized = True

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        if not self._initialized:
            self._init_arrays(len(state))

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple:
        """Sample a random batch and return as tensors on device."""
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        next_states = torch.from_numpy(self.next_states[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        """Serialize buffer contents for checkpointing."""
        if self.size == 0:
            return {"size": 0, "pos": 0}
        return {
            "states": self.states[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy(),
            "next_states": self.next_states[:self.size].copy(),
            "dones": self.dones[:self.size].copy(),
            "size": self.size,
            "pos": self.pos,
        }

    def load_state_dict(self, state_dict: dict):
        """Load buffer from checkpoint."""
        size = state_dict.get("size", 0)
        if size == 0:
            return
        if not self._initialized:
            self._init_arrays(state_dict["states"].shape[1])
        n = min(size, self.capacity)
        self.states[:n] = state_dict["states"][:n]
        self.actions[:n] = state_dict["actions"][:n]
        self.rewards[:n] = state_dict["rewards"][:n]
        self.next_states[:n] = state_dict["next_states"][:n]
        self.dones[:n] = state_dict["dones"][:n]
        self.size = n
        self.pos = state_dict.get("pos", n) % self.capacity
