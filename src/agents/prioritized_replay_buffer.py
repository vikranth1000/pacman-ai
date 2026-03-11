"""Prioritized Experience Replay with SumTree for O(log n) proportional sampling."""

import numpy as np
import torch


class SumTree:
    """Binary tree where each leaf stores a priority and internal nodes store sums.

    Enables O(log n) proportional sampling and O(log n) priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_idx = 0

    def update(self, idx: int, priority: float):
        """Update priority of leaf node at data index."""
        tree_idx = idx + self.capacity
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate up
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] += delta
            tree_idx //= 2

    def total(self) -> float:
        return self.tree[1]

    def sample(self, value: float) -> int:
        """Sample a leaf index proportional to priorities."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - self.capacity

    def min_priority(self, size: int) -> float:
        """Get minimum priority among the first `size` leaves."""
        leaves = self.tree[self.capacity:self.capacity + size]
        nonzero = leaves[leaves > 0]
        if len(nonzero) == 0:
            return 1.0
        return float(nonzero.min())


class PrioritizedReplayBuffer:
    """Prioritized replay buffer using SumTree for proportional sampling.

    Same external interface as ReplayBuffer but with priority-weighted sampling
    and importance sampling weights for unbiased gradient updates.
    """

    def __init__(self, capacity: int = 200000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal_episodes: int = 2500):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_episodes = beta_anneal_episodes

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.size = 0
        self.pos = 0
        self._initialized = False

    def _init_arrays(self, state_size: int):
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

        # New transitions get max priority (ensures they are sampled at least once)
        self.tree.update(self.pos, self.max_priority ** self.alpha)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple:
        """Sample batch proportional to priorities.

        Returns: (states, actions, rewards, next_states, dones, indices, weights)
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx = self.tree.sample(value)
            # Clamp to valid range
            idx = min(idx, self.size - 1)
            indices[i] = idx
            priorities[i] = self.tree.tree[idx + self.capacity]

        # Importance sampling weights
        min_prob = self.tree.min_priority(self.size) / total
        min_prob = max(min_prob, 1e-8)
        probs = priorities / total
        probs = np.maximum(probs, 1e-8)
        weights = (self.size * probs) ** (-self.beta)
        max_weight = (self.size * min_prob) ** (-self.beta)
        weights /= max_weight  # normalize

        states = torch.from_numpy(self.states[indices]).to(device)
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        next_states = torch.from_numpy(self.next_states[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        weights = torch.from_numpy(weights.astype(np.float32)).to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def update_beta(self, episode: int):
        """Anneal beta toward 1.0 over training."""
        fraction = min(episode / max(self.beta_anneal_episodes, 1), 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.size
