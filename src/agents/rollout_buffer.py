"""Rollout buffer for PPO — stores one episode trajectory and computes GAE returns."""

import numpy as np
import torch


class RolloutBuffer:
    """Stores on-policy trajectory data for PPO training.

    Collects (state, action, log_prob, reward, value, done) per step,
    then computes Generalized Advantage Estimation (GAE) at episode end.
    """

    def __init__(self, gamma: float = 0.95, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
        self.advantages = None
        self.returns = None

    def push(self, state: np.ndarray, action: int, log_prob: float,
             reward: float, value: float, done: bool,
             action_mask: np.ndarray | None = None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        values_arr = np.array(self.values, dtype=np.float32)
        self.advantages = advantages
        self.returns = advantages + values_arr

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield minibatches for PPO epochs."""
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        log_probs = np.array(self.log_probs, dtype=np.float32)

        # Build action mask array: 0 for legal, -inf for illegal
        has_masks = any(m is not None for m in self.action_masks)
        if has_masks:
            mask_list = []
            for m in self.action_masks:
                if m is not None:
                    mask_list.append(m.astype(np.float32))
                else:
                    mask_list.append(np.zeros(4, dtype=np.float32))
            action_masks = np.array(mask_list, dtype=np.float32)
        else:
            action_masks = np.zeros((n, 4), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield (
                torch.from_numpy(states[batch_idx]).to(device),
                torch.from_numpy(actions[batch_idx]).to(device),
                torch.from_numpy(log_probs[batch_idx]).to(device),
                torch.from_numpy(self.returns[batch_idx]).to(device),
                torch.from_numpy(self.advantages[batch_idx]).to(device),
                torch.from_numpy(action_masks[batch_idx]).to(device),
            )

    def __len__(self) -> int:
        return len(self.states)
