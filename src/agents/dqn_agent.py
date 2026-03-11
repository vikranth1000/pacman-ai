"""DQN Agent with target network, experience replay, and epsilon-greedy exploration."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.agents.networks import QNetwork
from src.agents.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """Independent DQN agent. Each Pac-Man/Ghost gets its own instance."""

    def __init__(self, name: str, input_size: int, config: dict, device: torch.device):
        self.name = name
        self.input_size = input_size
        self.device = device

        agent_cfg = config.get("agent", {})
        self.gamma = agent_cfg.get("gamma", 0.99)
        self.epsilon = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_start = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = agent_cfg.get("epsilon_end", 0.05)
        self.epsilon_decay_episodes = agent_cfg.get("epsilon_decay_episodes", 500)
        self.batch_size = agent_cfg.get("batch_size", 64)
        self.tau = agent_cfg.get("target_update_tau", 0.005)
        self.min_replay = agent_cfg.get("min_replay_before_learn", 1000)
        self.learn_every = agent_cfg.get("learn_every_n_steps", 4)
        self.grad_clip = agent_cfg.get("gradient_clip_norm", 1.0)
        hidden_sizes = agent_cfg.get("hidden_sizes", [256, 128, 64])

        # Networks
        self.q_network = QNetwork(input_size, 4, hidden_sizes).to(device)
        self.target_network = QNetwork(input_size, 4, hidden_sizes).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss
        lr = agent_cfg.get("learning_rate", 3e-4)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        buffer_size = agent_cfg.get("replay_buffer_size", 100000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Tracking
        self.step_count = 0
        self.last_loss = None
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.avg_q_value = 0.0

    def act(self, state: np.ndarray, legal_actions: list[int], training: bool = True) -> int:
        """Epsilon-greedy action selection with action masking."""
        if not legal_actions:
            return random.randint(0, 3)

        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)

            # Action masking: set illegal actions to -inf
            mask = torch.full((4,), float('-inf'), device=self.device)
            for a in legal_actions:
                mask[a] = 0.0
            masked_q = q_values + mask

            self.avg_q_value = q_values[legal_actions].mean().item()
            return masked_q.argmax().item()

    def learn(self) -> float | None:
        """Sample batch from replay buffer and perform one gradient step."""
        if len(self.replay_buffer) < max(self.min_replay, self.batch_size):
            return None

        self.step_count += 1
        if self.step_count % self.learn_every != 0:
            return self.last_loss

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q-values for chosen actions
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online network selects best action, target network evaluates it
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (~dones).float()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target()

        self.last_loss = loss.item()
        return self.last_loss

    def _soft_update_target(self):
        """Polyak averaging: target = τ * online + (1-τ) * target."""
        for tp, op in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.episode_reward += reward

    def update_epsilon(self, episode: int):
        """Linear epsilon decay."""
        if episode >= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_end
        else:
            fraction = episode / self.epsilon_decay_episodes
            self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def reset_episode_stats(self):
        """Reset per-episode tracking."""
        self.episode_reward = 0.0
        self.avg_q_value = 0.0

    def save(self, path: str):
        """Save complete agent state."""
        torch.save({
            "name": self.name,
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "input_size": self.input_size,
        }, path)

    def load(self, path: str):
        """Load agent state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint.get("step_count", 0)
