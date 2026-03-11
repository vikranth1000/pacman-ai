"""PPO Agent — Proximal Policy Optimization for stable multi-agent training."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.agents.networks import ActorCriticNetwork
from src.agents.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """PPO agent with actor-critic, GAE, and clipped surrogate objective."""

    def __init__(self, name: str, input_size: int, config: dict, device: torch.device):
        self.name = name
        self.input_size = input_size
        self.device = device

        agent_cfg = config.get("agent", {})
        self.gamma = agent_cfg.get("gamma", 0.95)
        self.epsilon = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_start = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = agent_cfg.get("epsilon_end", 0.08)
        self.epsilon_decay_episodes = agent_cfg.get("epsilon_decay_episodes", 2000)

        # PPO hyperparameters
        self.clip_epsilon = agent_cfg.get("ppo_clip_epsilon", 0.2)
        self.ppo_epochs = agent_cfg.get("ppo_epochs", 4)
        self.minibatch_size = agent_cfg.get("ppo_minibatch_size", 64)
        self.value_loss_coef = agent_cfg.get("ppo_value_loss_coef", 0.5)
        self.entropy_coef = agent_cfg.get("ppo_entropy_coef", 0.01)
        self.max_grad_norm = agent_cfg.get("ppo_max_grad_norm", 0.5)
        gae_lambda = agent_cfg.get("ppo_gae_lambda", 0.95)

        hidden_sizes = agent_cfg.get("hidden_sizes", [256, 256, 128])

        # Network and optimizer
        self.network = ActorCriticNetwork(input_size, 4, hidden_sizes).to(device)
        lr = agent_cfg.get("learning_rate", 0.0003)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Rollout buffer
        self.rollout = RolloutBuffer(gamma=self.gamma, gae_lambda=gae_lambda)

        # Tracking
        self.step_count = 0
        self.last_loss = None
        self.episode_reward = 0.0
        self.avg_q_value = 0.0  # stores avg value estimate for compatibility

    def act(self, state: np.ndarray, legal_actions: list[int], training: bool = True) -> int:
        """Select action using policy with action masking."""
        if not legal_actions:
            return 0

        # Epsilon-greedy exploration on top of policy
        if training and np.random.random() < self.epsilon:
            action = np.random.choice(legal_actions)
            # Still need log_prob and value for the rollout buffer
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                logits, value = self.network(state_t)
                # Apply action mask
                mask = torch.full((4,), float('-inf'), device=self.device)
                for a in legal_actions:
                    mask[a] = 0.0
                logits = logits.squeeze(0) + mask
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
                self.avg_q_value = value.item()
                self._last_log_prob = log_prob
                self._last_value = value.item()
            return action

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.network(state_t)

            # Action masking
            mask = torch.full((4,), float('-inf'), device=self.device)
            for a in legal_actions:
                mask[a] = 0.0
            logits = logits.squeeze(0) + mask

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.avg_q_value = value.item()
            self._last_log_prob = log_prob.item()
            self._last_value = value.item()

            return action.item()

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store transition in rollout buffer."""
        self.rollout.push(
            state, action,
            self._last_log_prob,
            reward,
            self._last_value,
            done,
        )
        self.episode_reward += reward

    def learn(self) -> float | None:
        """No-op during episode. PPO learns at episode end via end_episode()."""
        return self.last_loss

    def end_episode(self) -> float | None:
        """Perform PPO update at end of episode."""
        if len(self.rollout) == 0:
            return None

        # Compute GAE
        self.rollout.compute_returns_and_advantages(last_value=0.0)

        total_loss = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for states, actions, old_log_probs, returns, advantages in \
                    self.rollout.get_batches(self.minibatch_size, self.device):

                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass
                logits, values = self.network(states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        self.rollout.clear()
        self.last_loss = total_loss / max(num_updates, 1)
        return self.last_loss

    def update_epsilon(self, episode: int):
        """Linear epsilon decay."""
        if episode >= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_end
        else:
            fraction = episode / self.epsilon_decay_episodes
            self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def set_epsilon(self, value: float):
        self.epsilon = value

    def set_learning_rate(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def reset_episode_stats(self):
        self.episode_reward = 0.0
        self.avg_q_value = 0.0

    def save(self, path: str):
        torch.save({
            "name": self.name,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "input_size": self.input_size,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint.get("step_count", 0)
