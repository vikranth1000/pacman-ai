"""Neural network architectures for DQN and PPO agents."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP Q-Network: input features → Q-values for each action."""

    def __init__(self, input_size: int, output_size: int = 4,
                 hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorCriticNetwork(nn.Module):
    """Shared-backbone actor-critic network for PPO.

    Actor head outputs action logits (4 actions).
    Critic head outputs state value (scalar).
    """

    def __init__(self, input_size: int, num_actions: int = 4,
                 hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        # Shared feature extractor
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        self.backbone = nn.Sequential(*layers)

        # Actor head (action logits)
        self.actor = nn.Linear(prev_size, num_actions)
        # Critic head (state value)
        self.critic = nn.Linear(prev_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        features = self.backbone(x)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action_mask: torch.Tensor | None = None):
        """Get action logits and value, with optional action masking."""
        logits, value = self.forward(x)
        if action_mask is not None:
            logits = logits + action_mask  # mask has -inf for illegal actions
        return logits, value
