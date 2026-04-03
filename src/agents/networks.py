"""Neural network architectures for DQN agents."""

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
