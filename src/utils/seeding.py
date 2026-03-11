"""Reproducible random seed management."""

import random
import numpy as np
import torch


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """Get device for training.

    Uses CPU by default — for the small MLP networks in this project (~55K params),
    CPU is ~5x faster than MPS due to kernel launch overhead on Apple Silicon.
    Set PACMAN_DEVICE=mps to force MPS if experimenting with larger networks.
    """
    import os
    override = os.environ.get("PACMAN_DEVICE")
    if override:
        return torch.device(override)
    return torch.device("cpu")
