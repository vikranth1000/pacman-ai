"""Tests for DQN agents — creation, action selection, learning, save/load."""

import tempfile
import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.replay_buffer import ReplayBuffer
from src.utils.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def agent(config, device):
    return DQNAgent("test_agent", input_size=50, config=config, device=device)


class TestDQNAgent:
    def test_creation(self, agent):
        assert agent.name == "test_agent"
        assert agent.input_size == 50
        assert agent.epsilon == 1.0

    def test_act_returns_legal_action(self, agent):
        state = np.random.randn(50).astype(np.float32)
        legal_actions = [0, 1, 2]
        action = agent.act(state, legal_actions, training=True)
        assert action in legal_actions

    def test_act_respects_action_masking(self, agent):
        agent.epsilon = 0.0  # no exploration
        state = np.random.randn(50).astype(np.float32)
        legal_actions = [2]  # only LEFT is legal
        action = agent.act(state, legal_actions, training=False)
        assert action == 2

    def test_store_transition(self, agent):
        state = np.random.randn(50).astype(np.float32)
        next_state = np.random.randn(50).astype(np.float32)
        agent.store_transition(state, 0, 1.0, next_state, False)
        assert len(agent.replay_buffer) == 1

    def test_learn_returns_none_when_insufficient_data(self, agent):
        result = agent.learn()
        assert result is None

    def test_learn_returns_loss_with_data(self, agent):
        # Fill buffer past minimum (must exceed max(min_replay, batch_size))
        needed = max(agent.min_replay, agent.batch_size) + 100
        for _ in range(needed):
            s = np.random.randn(50).astype(np.float32)
            ns = np.random.randn(50).astype(np.float32)
            agent.replay_buffer.push(s, np.random.randint(4), np.random.randn(), ns, False)
        agent.step_count = agent.learn_every - 1  # next learn will trigger
        loss = agent.learn()
        assert loss is not None
        assert isinstance(loss, float)

    def test_epsilon_decay(self, agent):
        agent.update_epsilon(0)
        assert agent.epsilon == agent.epsilon_start
        agent.update_epsilon(agent.epsilon_decay_episodes)
        assert agent.epsilon == agent.epsilon_end

    def test_save_load(self, agent):
        state = np.random.randn(50).astype(np.float32)
        agent.epsilon = 0.0
        action_before = agent.act(state, [0, 1, 2, 3], training=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        agent.save(path)

        # Create new agent and load
        new_agent = DQNAgent("test_agent", 50, load_config(), torch.device("cpu"))
        new_agent.load(path)
        new_agent.epsilon = 0.0
        action_after = new_agent.act(state, [0, 1, 2, 3], training=False)

        assert action_before == action_after


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        s = np.zeros(10, dtype=np.float32)
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

    def test_capacity(self):
        buf = ReplayBuffer(capacity=10)
        s = np.zeros(5, dtype=np.float32)
        for i in range(20):
            buf.push(s, i % 4, float(i), s, False)
        assert len(buf) == 10

    def test_sample(self):
        buf = ReplayBuffer(capacity=100)
        s = np.zeros(5, dtype=np.float32)
        for _ in range(50):
            buf.push(s, 0, 1.0, s, False)
        states, actions, rewards, next_states, dones = buf.sample(10, torch.device("cpu"))
        assert states.shape == (10, 5)
        assert actions.shape == (10,)
