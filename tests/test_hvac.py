"""
tests/test_hvac.py
Basic tests for the HVAC environment and DQN agent.
These run in GitHub Actions on every push.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env.hvac_env import HVACEnv
from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------

class TestHVACEnv:

    def setup_method(self):
        self.env = HVACEnv()

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset(seed=0)
        assert obs.shape == (4,), "State must be 4-dimensional"

    def test_reset_obs_within_bounds(self):
        obs, _ = self.env.reset(seed=0)
        low  = self.env.observation_space.low
        high = self.env.observation_space.high
        assert np.all(obs >= low) and np.all(obs <= high)

    def test_step_returns_correct_types(self):
        self.env.reset(seed=0)
        obs, reward, done, truncated, info = self.env.step(0)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert obs.shape == (4,)

    def test_episode_ends_at_24_steps(self):
        self.env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, done, _, _ = self.env.step(self.env.action_space.sample())
            steps += 1
        assert steps == 24, f"Episode should be 24 steps, got {steps}"

    def test_comfort_zone_gives_zero_discomfort(self):
        """Indoor temp at exactly 22°C should give zero discomfort penalty."""
        penalty = self.env._discomfort_penalty(22.0)
        assert penalty == 0.0

    def test_discomfort_increases_outside_comfort(self):
        low  = self.env._discomfort_penalty(18.0)
        high = self.env._discomfort_penalty(27.0)
        assert low > 0 and high > 0

    def test_all_actions_valid(self):
        for action in [0, 1, 2]:
            self.env.reset(seed=0)
            obs, reward, done, _, info = self.env.step(action)
            assert obs is not None
            assert reward is not None

    def test_outdoor_profile_length(self):
        self.env.reset(seed=0)
        assert len(self.env.outdoor_temps) == 24


# ---------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------

class TestReplayBuffer:

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(
                np.zeros(4), 0, -1.0, np.zeros(4), False
            )
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert states.shape == (5, 4)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros(4), 0, float(i), np.zeros(4), False)
        assert len(buf) == 10  # oldest should be overwritten

    def test_cannot_sample_more_than_buffer(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(5):
            buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        with pytest.raises(ValueError):
            buf.sample(10)


# ---------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------

class TestDQNAgent:

    def setup_method(self):
        self.agent = DQNAgent()

    def test_action_in_valid_range(self):
        state = np.array([22.0, 34.0, 10.0, 1.0], dtype=np.float32)
        for _ in range(20):
            action = self.agent.select_action(state)
            assert action in [0, 1, 2]

    def test_epsilon_decays(self):
        initial_eps = self.agent.epsilon
        self.agent.decay_epsilon()
        assert self.agent.epsilon < initial_eps

    def test_epsilon_floor(self):
        self.agent.epsilon = 0.05
        for _ in range(100):
            self.agent.decay_epsilon()
        assert self.agent.epsilon >= self.agent.epsilon_end

    def test_train_step_returns_none_when_buffer_empty(self):
        result = self.agent.train_step()
        assert result is None

    def test_train_step_returns_loss_when_buffer_full(self):
        env = HVACEnv()
        obs, _ = env.reset()
        for _ in range(100):
            action = self.agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            self.agent.store(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()[0]
        loss = self.agent.train_step()
        assert loss is not None
        assert loss >= 0.0

    def test_target_network_updates(self):
        import torch
        before = list(self.agent.target_net.parameters())[0].clone()
        # Manually force update
        self.agent.update_target_network()
        after = list(self.agent.target_net.parameters())[0]
        assert torch.equal(before, after)  # hard copy — should be identical