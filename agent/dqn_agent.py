import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn_network import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with:
      - Epsilon-greedy exploration
      - Experience replay
      - Target network (hard update every C steps)
    """

    def __init__(
        self,
        state_dim=4,
        action_dim=3,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=10_000,
        target_update_freq=50,   # update target network every N episodes
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network and target network (same architecture, separate weights)
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.episode_count = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state):
        """
        Epsilon-greedy: explore randomly with probability epsilon,
        otherwise pick the action with the highest Q-value.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        """Push one transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample a random mini-batch and do one gradient update.
        Returns the loss value (float) for logging.
        Returns None if buffer doesn't have enough samples yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for the actions that were actually taken
        # q_net output shape: (batch, 3)  →  gather action column → (batch, 1)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using the Bellman equation
        # target = r + gamma * max_a' Q_target(s', a')   (0 if terminal)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — keeps training stable
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    # End-of-episode updates
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Call once per episode after training."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Hard copy Q-network weights into target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def end_episode(self):
        """
        Convenience method — call at the end of every episode.
        Handles epsilon decay and periodic target network update.
        """
        self.episode_count += 1
        self.decay_epsilon()
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]