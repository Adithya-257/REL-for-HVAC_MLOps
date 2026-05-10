import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Simple 3-layer fully connected network.

    Input:  state vector of size 4
            [indoor_temp, outdoor_temp, hour_of_day, occupancy]
    Output: Q-value for each of the 3 actions
            [Q(s, off), Q(s, cool), Q(s, heat)]

    Architecture: 4 -> 64 -> 64 -> 3
    ReLU activations, no activation on output layer.
    """

    def __init__(self, state_dim=4, action_dim=3, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)