import torch
import torch.nn as nn
import numpy as np
from src.actions import NUM_ACTIONS
from src.states import STATE_SIZE

class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS)     # Linear output — raw Q-values, no activation
        )

        self._xavier_init()

    def _xavier_init(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(
                    layer.weight,
                    -1 / np.sqrt(layer.in_features),
                    +1 / np.sqrt(layer.in_features)
                )
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)