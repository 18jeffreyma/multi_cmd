
# NOTE: My branch of marlenv has action spaced changed to 3 and has changed player observations to be symmetrical.

import time
import torch
import torch.nn as nn
from torch.distributions import Categorical

class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Conv2d(8, 32, 2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, 2),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(2592, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 5),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        # Change from NHWC to NCHW for Conv2D.
        state_transposed = state.permute(0, 3, 1, 2)
        probs = self.actor(state_transposed)
        return Categorical(probs)


class critic(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Conv2d(8, 32, 2),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, 2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(2592, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

    def forward(self, state):
        state_transposed = state.permute(0, 3, 1, 2)
        value = self.critic(state_transposed)
        return value


