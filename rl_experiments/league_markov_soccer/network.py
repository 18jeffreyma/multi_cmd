
# NOTE: My branch of marlenv has action spaced changed to 3 and has changed player observations to be symmetrical.

import torch
import torch.nn as nn
from torch.distributions import Categorical

class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Linear(80, 64),  # 84*50
                                   nn.ReLU(),
                                   nn.Linear(64, 32),  # 50*20
                                   nn.ReLU(),
                                   nn.Linear(32, 5),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return Categorical(mu)


class critic(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(80, 64),  # 84*50
                                    nn.ReLU(),
                                    nn.Linear(64, 32),  # 50*20
                                    nn.ReLU(),
                                    nn.Linear(32, 1))

    def forward(self, state):
        value = self.critic(state)
        return value
