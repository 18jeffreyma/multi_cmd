
# NOTE: My branch of marlenv has action spaced changed to 3 and has changed player observations to be symmetrical.

import torch
import torch.nn as nn
from torch.distributions import Categorical

class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Conv2d(4, 32, 2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, 2),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(10368, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return Categorical(mu)


class critic(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Conv2d(4, 32, 2),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, 2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(10368, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

    def forward(self, state):
        value = self.critic(state)
        return value


class policy2(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy2, self).__init__()
        self.actor = nn.Sequential(nn.Conv2d(5, 32, 2),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, 2),
                                   nn.ReLU(),
                                   nn.Flatten(),
                                   nn.Linear(10368, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 3),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return Categorical(mu)


class critic2(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self):
        super(critic2, self).__init__()

        self.critic = nn.Sequential(nn.Conv2d(5, 32, 2),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, 2),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(10368, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

    def forward(self, state):
        value = self.critic(state)
        return value

