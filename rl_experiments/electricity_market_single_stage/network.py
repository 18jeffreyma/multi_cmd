
import torch.nn as nn
from torch.distributions import Normal

class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Linear(6, 64), 
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1))

    def forward(self, state):
        mu = self.actor(state)
        return Normal(mu, 25)

class critic(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(6, 64), 
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1))

    def forward(self, state):
        mu = self.critic(state)
        return mu