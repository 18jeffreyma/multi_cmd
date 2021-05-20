
# NOTE: My branch of marlenv has action spaced changed to 3 and has changed player observations to be symmetrical.

import torch
import torch.nn as nn
from torch.distributions import LogNormal

import numpy as np

class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Linear(6, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU())
        self.actor_loc = nn.Sequential(nn.Linear(128, 1))
        self.actor_scale = nn.Sequential(nn.Linear(128, 1))

    def forward(self, state):
        hidden_layers = self.actor(state)
        loc = self.actor_loc(hidden_layers)

        log_std = self.actor_scale(hidden_layers)
        log_std_clamped = torch.clamp(log_std, min=-20, max=3)
        scale = log_std_clamped.exp().expand_as(loc)
        return LogNormal(loc, scale)

class critic(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(critic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(6, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1))

    def forward(self, state):
        return self.critic(state)

if __name__ == '__main__':
    p = policy()
    dist = p(torch.tensor([0., 0., 0., 0., 0., 0.]))
    print(dist)


    q = critic()
    print(q(torch.tensor([0., 0., 0., 0., 0., 0.])))


