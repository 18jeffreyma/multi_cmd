
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self, state_dim, action_dim):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, 32),
                                   nn.Tanh(),
                                   nn.Linear(32, action_dim),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return mu


class critic(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self, state_dim):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 32),
                                    nn.Tanh(),
                                    nn.Linear(32, 1))

    def forward(self, state):
        value = self.critic(state)
        return value
