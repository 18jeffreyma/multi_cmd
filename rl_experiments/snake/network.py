
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(torch.nn.Conv2d(4, 64, 3),
                                   nn.Tanh(),
                                   torch.nn.Conv2d(64, 128, 3),
                                   nn.Tanh(),
                                   nn.Flatten(),
                                   nn.Linear(32768, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, 4),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return mu


class critic(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self, state_dim):
        super(critic, self).__init__()

        self.critic = nn.Sequential(torch.nn.Conv2d(4, 64, 3),
                                   nn.Tanh(),
                                   torch.nn.Conv2d(64, 128, 3),
                                   nn.Tanh(),
                                   nn.Flatten(),
                                   nn.Linear(32768, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, 1))

    def forward(self, state):
        value = self.critic(state)
        return value
