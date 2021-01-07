
import torch
import torch.nn as nn
#
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0., std=0.1)
#         nn.init.constant_(m.bias, 0.1)


class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(nn.Conv2d(4, 16, 3),
                                   nn.Tanh(),
                                   nn.Conv2d(16, 32, 3),
                                   nn.Tanh(),
                                   nn.Flatten(),
                                   nn.Linear(8192, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 5),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return mu


class critic(nn.Module):
    """Critic model for estimating value from state."""
    def __init__(self):
        super(critic, self).__init__()

        self.critic = nn.Sequential(nn.Conv2d(4, 16, 3),
                                   nn.Tanh(),
                                   nn.Conv2d(16, 32, 3),
                                   nn.Tanh(),
                                   nn.Flatten(),
                                   nn.Linear(8192, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 1))

    def forward(self, state):
        value = self.critic(state)
        return value
