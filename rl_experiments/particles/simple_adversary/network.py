
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class AdversaryPolicy(nn.Module):
    """Policy for adversary player."""
    def __init__(self, num_landmarks=2, num_good=2, action_dim=2, std=0.1):
        super(AdversaryPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(2 * (num_landmarks + num_good), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.std_dev = torch.ones(action_dim) * 0.5

        # self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        self.apply(init_weights)

    def forward(self, state):
        mu = self.actor(state)
        # std = self.log_std.exp().expand_as(mu)

        return torch.distributions.Normal(mu, self.std_dev)


class AdversaryCritic(nn.Module):
    """Critic for adversary player."""
    def __init__(self, num_landmarks=2, num_good=2):
        super(AdversaryCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(2 * (num_landmarks + num_good), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.critic(state)



class PlayerPolicy(nn.Module):
    """Policy for adversary player."""
    def __init__(self, num_landmarks=2, num_good=2, action_dim=2, std=0.1):
        super(PlayerPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(2 * (1 + num_landmarks + num_good), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.std_dev = torch.ones(action_dim) * 0.5

        # self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        self.apply(init_weights)

    def forward(self, state):
        mu = self.actor(state)
        # std = self.log_std.exp().expand_as(mu)

        return torch.distributions.Normal(mu, self.std_dev)


class PlayerCritic(nn.Module):
    """Critic for adversary player."""
    def __init__(self, num_landmarks=2, num_good=2):
        super(PlayerCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(2 * (1 + num_landmarks + num_good), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.apply(init_weights)

    def forward(self, state):
        return self.critic(state)
