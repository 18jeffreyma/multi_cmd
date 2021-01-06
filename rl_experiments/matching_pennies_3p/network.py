import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class policy1(nn.Module):
    def __init__(self):
        super(policy1, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.actor = nn.Parameter(torch.FloatTensor([0.99,0.01])) # 0.6, 0.4

    def forward(self):
        mu = self.sm(self.actor)
        return mu

class policy2(nn.Module):
    def __init__(self):
        super(policy2, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.actor = nn.Parameter(torch.FloatTensor([0.1,0.9]))# 0.4, 0.6

    def forward(self):
        mu = self.sm(self.actor)
        return mu

class policy3(nn.Module):
    def __init__(self):
        super(policy3, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.actor = nn.Parameter(torch.FloatTensor([0.01,0.99]))# 0.4, 0.6

    def forward(self):
        mu = self.sm(self.actor)
        return mu

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
