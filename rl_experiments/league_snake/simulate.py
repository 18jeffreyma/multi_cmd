# Import game utilities from marlenv package.
import gym, envs, sys
from network import policy, critic

import torch
import numpy as np

# Initialize game environment.
env = gym.make('python_4p-v1')

# Load policy from saved checkpoints.
p1 = policy()
p2 = policy()
p3 = policy()
p4 = policy()
p1.load_state_dict(torch.load("model/copg_try1_lr1e-3/actor1_100.pth", map_location=torch.device('cpu')))
p2.load_state_dict(torch.load("model/copg_try1_lr1e-3/actor2_100.pth", map_location=torch.device('cpu')))
p1.load_state_dict(torch.load("model/copg_try1_lr1e-3/actor3_100.pth", map_location=torch.device('cpu')))
p2.load_state_dict(torch.load("model/copg_try1_lr1e-3/actor4_100.pth", map_location=torch.device('cpu')))
policies = [p1 for _ in range(3)] + [p2]

# Reset environment for visualization.
obs = env.reset()
dones = [False for _ in range(len(policies))]

# Run one trajectory of the game.
while (True):
    # Render environment.
    env.render()

    # Sample actions from policy.
    actions = []
    for i in range(len(policies)):
        obs_gpu = torch.tensor([obs[i]], dtype=torch.float32)
        dist = policies[i](obs_gpu)
        print(dist.probs)
        action = dist.sample().numpy()
        # TODO(jjma): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
        if action.ndim == 1 and action.size == 1:
            action = action[0]
        actions.append(action)

    # Advance environment one step forwards.
    obs, rewards, dones, _ = env.step(actions)
    if 1. in rewards:
        print('fruit_eaten')

    # Break once all players are done.
    if all(dones):
        break

    # Press Enter to advance game.
    input()

env.close()
