# Import game utilities from marlenv package.
from multi_cmd.envs.markov_soccer import MarkovSoccer
from network import policy, critic

import torch
import numpy as np

# Initialize game environment.
env = MarkovSoccer()

# Load policy from saved checkpoints.
p1 = policy()
p1.load_state_dict(torch.load("model/try1/actor1_18300.pth", map_location=torch.device('cpu')))
policies = [p1 for _ in range(4)]

# Reset environment for visualization.
obs = env.reset()
dones = [False for _ in range(len(policies))]

# Run one trajectory of the game.
while (True):
    print('====================')
    # no move, up, right, down, left
    # Render environment.
    env.render()

    # Sample actions from policy.
    actions = []
    for i in range(len(policies)):

        obs_gpu = torch.tensor([obs[i]], dtype=torch.float32)
        dist = policies[i](obs_gpu)
        print(MarkovSoccer.COLORS[i] + ":", dist.probs)
        action = dist.sample().numpy()
        # TODO(jjma): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
        if action.ndim == 1 and action.size == 1:
            action = action[0]
        actions.append(action)

    # Advance environment one step forwards.
    obs, rewards, dones, _ = env.step(actions)

    input()

    # Break once all players are done.
    if all(dones):
        break



env.close()
