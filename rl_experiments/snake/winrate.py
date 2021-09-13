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
p1.load_state_dict(torch.load("model/sgd_try1_lr1e-3/actor1_15000.pth", map_location=torch.device('cpu')))
p2.load_state_dict(torch.load("model/copg_try1_lr1e-3/actor1_15000.pth", map_location=torch.device('cpu')))
# policies = [p2, p1, p1, p1]
# policies = [p2, p2, p2, p1]
policies = [p2, p2, p1, p1]



# Run one trajectory of the game.
num_trajectories = 250
win_count = [0 for _ in policies]
for i in range(num_trajectories):
    print(i)


    # Reset environment for each trajectory.
    obs = env.reset()
    dones = [False for _ in range(len(policies))]

    rewards = [0. for _ in policies]
    while (True):
        # Sample actions from policy.
        actions = []
        for i in range(len(policies)):
            obs_gpu = torch.tensor([obs[i]], dtype=torch.float32)
            dist = policies[i](obs_gpu)
            # print(dist.probs)
            action = dist.sample().numpy()
            # TODO(anonymous): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
            if action.ndim == 1 and action.size == 1:
                action = action[0]
            actions.append(action)

        # Advance environment one step forwards.
        obs, rewards, dones, _ = env.step(actions)
        # print(rewards)

        # Break once all players are done.
        if all(dones):
            break
    print(rewards)
    winner_idx = list(rewards).index(max(rewards))
    win_count[winner_idx] += 1

win_percentage = [elem/num_trajectories for elem in win_count]
print(win_percentage)
env.close()
