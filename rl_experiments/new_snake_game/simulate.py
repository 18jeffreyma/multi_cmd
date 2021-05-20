# Import game utilities from marlenv package.
import gym, envs, sys
import marlenv
from network import policy, critic

import torch
import numpy as np

# Initialize game environment.
env = gym.make(
    'Snake-v1',
	height=20,       # Height of the grid map
	width=20,        # Width of the grid map
	num_snakes=4,    # Number of snakes to spawn on grid
	snake_length=5,  # Initail length of the snake at spawn time
	vision_range=5,  # Vision range (both width height), map returned if None
	frame_stack=1,   # Number of observations to stack on return
    **{
        'num_fruits': 4,
        'reward_dict': {
            'fruit': 1.0,
            'kill': 10.0,
            'lose': -1.0,
            'win': 1.0,
            'time': 0.,
        }
    }
)

# Load policy from saved checkpoints.
p1 = policy()
p2 = policy()
p3 = policy()
p4 = policy()
p1.load_state_dict(torch.load("model/sgd_try4/actor0_10250.pth", map_location=torch.device('cpu')))
p2.load_state_dict(torch.load("model/sgd_try4/actor1_10250.pth", map_location=torch.device('cpu')))
p3.load_state_dict(torch.load("model/sgd_try4/actor2_10250.pth", map_location=torch.device('cpu')))
p4.load_state_dict(torch.load("model/sgd_try4/actor3_10250.pth", map_location=torch.device('cpu')))

policies = [p1, p2, p3, p4]

# Reset environment for visualization.
obs = env.reset()
dones = [False for _ in range(len(policies))]

# Run one trajectory of the game.
while (True):
    # Render environment.
    env.render(mode='gif')

    # Sample actions from policy.
    actions = []
    for i in range(len(policies)):
        obs_gpu = torch.tensor([obs[i]], dtype=torch.float32)
        dist = policies[i](obs_gpu)

        # Green, Yellow, Red, Blue
        print(dist.probs)

        action = dist.sample().numpy()
        # TODO(jjma): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
        if action.ndim == 1 and action.size == 1:
            action = action[0]
        actions.append(action)

    # Advance environment one step forwards.
    obs, rewards, dones, _ = env.step(actions)
    print(rewards)
    if 1. in rewards:
        print('fruit_eaten')

    # Break once all players are done.
    if all(dones):
        break

    # Press Enter to advance game.
    input()

env.save_gif('simulate_test.gif')
env.close()
