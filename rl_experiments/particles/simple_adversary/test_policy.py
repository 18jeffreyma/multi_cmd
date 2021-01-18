# PyTorch imports.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Additional imports.
import numpy as np
import sys, os, time

# Import game utilities
from network import AdversaryPolicy, PlayerPolicy
from multi_cmd.rl_utils.critic_functions import critic_update, get_advantage

# Import multiplayer CMD RL optimizer.
# from copg_optim import CoPG
from multi_cmd.optim import cmd_utils
from multi_cmd.optim import potentials

# Get env utility from OpenAI package...
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# Initialize game environment.
scenario = scenarios.load("simple_adversary.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                    scenario.observation)


# Initialize policy for both agents for matching pennies.

folder_location = 'tensorboard/'
experiment_name = 'copg/'
directory = folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(folder_location + experiment_name + 'data')


t_eps = 8000

a = AdversaryPolicy()
a.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/agent0_' + str(t_eps) + ".pth")
)

p1 = PlayerPolicy()
p1.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/agent1_' + str(t_eps) + ".pth")
)

p2 = PlayerPolicy()
p2.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/agent1_' + str(t_eps) + ".pth")
)


while(True):
    obs = env.reset()
    for i in range(25):
        env.render()

        a_dist = a(torch.FloatTensor(obs[0]))
        p1_dist = p1(torch.FloatTensor(obs[1]))
        p2_dist = p2(torch.FloatTensor(obs[2]))

        print(a_dist.mean)

        a_sample = a_dist.sample()
        p1_sample = p1_dist.sample()
        p2_sample = p2_dist.sample()

        a_action = np.array([0, a_sample[0], 0, a_sample[1], 0])
        p1_action = np.array([0, p1_sample[0], 0, p1_sample[1], 0])
        p2_action = np.array([0, p2_sample[0], 0, p2_sample[1], 0])

        # Record actions and advance game state...
        actions = np.stack([a_action, p1_action, p2_action])
        obs, rewards, dones, _ = env.step(actions)

        time.sleep(0.1)
    input()
