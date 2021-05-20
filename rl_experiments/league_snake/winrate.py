# Import game utilities from marlenv package.
import gym, envs, sys
from network import policy, critic

import random
import torch
import numpy as np

# Initialize game environment.
env = gym.make('python_4p-v1')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('classic')

# Load policy from saved checkpoints.
p_cgd = policy()
p_sgd = policy()

p_cgd.load_state_dict(torch.load("model/copg_try2_lr1e-3/actor0_3000.pth", map_location=torch.device('cpu')))
p_sgd.load_state_dict(torch.load("model/sgd_try2_lr1e-3/actor0_3000.pth", map_location=torch.device('cpu')))

# policies = [p2, p2, p2, p1]

scenario_names = [
    '1 PCGD, 3 SimGD',
    '2 PCGD, 2 SimGD',
    '3 PCGD, 1 SimGD'
]

combinations = [
    [p_cgd, p_sgd, p_sgd, p_sgd],
    [p_cgd, p_cgd, p_sgd, p_sgd],
    [p_cgd, p_cgd, p_cgd, p_sgd]
]

winrate_list = []
kill_winrate_list = []
fruit_winrate_list = []

for policies in combinations:
    # Run one trajectory of the game.
    num_trajectories = 1000
    win_count = np.array([0., 0., 0., 0.])
    kill_win_count = np.array([0., 0., 0., 0.])
    fruit_win_count = np.array([0., 0., 0., 0.])
    for i in range(num_trajectories):
        # Reset environment for each trajectory.
        obs = env.reset()
        prev_dones = [False for _ in range(len(policies))]
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
                # TODO(jjma): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
                if action.ndim == 1 and action.size == 1:
                    action = action[0]
                actions.append(action)

            # Track previous dones to determine who winner was.
            prev_dones = list(dones)

            # Advance environment one step forwards.
            obs, rewards, dones, debug = env.step(actions)

            # Break once all players are done.
            if all(dones):
                break

        if 'winner' in debug:
            win_count[debug['winner']] += 1
        
        kill_win_idxs = np.concatenate(
            np.argwhere(debug['kills'] == np.amax(debug['kills'])))
        for idx in kill_win_idxs:
            kill_win_count[idx] += 1/len(kill_win_idxs)

        fruit_win_idxs = np.concatenate(
            np.argwhere(debug['fruits'] == np.amax(debug['fruits'])))
        for idx in fruit_win_idxs:
            fruit_win_count[idx] += 1/len(kill_win_idxs)

    win_percentage = win_count/num_trajectories
    kill_win_percentage = kill_win_count/num_trajectories
    fruit_win_percentage = fruit_win_count/num_trajectories

    winrate_list.append(win_percentage)
    kill_winrate_list.append(kill_win_percentage)
    fruit_winrate_list.append(fruit_win_percentage)

    print('win_percentage:', win_percentage)
    print("kill_win_percentage:", kill_win_percentage)
    print('fruit_win_percentage:', fruit_win_percentage)

env.close()

print(winrate_list)
print(kill_winrate_list)
print(fruit_winrate_list)

winrate_list_np = np.transpose(np.array(winrate_list))
kill_winrate_list_np = np.transpose(np.array(kill_winrate_list))
fruit_winrate_list_np = np.transpose(np.array(fruit_winrate_list))

data = winrate_list_np
X = np.arange(3)
fig = plt.figure(figsize=(6, 3))
ax = fig.add_axes([0,0,1,1])
ax.set_xlabel('Scenarios', fontsize=12)
ax.set_ylabel('Winrate', fontsize=12)
ax.set_xticks([0.30, 1.30, 2.30])
ax.set_xticklabels(scenario_names)

LIGHT_BLUE = (0.18, 0.42, 0.41)
ORANGE = (0.85, 0.55, 0.13)


ax.bar(X + 0.01, data[0], color =[LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.21, data[1], color =[ORANGE, LIGHT_BLUE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.41, data[2], color =[ORANGE, ORANGE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.61, data[3], color =[ORANGE, ORANGE, ORANGE], width = 0.18, edgecolor='none')

custom_lines = [matplotlib.lines.Line2D([0], [0], color=LIGHT_BLUE, lw=8),
                matplotlib.lines.Line2D([0], [0], color=ORANGE, lw=8)]
ax.legend(custom_lines, ['PCGD', 'CGD'], fontsize=8)

plt.savefig("snake_winrate.png", bbox_inches="tight")

data = kill_winrate_list_np
X = np.arange(3)
fig = plt.figure(figsize=(6, 3))
ax = fig.add_axes([0,0,1,1])
ax.set_xlabel('Scenarios', fontsize=12)
ax.set_ylabel('Winrate', fontsize=12)
ax.set_xticks([0.30, 1.30, 2.30])
ax.set_xticklabels(scenario_names)

LIGHT_BLUE = (0.18, 0.42, 0.41)
ORANGE = (0.85, 0.55, 0.13)

ax.bar(X + 0.01, data[0], color =[LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.21, data[1], color =[ORANGE, LIGHT_BLUE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.41, data[2], color =[ORANGE, ORANGE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.61, data[3], color =[ORANGE, ORANGE, ORANGE], width = 0.18, edgecolor='none')

custom_lines = [matplotlib.lines.Line2D([0], [0], color=LIGHT_BLUE, lw=8),
                matplotlib.lines.Line2D([0], [0], color=ORANGE, lw=8)]
ax.legend(custom_lines, ['PCGD', 'CGD'], fontsize=8)

plt.savefig("snake_kill_winrate.png", bbox_inches="tight")

data = fruit_winrate_list_np
X = np.arange(3)
fig = plt.figure(figsize=(6, 3))
ax = fig.add_axes([0,0,1,1])
ax.set_xlabel('Scenarios', fontsize=12)
ax.set_ylabel('Winrate', fontsize=12)
ax.set_xticks([0.30, 1.30, 2.30])
ax.set_xticklabels(scenario_names)

LIGHT_BLUE = (0.18, 0.42, 0.41)
ORANGE = (0.85, 0.55, 0.13)

ax.bar(X + 0.01, data[0], color =[LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.21, data[1], color =[ORANGE, LIGHT_BLUE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.41, data[2], color =[ORANGE, ORANGE, LIGHT_BLUE], width = 0.18, edgecolor='none')
ax.bar(X + 0.61, data[3], color =[ORANGE, ORANGE, ORANGE], width = 0.18, edgecolor='none')

custom_lines = [matplotlib.lines.Line2D([0], [0], color=LIGHT_BLUE, lw=8),
                matplotlib.lines.Line2D([0], [0], color=ORANGE, lw=8)]
ax.legend(custom_lines, ['PCGD', 'CGD'], fontsize=8)

plt.savefig("snake_fruit_winrate.png", bbox_inches="tight")