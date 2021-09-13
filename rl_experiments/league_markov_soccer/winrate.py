# Import game utilities from marlenv package.
from multi_cmd.envs.markov_soccer import MarkovSoccer
# Import game utilities from marlenv package.
import gym
from network import policy, critic

import torch
import numpy as np

# Initialize game environment.
env = MarkovSoccer()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('classic')

# Load policy from saved checkpoints.
p_cgd1 = policy()
p_cgd2 = policy()
p_cgd3 = policy()
p_cgd4 = policy()

p_sgd1 = policy()
p_sgd2 = policy()
p_sgd3 = policy()
p_sgd4 = policy()

p_cgd1.load_state_dict(torch.load("model/copg_batch16_lr1e3_try1/actor0_10000.pth", map_location=torch.device('cpu')))
p_cgd2.load_state_dict(torch.load("model/copg_batch16_lr1e3_try1/actor1_10000.pth", map_location=torch.device('cpu')))
p_cgd3.load_state_dict(torch.load("model/copg_batch16_lr1e3_try1/actor2_10000.pth", map_location=torch.device('cpu')))
p_cgd4.load_state_dict(torch.load("model/copg_batch16_lr1e3_try1/actor3_10000.pth", map_location=torch.device('cpu')))

p_sgd1.load_state_dict(torch.load("model/sgd_batch16_lr1e3_try1/actor0_10000.pth", map_location=torch.device('cpu')))
p_sgd2.load_state_dict(torch.load("model/sgd_batch16_lr1e3_try1/actor1_10000.pth", map_location=torch.device('cpu')))
p_sgd3.load_state_dict(torch.load("model/sgd_batch16_lr1e3_try1/actor2_10000.pth", map_location=torch.device('cpu')))
p_sgd4.load_state_dict(torch.load("model/sgd_batch16_lr1e3_try1/actor3_10000.pth", map_location=torch.device('cpu')))

# policies = [p2, p2, p2, p1]

scenario_names = [
    '1 PCGD, 3 SimGD',
    '2 PCGD, 2 SimGD',
    '3 PCGD, 1 SimGD'
]

combinations = [
    [p_cgd1, p_sgd2, p_sgd3, p_sgd4],
    [p_cgd1, p_cgd2, p_sgd3, p_sgd4],
    [p_cgd1, p_cgd2, p_cgd3, p_sgd4]
]

combinations = [
    [p_cgd1, p_sgd1, p_sgd1, p_sgd1],
    [p_cgd1, p_cgd1, p_sgd1, p_sgd1],
    [p_cgd1, p_cgd1, p_cgd1, p_sgd1]
]

winrate_list = []
steal_winrate_list = []

for policies in combinations:
    # Run one trajectory of the game.
    num_trajectories = 100
    win_count = np.array([0., 0., 0., 0.])
    steal_win_count = np.array([0., 0., 0., 0.])

    for i in range(num_trajectories):
        print(i)
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
                print(dist.probs)
                action = dist.sample().numpy()
                # TODO(anonymous): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
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
        
        steal_win_idxs = np.concatenate(
            np.argwhere(debug['steals'] == np.amax(debug['steals'])))
        for idx in steal_win_idxs:
            steal_win_count[idx] += 1/len(steal_win_idxs)


    win_percentage = win_count/num_trajectories
    steal_win_percentage = steal_win_count/num_trajectories

    winrate_list.append(win_percentage)
    steal_winrate_list.append(steal_win_percentage)

    print('win_percentage:', win_percentage)
    print("steal_win_percentage:", steal_win_percentage)

env.close()

winrate_list_np = np.transpose(np.array(winrate_list))
steal_winrate_list_np = np.transpose(np.array(steal_winrate_list))


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
ax.legend(custom_lines, ['PCGD', 'SimGD'], fontsize=10)

plt.savefig("soccer_winrate.png", bbox_inches="tight")

data = steal_winrate_list_np
data = np.transpose([[0.418, 0.201, 0.199, 0.278], [0.322, 0.337, 0.165, 0.145], [0.298, 0.285, 0.299, 0.118]])
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
ax.legend(custom_lines, ['PCGD', 'SimGD'], fontsize=10)

plt.savefig("soccer_steal_winrate.png", bbox_inches="tight")
