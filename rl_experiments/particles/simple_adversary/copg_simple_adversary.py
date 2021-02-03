# PyTorch imports.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Additional imports.
import numpy as np
import sys, os, time

# Import game utilities
from network import AdversaryPolicy, PlayerPolicy, AdversaryCritic, PlayerCritic
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


# Set up directory for results and SummaryWriter.
folder_location = 'tensorboard/'
experiment_name = 'copg/'
directory = folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(folder_location + experiment_name + 'data')

last_teps = 10000

# Initialize policy for both agents for matching pennies.
a = AdversaryPolicy()
a.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/agent0_' + str(last_teps) + ".pth")
)

# Using self play, both blue agents are same policy.
p1 = PlayerPolicy()
p1.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/agent1_' + str(last_teps) + ".pth")
)

a_critic = AdversaryCritic()
a_critic.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/critic0_' + str(last_teps) + ".pth")
)
p1_critic = PlayerCritic()
p1_critic.load_state_dict(
    torch.load(folder_location + experiment_name + 'model/critic1_' + str(last_teps) + ".pth")
)
# p2_critic = PlayerCritic()

# Initialize optimizer (changed this to new optimizer). Alpha is inverse of learning rate.
optim = cmd_utils.CMD_RL([a.parameters(), p1.parameters(), p1.parameters()],
                         bregman=potentials.squared_distance(500))

aq_optim = torch.optim.Adam(a_critic.parameters(), lr=1e-3)
pq1_optim = torch.optim.Adam(p1_critic.parameters(), lr=1e-3)
# pq2_optim = torch.optim.Adam(p2_critic.parameters(), lr=1e-2)

# Game parameters...
num_episode = 20000
batch_size = 50
horizon_length = 40

for t_eps in range(last_teps+1, num_episode+1):
    print('t_eps:', t_eps)
    mat_states = [[], [], []]
    mat_actions_t = []
    mat_log_probs = [[], [], []]
    mat_rewards_t = []
    mat_done_t = []

    for j in range(batch_size):
        # Reset environment for each trajectory in batch.
        obs = env.reset()
        done = False

        # TODO(jjma): Longer trajectory to use for game?
        for i in range(horizon_length):
            # Sample actions...
            a_dist = a(torch.FloatTensor(obs[0]))
            p1_dist = p1(torch.FloatTensor(obs[1]))
            p2_dist = p1(torch.FloatTensor(obs[2]))

            a_sample = a_dist.sample()
            p1_sample = p1_dist.sample()
            p2_sample = p2_dist.sample()

            a_action = np.array([0, a_sample[0], 0, a_sample[1], 0])
            p1_action = np.array([0, p1_sample[0], 0, p1_sample[1], 0])
            p2_action = np.array([0, p2_sample[0], 0, p2_sample[1], 0])

            # Record actions and advance game state...
            actions = np.stack([a_sample, p1_sample, p2_sample])
            env_actions = np.stack([a_action, p1_action, p2_action])
            obs, rewards, dones, _ = env.step(env_actions)

            # Intermediate reward should be zero, to encourage strategy in between.
            actual_reward = [0., 0., 0.]

            # Mark end of horizon as done...
            if (i == horizon_length - 1):
                actual_reward = rewards
                dones = [True] * 3

            # Record other important information...
            for i in range(3):
                mat_states[i].append(torch.FloatTensor(obs[i]))

            mat_rewards_t.append(torch.FloatTensor(actual_reward))
            mat_actions_t.append(torch.FloatTensor(actions))

            # Record done for calculating advantage later...
            mat_done_t.append(1 - torch.Tensor(dones))

            # env.render()


    mat_rewards_t = torch.stack(mat_rewards_t)
    mat_done_t = torch.stack(mat_done_t)
    mat_actions_t = torch.stack(mat_actions_t)

    mat_states = [torch.stack(state_list) for state_list in mat_states]
    mat_actions = mat_actions_t.transpose(0, 1)
    mat_rewards = mat_rewards_t.transpose(0, 1)
    mat_done = mat_done_t.transpose(0, 1)

    q_outputs = [c(state_list) for c, state_list in zip([a_critic, p1_critic, p1_critic], mat_states)]
    returns = [
        torch.cat(get_advantage(0, mat_rewards_t[:, i:i+1], q_output, mat_done_t[:, i:i+1] ))
        for i, q_output in enumerate(q_outputs)
    ]

    mat_advs = [ret - q_output.transpose(0,1)[0] for ret, q_output in zip(returns, q_outputs)]

    for loss_critic, gradient_norm in critic_update(mat_states[0], returns[0], a_critic, aq_optim):
        print(loss_critic)

    for loss_critic, gradient_norm in critic_update(torch.cat([mat_states[1], mat_states[2]]),
                                                    torch.cat([returns[1], returns[2]]),
                                                    p1_critic, pq1_optim):
        print(loss_critic)


    mat_advs_t = [torch.unsqueeze(adv_list, 1) for adv_list in mat_advs]

    mat_log_probs = [policy(mat_states[i]).log_prob(mat_actions[i]) for i, policy in enumerate([a, p1, p1])]
    mat_slog_probs = [elem.clone() for elem in mat_log_probs]
    for i in range(3):
        for j in range(len(mat_slog_probs[i])):
            if j == 0 or mat_done[i][j] == 0:
                mat_slog_probs[i][j] = 0.
            else:
                mat_slog_probs[i][j] = torch.add(mat_slog_probs[i][j-1], mat_log_probs[i][j-1])

    hessian_losses = [0.] * 3
    for i in range(3):
        for j in range(3):
            if (i != j):
                hessian_losses[i] += -(mat_log_probs[i][1:] * mat_log_probs[j][1:] * mat_advs_t[i][1:]).mean()
                hessian_losses[i] += -(mat_slog_probs[i][1:] * mat_log_probs[j][1:] * mat_advs_t[i][1:]).mean()
                hessian_losses[i] += -(mat_slog_probs[i][1:] * mat_slog_probs[j][1:] * mat_advs_t[i][1:]).mean()

    gradient_losses = [-(mat_log_probs[i][1:] * mat_advs_t[i][1:]).mean() for i in range(3)]

    print('finish_traj')

    optim.zero_grad()

    # Negative objectives, since optimizer minimizes by default.
    optim.step(gradient_losses, hessian_losses, cgd=True)

    print('conj gradient iter:', optim.state_dict()['last_dual_soln_n_iter'])
    print('conj gradient residual:', optim.state_dict()['last_dual_residual'])

    print('finish_optim')

    if t_eps%25==0:
        print('checkpoint (t_eps):', t_eps)

        for i, p in enumerate([a, p1]):
            torch.save(p.state_dict(),
                       folder_location + experiment_name + 'model/agent' + str(i) + '_' + str(
                           t_eps) + ".pth")

        for i, c in enumerate([a_critic, p1_critic]):
            torch.save(c.state_dict(),
                       folder_location + experiment_name + 'model/critic' + str(i) + '_' + str(
                           t_eps) + ".pth")
