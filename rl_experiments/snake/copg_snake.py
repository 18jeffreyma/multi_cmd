# PyTorch imports.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Additional imports.
import numpy as np
import sys, os, time
import itertools

# Import game utilities
from network import policy, critic
from multi_cmd.rl_utils.critic_functions import critic_update, get_advantage

# Import multiplayer CMD RL optimizer.
# from copg_optim import CoPG
from multi_cmd.optim import cmd_utils
from multi_cmd.optim import potentials

# Import game utilities from marlenv package.
import gym, envs

# Set up directory for results and SummaryWriter.
folder_location = 'tensorboard/'
experiment_name = 'copg/'
directory = folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(folder_location + experiment_name + 'data')

# Initialize policy for both agents for matching pennies.

p1 = policy()
policy_list = [p1 for _ in range(4)]
q = critic()

# Initialize game environment.
env = gym.make('python_4p-v1')

# Initialize optimizer (changed this to new optimizer). Alpha is inverse of learning rate.
optim = cmd_utils.CMD_RL([p.parameters() for p in policy_list],
                          bregman=potentials.squared_distance(100))
optim_q = torch.optim.Adam(q.parameters(), lr=1e-2)

# Game parameters...
num_players = 4
num_episode = 1000
batch_size = 20
horizon_length = 25


for t_eps in range(1, num_episode+1):
    print('t_eps:', t_eps)
    mat_states_t = []
    mat_actions_t = []
    mat_rewards_t = []
    mat_done_t = []

    for j in range(batch_size):
        # Reset environment for each trajectory in batch.
        obs = env.reset()
        done = False

        # TODO(jjma): Longer trajectory to use for game?
        idx = 0
        while (not done):
            # Record state...
            mat_states_t.append(obs)

            # Sample actions from player policies and record...
            actions = torch.cat([
                Categorical(
                    p(torch.stack([torch.FloatTensor(obs[i])]))
                ).sample()
                for i, p in enumerate(policy_list)
            ])
            mat_actions_t.append(actions)

            obs, rewards, dones, _ = env.step(actions)

            if (idx == horizon_length - 1):
                dones = np.array([True] * 4)

            # Record done for calculating advantage later...
            mat_done_t.append(1 - dones)
            mat_rewards_t.append(rewards)

            done = dones.all()
            idx += 1

    print('finish_batch')

    mat_states_t = torch.Tensor(mat_states_t)
    mat_rewards_t = torch.Tensor(mat_rewards_t)
    mat_done_t = torch.Tensor(mat_done_t)
    mat_actions_t = torch.stack(mat_actions_t)

    mat_states = mat_states_t.transpose(0, 1)
    mat_actions = mat_actions_t.transpose(0, 1)
    mat_rewards = mat_rewards_t.transpose(0, 1)

    # Log average reward per trajectory per policy...
    avg_reward_per_traj = torch.sum(mat_rewards, 1) / batch_size
    for i in range(num_players):
        writer.add_scalar('Agent' + str(i) +  '/AvgRewardPerTrajectory',
                          avg_reward_per_traj[i], t_eps)

    # Sample from critic predictiong value function.
    q_outputs = [q(mat_states[i]) for i in range(num_players)]

    returns = [
        torch.cat(get_advantage(0, mat_rewards_t[:, i:i+1], q_output, mat_done_t[:, i:i+1] ))
        for i, q_output in enumerate(q_outputs)
    ]

    mat_advantages = [ret - q_output.transpose(0,1)[0]
                      for ret, q_output in zip(returns, q_outputs)]

    for loss_critic, gradient_norm in critic_update(
            torch.cat(mat_states_t.transpose(0, 1).unbind()),
            torch.cat(returns), q, optim_q
        ):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)

    log_probs = [
        Categorical(
            p(state_list)
        ).log_prob(action_list)
        for state_list, action_list, p in zip(mat_states, mat_actions, policy_list)
    ]

    s_log_probs = [elem.clone() for elem in log_probs]

    for j in range(num_players):
        for i in range(len(log_probs[j])):
            if i == 0 or mat_done_t[i-1][j] == 0:
                s_log_probs[j][i] = 0
            else:
                s_log_probs[j][i] = torch.add(s_log_probs[j][i-1], log_probs[j][i-1])

    hessian_losses = [0.] * num_players
    for i in range(num_players):
        for j in range(num_players):
            if (i != j):
                hessian_losses[i] += -(log_probs[i] * log_probs[j] * mat_advantages[i]).mean()
                hessian_losses[i] += -(s_log_probs[i] * log_probs[j] * mat_advantages[i]).mean()
                hessian_losses[i] += -(log_probs[i] * s_log_probs[j] * mat_advantages[i]).mean()

    gradient_losses = [-(log_probs[i] * mat_advantages[i]).mean() for i in range(num_players)]

    print('finish_traj')

    optim.zero_grad()

    # Negative objectives, since optimizer minimizes by default.
    optim.step(gradient_losses, hessian_losses, cgd=True)

    print('conj gradient iter:', optim.state_dict()['last_dual_soln_n_iter'])
    print('conj gradient residual:', optim.state_dict()['last_dual_residual'])


    print('finish_optim')

    if t_eps%10==0:
        print('checkpoint (t_eps):', t_eps)

        for i in range(num_players):
            torch.save(policy_list[i].state_dict(),
                       folder_location + experiment_name + 'model/agent' + str(i) + '_' + str(
                           t_eps) + ".pth")
