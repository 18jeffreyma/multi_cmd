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

# Import game utilities.
import pig

# Set up directory for results and SummaryWriter.
folder_location = 'tensorboard/'
experiment_name = 'copg/'
directory = folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(folder_location + experiment_name + 'data')

# Initialize policy for both agents for matching pennies.
p1 = policy(4, 2)
p2 = policy(4, 2)
p3 = policy(4, 2)

q = critic(4)

print('p1:', p1(torch.FloatTensor([0, 0, 0, 0])))
print('p2:', p2(torch.FloatTensor([0, 0, 0, 0])))
print('p3:', p3(torch.FloatTensor([0, 0, 0, 0])))

policy_list = [p1, p2, p3]

# Initialize game environment.
game = pig.PigGame(num_players=3)

# Initialize optimizer (changed this to new optimizer). Alpha is inverse of learning rate.
optim = cmd_utils.CMD_RL([p1.parameters(), p2.parameters(), p3.parameters()],
                         bregman=potentials.squared_distance(100))
optim_q = torch.optim.Adam(q.parameters(), lr=1e-3)

num_players = 3
num_episode = 1
batch_size = 100

for t_eps in range(1, num_episode+1):
    print('t_eps:', t_eps)
    mat_states = [[] for _ in range(num_players)]
    mat_actions = [[] for _ in range(num_players)]
    mat_rewards = [[] for _ in range(num_players)]
    mat_done = [[] for _ in range(num_players)]

    win_count = [0] * 3

    for _ in range(batch_size):
        # Randomize game order each trajectory.
        game_order = np.arange(num_players)
        np.random.shuffle(game_order)

        game.reset()

        for idx in itertools.cycle(game_order):
            game_over = False

            while (True):
                # Sample and record state.
                state = torch.FloatTensor(game.current_player_state())

                # Feed state into policy and sample and record action.
                probs = policy_list[idx](state)
                dist = Categorical(probs)
                action = dist.sample()

                # Record information
                mat_states[idx].append(state)
                mat_actions[idx].append(action)

                # Advance game with that action.
                same_player, winning_player = game.step(action)

                # If we have a winning player, game is over, otherwise no reward.
                if (winning_player is None):
                    mat_rewards[idx].append(torch.FloatTensor([0]))
                else:
                    game_over = True
                    mat_rewards[idx].append(torch.FloatTensor([0]))

                # Mask for filtering later.
                mat_done[idx].append(torch.FloatTensor([1]))

                # If not same player, break from this loop and move on to next.
                if game_over or not same_player:
                    break

            # Break from forever cycle if game is over...
            if game_over:
                break

        # Since all games are collected in a list, this just makes it easier
        # to denote splits between trajectories.
        for i in range(num_players):
            mat_done[i][-1] = torch.FloatTensor([0])


    q_outputs = [q(torch.stack(elem_list)).detach() for elem_list in mat_states]
    returns = [
        torch.cat(get_advantage(0, torch.stack(rewards), q_output, torch.cat(done_list)))
        for rewards, q_output, done_list in zip(mat_rewards, q_outputs, mat_done)
    ]

    mat_advantages = [ret - q_output.transpose(0,1)[0]
                      for ret, q_output in zip(returns, q_outputs)]

    for loss_critic, gradient_norm in critic_update(
            torch.cat([torch.stack(elem_list) for elem_list in mat_states]),
            torch.cat(returns), q, optim_q
        ):
        print(loss_critic)

    log_probs = [
        Categorical(p(torch.stack(state_list))).log_prob(torch.stack(action_list))
        for state_list, action_list, p in zip(mat_states, mat_actions, policy_list)
    ]

    s_log_probss

    print(distributions[0].shape)
