# PyTorch imports.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Additional imports.
import numpy as np
import sys, os, time

# Import game utilities
from matching_pennies import pennies_game
from network import policy1, policy2

# Import SGD optimizer for multiple players.
from multi_cmd.optim import gda_utils

# Set up directory for results and SummaryWriter.
folder_location = 'tensorboard/'
experiment_name = 'gda/'
directory = folder_location + experiment_name + 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
writer = SummaryWriter(folder_location + experiment_name + 'data')

# Initialize policy for both agents for matching pennies.
p1 = policy1()
p2 = policy2()
for p in p1.parameters():
    print(p)
for p in p2.parameters():
    print(p)

# Initialize game environment.
env = pennies_game()

# Initialize optimizer (changed this to new optimizer). Alpha is inverse of learning rate.
optim = gda_utils.SGD([p1.parameters(), p2.parameters()], lr_list=[1, 1])

num_episode = 200
batch_size = 1000

# Iterate over number of episodes.
for t_eps in range(num_episode):
    # Collect rewards, states and actions.
    mat_action = []

    mat_state1 = []
    mat_reward1 = []

    mat_state2 = []
    mat_reward2 = []

    mat_done = []

    state, _, _, _, _ = env.reset()

    # Data collection:
    for i in range(batch_size):
        # Sample from each policy and collect action.
        pi1 = p1()
        dist1 = Categorical(pi1)
        action1 = dist1.sample()

        pi2 = p2()
        dist2 = Categorical(pi2)
        action2 = dist2.sample()

        state = np.array([0, 0])
        mat_state1.append(torch.FloatTensor(state))
        mat_state2.append(torch.FloatTensor(state))

        action = np.array([action1, action2])
        mat_action.append(torch.FloatTensor(action))

        state, reward1, reward2, done, _ = env.step(action)

        mat_reward1.append(torch.FloatTensor([reward1]))
        mat_reward2.append(torch.FloatTensor([reward2]))

        mat_done.append(torch.FloatTensor([1 - done]))

    action_both = torch.stack(mat_action)

    # Get reward for first player (2nd player reward is just negative of this)
    val1_p = torch.stack(mat_reward1).transpose(0, 1)
    if val1_p.size(0) != 1:
        raise 'error'

    val2_p = torch.stack(mat_reward2).transpose(0, 1)
    if val2_p.size(0) != 1:
        raise 'error'

    # Define policy distributions from a forward pass of policy model.
    pi_a1_s = p1()
    dist_pi1 = Categorical(pi_a1_s)
    pi_a2_s = p2()
    dist_pi2 = Categorical(pi_a2_s)

    action_both = torch.stack(mat_action)

    writer.add_scalar('Entropy/Agent1', dist1.entropy().data, t_eps)
    writer.add_scalar('Entropy/agent2', dist2.entropy().data, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(action_both[:,0]), t_eps)
    writer.add_scalar('Action/agent2', torch.mean(action_both[:,1]), t_eps)

    writer.add_scalar('Agent1/sm1', pi1.data[0], t_eps)
    writer.add_scalar('Agent1/sm2', pi1.data[1], t_eps)
    writer.add_scalar('Agent2/Agent1', pi2.data[0], t_eps)
    writer.add_scalar('Agent2/agent2', pi2.data[1], t_eps)

    # Get log probabilities per player.
    log_probs1 = dist_pi1.log_prob(action_both[:, 0])
    log_probs2 = dist_pi2.log_prob(action_both[:, 1])

    p1_gradient_obj = (log_probs1 * val1_p).mean()
    p2_gradient_obj = (log_probs2 * val2_p).mean()

    optim.zero_grad()

    optim_time_start = time.time()

    # Negative objectives, since optimizer minimizes by default.
    optim.step([-p1_gradient_obj, -p2_gradient_obj])

    if t_eps%100==0:
        print('t_eps:', t_eps)
        print('p1 policy', p1())
        print('p2 policy', p2())

        torch.save(p1.state_dict(),
                   folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")

print('final policy probabilities:')
print('p1 policy', p1())
print('p2 policy', p2())
