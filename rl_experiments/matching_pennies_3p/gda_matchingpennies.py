# PyTorch imports.
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Additional imports.
import numpy as np
import sys, os, time

# Import game utilities
from open_spiel.python import rl_environment
from network import policy1, policy2, policy3

# Import multiplayer CMD RL optimizer.
# from copg_optim import CoPG
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
p3 = policy3()

print('p1:', p1())
print('p2:', p2())
print('p3:', p3())

# Initialize game environment.
env = rl_environment.Environment('matching_pennies_3p')

# Initialize optimizer (changed this to new optimizer). Alpha is inverse of learning rate.
optim = gda_utils.SGD([p1.parameters(), p2.parameters(), p3.parameters()], lr_list=[1/0.2]*3)

num_episode = 100
batch_size = 1000

# Iterate over number of episodes.
for t_eps in range(1, num_episode + 1):
    # Collect rewards, states and actions.
    mat_action = []

    mat_reward1 = []
    mat_reward2 = []
    mat_reward3 = []

    # Data collection:
    for i in range(batch_size):
        _ = env.reset()

        # Sample from each policy and collect action.
        pi1 = p1()
        dist1 = Categorical(pi1)
        action1 = dist1.sample()

        pi2 = p2()
        dist2 = Categorical(pi2)
        action2 = dist2.sample()

        pi3 = p3()
        dist3 = Categorical(pi3)
        action3 = dist3.sample()

        action = np.array([action1, action2, action3])
        mat_action.append(torch.FloatTensor(action))

        time_step = env.step(action)
        mat_reward1.append(torch.FloatTensor([time_step.rewards[0]]))
        mat_reward2.append(torch.FloatTensor([time_step.rewards[1]]))
        mat_reward3.append(torch.FloatTensor([time_step.rewards[2]]))

    action_all = torch.stack(mat_action)

    # Get reward for first player (2nd player reward is just negative of this)
    val1_p = torch.stack(mat_reward1).transpose(0, 1)
    val2_p = torch.stack(mat_reward2).transpose(0, 1)
    val3_p = torch.stack(mat_reward3).transpose(0, 1)

    # Define policy distributions from a forward pass of policy model.
    pi_a1_s = p1()
    dist_pi1 = Categorical(pi_a1_s)
    pi_a2_s = p2()
    dist_pi2 = Categorical(pi_a2_s)
    pi_a3_s = p3()
    dist_pi3 = Categorical(pi_a3_s)

    writer.add_scalar('Entropy/Agent1', dist1.entropy().data, t_eps)
    writer.add_scalar('Entropy/Agent2', dist2.entropy().data, t_eps)
    writer.add_scalar('Entropy/Agent3', dist3.entropy().data, t_eps)

    writer.add_scalar('Action/Agent1', torch.mean(action_all[:,0]), t_eps)
    writer.add_scalar('Action/Agent2', torch.mean(action_all[:,1]), t_eps)
    writer.add_scalar('Action/Agent3', torch.mean(action_all[:,2]), t_eps)

    writer.add_scalar('Agent1/sm1', pi1.data[0], t_eps)
    writer.add_scalar('Agent1/sm2', pi1.data[1], t_eps)
    writer.add_scalar('Agent2/sm1', pi2.data[0], t_eps)
    writer.add_scalar('Agent2/sm2', pi2.data[1], t_eps)
    writer.add_scalar('Agent3/sm1', pi3.data[0], t_eps)
    writer.add_scalar('Agent3/sm2', pi3.data[1], t_eps)

    # Get log probabilities per player.
    log_probs1 = dist_pi1.log_prob(action_all[:, 0])
    log_probs2 = dist_pi2.log_prob(action_all[:, 1])
    log_probs3 = dist_pi3.log_prob(action_all[:, 2])

    # Calculate Hessian and Gradient objectives. Taking hessian of this
    # objective should recover the terms presented in the CoPG paper.
    # p1_hessian_obj = (log_probs1 * log_probs2 * (val1_p)).mean() + (log_probs1 * log_probs3 * (val1_p)).mean()
    # p2_hessian_obj = (log_probs2 * log_probs1 * (val2_p)).mean() + (log_probs2 * log_probs3 * (val2_p)).mean()
    # p3_hessian_obj = (log_probs3 * log_probs1 * (val3_p)).mean() + (log_probs3 * log_probs2 * (val3_p)).mean()

    p1_gradient_obj = (log_probs1 * val1_p).mean()
    p2_gradient_obj = (log_probs2 * val2_p).mean()
    p3_gradient_obj = (log_probs3 * val3_p).mean()

    optim.zero_grad()

    # Negative objectives, since optimizer minimizes by default.
    optim.step([-p1_gradient_obj, -p2_gradient_obj, -p3_gradient_obj])

    if t_eps%10==0:
        print('t_eps:', t_eps)
        print('p1 policy', p1())
        print('p2 policy', p2())
        print('p2 policy', p3())

        torch.save(p1.state_dict(),
                   folder_location + experiment_name + 'model/agent1_' + str(
                       t_eps) + ".pth")
        torch.save(p2.state_dict(),
                   folder_location + experiment_name + 'model/agent2_' + str(
                       t_eps) + ".pth")
        torch.save(p3.state_dict(),
                   folder_location + experiment_name + 'model/agent3_' + str(
                       t_eps) + ".pth")

print('final policy probabilities:')
print('p1 policy', p1())
print('p2 policy', p2())
print('p2 policy', p3())
