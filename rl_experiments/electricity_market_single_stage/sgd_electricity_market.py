# Normal Python Imports:
import os, sys
from multi_cmd.rl_utils.single_state_multi_copg import SingleStateMultiSimGD

# Import PyTorch and training wrapper for Multi CoPG.
import torch
from multi_cmd.optim import potentials
torch.backends.cudnn.benchmark = True

# Import game environment (snake env is called "envs").
from multi_cmd.envs.electricity_market import ProbabilisticElectricityMarket

# Import policy and critic network.
from network import policy, critic

# Import log utilities.
from torch.utils.tensorboard import SummaryWriter

# Training settings (CHECK THESE BEFORE RUNNING).
device = torch.device('cuda:0')
# device = torch.device('cpu') # Uncomment to use CPU.
batch_size = 32
n_steps = 30000
run_id = "small_lf_sgd1"
verbose = False

# Create log directories and specify Tensorboard writer.
model_location = 'model'
run_location = os.path.join(model_location, run_id)
if not os.path.exists(run_location):
    os.makedirs(run_location)

# Initialize game environment.
env = ProbabilisticElectricityMarket(game_end_prob=1.)
dtype = torch.float32

policies = [policy().to(device) for _ in range(3)]

# Tensorboard writer initialization.
logs_location = os.path.join(run_location, 'tensorboard')
if not os.path.exists(logs_location):
    os.makedirs(logs_location)
writer = SummaryWriter(logs_location)

# Define training environment with env provided.
train_wrap = SingleStateMultiSimGD(
    env,
    policies,
    tol=1e-6,
    batch_size=batch_size,
    device=device,
    policy_lr=1e-5
)

print('device:', device)
print('batch_size:', batch_size)
print('n_steps:', n_steps)


for t_eps in range(n_steps):
    print(t_eps)
    # Sample and compute update.
    states, actions, action_mask, rewards, done = train_wrap.sample(verbose=verbose)
    train_wrap.step(states, actions, action_mask, rewards, done, verbose=verbose)
    print("avg traj length", len(done[0])/batch_size)

    if True:
        print("logging progress:", t_eps + 1)

        # Calculating discounted average reward for current sample.
        disc_avg_reward = []
        for i in range(3):
            total_sum = 0.
            cumsum = 0.
            for j in range(len(rewards[i])):
                cumsum *= 1.0
                cumsum += rewards[i][j].cpu().item()
                if (done[i][j] == 0):
                    total_sum += cumsum
                    cumsum = 0
            disc_avg_reward.append(total_sum/batch_size)

        # Log values to Tensorboard.
        writer.add_scalar('agent1/disc_avg_reward', disc_avg_reward[0], t_eps)
        writer.add_scalar('agent2/disc_avg_reward', disc_avg_reward[1], t_eps)
        writer.add_scalar('agent3/disc_avg_reward', disc_avg_reward[2], t_eps)
        writer.add_scalar('game/avg_max_trajectory length', len(done[0]) / batch_size, t_eps)
