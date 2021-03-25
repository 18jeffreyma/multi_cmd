# Normal Python Imports:
import os, sys

# Import PyTorch and training wrapper for Multi CoPG.
import torch
from multi_cmd.optim import potentials
from multi_cmd.rl_utils import MultiCoPG
torch.backends.cudnn.benchmark = True

# Import game environment (snake env is called "envs").
import gym, envs

# Import policy and critic network.
from network import policy, critic

# Import log utilities.
from torch.utils.tensorboard import SummaryWriter


# Training settings (CHECK THESE BEFORE RUNNING).
device = torch.device('cuda:2')
# device = torch.device('cpu') # Uncomment to use CPU.
batch_size = 10
n_steps = 50000
verbose = False
run_id = "try2"

# Create log directories and specify Tensorboard writer.
writer = SummaryWriter(folder_location)
model_location = 'model'
run_location = os.path.join(model_location, run_id)
logs_location = os.path.join(run_location, 'tensorboard')
if not os.path.exists(folder_location):
    os.makedirs(run_location)
    os.makedirs(logs_location)

# Initialize game environment.
env = gym.make('python_4p-v1')
dtype = torch.float32

# Specify episode number to use as last checkpoint (for loading model).
last_teps = None # 2100
last_run_id = None

# Instantiate a policy and critic; we will use self play and a symmetric critic for this game.
p1 = policy().to(device).type(dtype)
if last_teps and last_run_id:
    actor_path = os.path.join(run_location, 'actor1_' + str(last_teps) + '.pth')
    p1.load_state_dict(torch.load(actor_path))
policies = [p1 for _ in range(4)]

q = critic().to(device).type(dtype)
if last_teps and last_run_id:
    critic_path = os.path.join(run_location, 'critic1_' + str(last_teps) + '.pth')
    q.load_state_dict(torch.load(critic_path))

# Define training environment with env provided.
train_wrap = MultiCoPG(
    env,
    policies,
    [q],
    batch_size=batch_size,
    self_play=True,
    potential=potentials.squared_distance(1/0.005),
    critic_lr=1e-3,
    device=device
)

print('device:', device)
print('batch_size:', batch_size)
print('n_steps:', n_steps)

for t_eps in range(last_teps, n_steps):
    # Sample and compute update.
    states, actions, action_mask, rewards, done = train_wrap.sample(verbose=verbose)
    train_wrap.step(states, actions, action_mask, rewards, done, verbose=verbose)

    if ((t_eps + 1) % 20) == 0:
        print("logging progress:", t_eps + 1)

        # Calculating discounted average reward for current sample.
        disc_avg_reward = []
        for i in range(4):
            total_sum = 0.
            cumsum = 0.
            for j in range(len(rewards[i])):
                cumsum *= 0.99
                cumsum += rewards[i][j].cpu().item()
                if (done[i][j] == 0):
                    total_sum += cumsum
                    cumsum = 0
            disc_avg_reward.append(total_sum/batch_size)

        # Log values to Tensorboard.
        writer.add_scalar('agent1/disc_avg_reward', disc_avg_reward[0], t_eps)
        writer.add_scalar('agent2/disc_avg_reward', disc_avg_reward[1], t_eps)
        writer.add_scalar('agent3/disc_avg_reward', disc_avg_reward[2], t_eps)
        writer.add_scalar('agent4/disc_avg_reward', disc_avg_reward[3], t_eps)
        writer.add_scalar('game/avg_max_trajectory length', len(done[0]) / batch_size, t_eps)

    if ((t_eps + 1) % 100) == 0:
        print('saving checkpoint:', t_eps + 1)
        actor_path = os.path.join(run_location, 'actor1_' + str(t_eps + 1) + '.pth')
        critic_path = os.path.join(run_location, 'critic1_' + str(t_eps + 1) + '.pth')
        torch.save(p1.state_dict(), actor_path)
        torch.save(q.state_dict(), critic_path)
