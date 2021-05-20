# Normal Python Imports:
import os, sys

# Import PyTorch and training wrapper for Multi CoPG.
import torch
from multi_cmd.optim import potentials
from multi_cmd.rl_utils.league_copg import LeagueTrainingCoPG
torch.backends.cudnn.benchmark = True

# Import game environment (snake env is called "envs").
import gym, envs

# Import policy and critic network.
from network import policy, critic
from network import policy2, critic2

# Import log utilities.
from torch.utils.tensorboard import SummaryWriter

# Training settings (CHECK THESE BEFORE RUNNING).
device = torch.device('cuda:0')
# device = torch.device('cpu') # Uncomment to use CPU.
batch_size = 64
n_steps = 50000
verbose = False
run_id = "copg_try11"

# Create log directories and specify Tensorboard writer.
model_location = 'model'
run_location = os.path.join(model_location, run_id)
if not os.path.exists(run_location):
    os.makedirs(run_location)

# Initialize game environment.
env = gym.make('python_4p-v1')
dtype = torch.float32

# Specify episode number to use as last checkpoint (for loading model).
last_teps = 4250 # 2100
last_run_id = "copg_try11"

# Instantiate a policy and critic; we will use self play and a symmetric critic for this game.
# p1 = policy().to(device).type(dtype)
# q = critic().to(device).type(dtype)
# if last_teps and last_run_id:
#     run_location = os.path.join(model_location, last_run_id)
#     actor_path = os.path.join(run_location, 'actor1_' + str(last_teps) + '.pth')
#     critic_path = os.path.join(run_location, 'critic1_' + str(last_teps) + '.pth')
#     p1.load_state_dict(torch.load(actor_path))
#     q.load_state_dict(torch.load(critic_path))

num_overall = 4
policies = [policy().to(device).type(dtype) for _ in range(num_overall)]
critics = [critic().to(device).type(dtype) for _ in range(num_overall)]

# Tensorboard writer initialization.
logs_location = os.path.join(run_location, 'tensorboard')
if not os.path.exists(logs_location):
    os.makedirs(logs_location)
writer = SummaryWriter(logs_location)

# Define training environment with env provided.
train_wrap = LeagueTrainingCoPG(
    env,
    policies,
    critics,
    4,
    batch_size=batch_size,
    policy_lr=1e-3,
    critic_lr=1e-2,
    device=device,
    tol=1e-4,
)

print('device:', device)
print('batch_size:', batch_size)
print('n_steps:', n_steps)

if last_teps is None:
    last_teps = 0

for t_eps in range(last_teps, n_steps):
    # Sample and compute update.
    states, actions, action_mask, rewards, done = train_wrap.step(verbose=verbose)

    # Everything below is logging and model checkpoint, can ignore.
    if ((t_eps + 1) % 20) == 0:
        print("logging progress:", t_eps + 1)

        # Calculating discounted average reward for current sample.
        disc_avg_reward = []
        for i in range(4):
            total_sum = 0.
            cumsum = 0.
            for j in range(len(rewards[i])):
                cumsum *= 0.95
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

    #     torch.cuda.empty_cache()

    if ((t_eps + 1) % 250) == 0:
        print('saving checkpoint:', t_eps + 1)

        for i, (actor, critic) in enumerate(zip(policies, critics)):
            actor_path = os.path.join(run_location, 'actor' + str(i) + '_' + str(t_eps + 1) + '.pth')
            critic_path = os.path.join(run_location, 'critic' + str(i) + '_' + str(t_eps + 1) + '.pth')
            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)
