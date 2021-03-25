import gym, envs
import torch
from network import policy, critic

num_players = 4

folder_location = 'tensorboard/'
experiment_name = 'copg/'
directory = folder_location + experiment_name + 'model'

p1 = policy()

policy_list = [policy() for _ in range(num_players)]

for i in range(num_players):
    policy_list[i].load_state_dict(
        torch.load(folder_location + experiment_name + 'model/agent' + str(i) + "_140.pth")
    )

env = gym.make('python_4p-v1')

obs = env.reset()
done = False

idx = 0
while not done:
    print('step: ', idx)
    env.render()

    for i, p in enumerate(policy_list):
        print('p'+str(i)+':', p(torch.stack([torch.FloatTensor(obs[i])])))

    actions = torch.cat([
        torch.distributions.Categorical(
            p(torch.stack([torch.FloatTensor(obs[i])]))
        ).sample()
        for i, p in enumerate(policy_list)
    ])

    obs, reward, dones, info = env.step(actions)

    done = dones.all()

    input()
    idx += 1

env.close()
