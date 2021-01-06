import gym
import envs
from utils.wrappers import Vechandler
import torch
import torch.nn as nn

env = gym.make('python_4p-v1')

env.reset()
done = False

print(env.action_space.sample())


class policy(nn.Module):
    """General policy model for calculating action policy from state."""
    def __init__(self):
        super(policy, self).__init__()
        self.actor = nn.Sequential(torch.nn.Conv2d(4, 64, 3),
                                   nn.Tanh(),
                                   torch.nn.Conv2d(64, 32, 3),
                                   nn.Tanh(),
                                   nn.Flatten(),
                                   nn.Linear(8192, 4),
                                   nn.Softmax(dim=-1))

    def forward(self, state):
        mu = self.actor(state)
        return mu

while not done:
    obs, reward, done, info = env.step(torch.FloatTensor([1, 1, 1, 1]))

    print(torch.FloatTensor(obs[0]))

    p1 = policy()

    # print(
    #     torch.distributions.Categorical(p1(torch.stack([torch.FloatTensor(obs[0])]))).sample()
    # )
    print('reward:', reward)
    print('done:', done)

    env.render()


    done = done.all()
    break


env.close()
