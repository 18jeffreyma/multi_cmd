

import gym
import marlenv

env = gym.make(
    'Snake-v1',
	height=20,       # Height of the grid map
	width=20,        # Width of the grid map
	num_snakes=4,    # Number of snakes to spawn on grid
	snake_length=5,  # Initail length of the snake at spawn time
	vision_range=5,  # Vision range (both width height), map returned if None
	frame_stack=1,   # Number of observations to stack on return
    **{
        'num_fruits': 4,
        'reward_dict': {
            'fruit': 1.0,
            'kill': 10.0,
            'lose': -1.0,
            'win': 0.,
            'time': 0.,
        }
    }
)


obs = env.reset()

print(env.observation_space)

# for i in range(len(obs)):
#     print('i:', i)
#     for grid in obs[i]:
#         print(grid)
env.render(mode='gif')
obs, _, _, info = env.step([0, 0, 0, 0])
for i in range(8):
    print(obs[0][:, :, i])
env.render(mode='gif')
env.save_gif(fp='./output.gif')



# NHWC