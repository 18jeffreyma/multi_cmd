from open_spiel.python import rl_environment
import numpy as np


env = rl_environment.Environment('matching_pennies_3p')

# There are 5 planes of the board: a, A, b, B, O (where lowercase represents
# player position w/out ball, and O is the location of the ball).

time_step = env.reset()

print('info_state_1:', time_step.observations['info_state'])
print('rewards:', time_step.rewards)
time_step = env.step(np.array([0, 0, 0]))

print('info_state_2:', time_step.observations['info_state'])
print('rewards:', time_step.rewards)
