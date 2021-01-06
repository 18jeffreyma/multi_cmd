from open_spiel.python import rl_environment
import numpy as np
import pyspiel


env = rl_environment.Environment('first_sealed_auction', players=3, max_value=20)

time_step = env.reset()

print('info_state_1:')
for elem in time_step.observations['info_state']:
    print(elem)
print('rewards:', time_step.rewards)

time_step = env.step([20])
print('is_last:', time_step.last())
print('info_state_2:')
for elem in time_step.observations['info_state']:
    print(elem)

print('rewards:', time_step.rewards)

time_step = env.step([20])
print('is_last:', time_step.last())
print('info_state_2:')
for elem in time_step.observations['info_state']:
    print(elem)

print('rewards:', time_step.rewards)

time_step = env.step([20])
print('is_last:', time_step.last())
print('info_state_2:')
for elem in time_step.observations['info_state']:
    print(elem)

print('rewards:', time_step.rewards)

# time_step = env.step([1])# XXX:
# print('is_last:', time_step.last())
# print('info_state_2:')
# for elem in time_step.observations['info_state']:
#     print(elem)
#
# time_step = env.step([1])
# print('is_last:', time_step.last())
# print('info_state_2:')
# for elem in time_step.observations['info_state']:
#     print(elem)
# print('rewards:', time_step.rewards)

# time_step = env.step(np.array([0, 0, 0]))
#
# print('info_state_2:', time_step.observations['info_state'])
# print('rewards:', time_step.rewards)
