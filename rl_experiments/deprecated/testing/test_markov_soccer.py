from open_spiel.python import rl_environment
import numpy as np


env = rl_environment.Environment('markov_soccer')

# There are 5 planes of the board: a, A, b, B, O (where lowercase represents
# player position w/out ball, and O is the location of the ball).

time_step = env.reset()

for row in time_step.observations['info_state']:
    print('player')
    part1 = row[:20]
    part2 = row[20:40]
    part3 = row[40:60]
    part4 = row[60:80]
    part5 = row[80:100]
    part6 = row[100:]

    print('a')
    print(np.reshape(part1, (4,5)))
    print('A')
    print(np.reshape(part2, (4,5)))
    print('b')
    print(np.reshape(part3, (4,5)))
    print('B')
    print(np.reshape(part4, (4,5)))
    print('O')
    print(np.reshape(part5, (4,5)))
    print('.')
    print(np.reshape(part6, (4,5)))
    break
print('step')

time_step = env.step(np.array([3, 3]))

for row in time_step.observations['info_state']:
    print('player')
    part1 = row[:20]
    part2 = row[20:40]
    part3 = row[40:60]
    part4 = row[60:80]
    part5 = row[80:100]
    part6 = row[100:]

    print('a')
    print(np.reshape(part1, (4,5)))
    print('A')
    print(np.reshape(part2, (4,5)))
    print('b')
    print(np.reshape(part3, (4,5)))
    print('B')
    print(np.reshape(part4, (4,5)))
    print('O')
    print(np.reshape(part5, (4,5)))
    print('.')
    print(np.reshape(part6, (4,5)))
    break

print('reward')
print(time_step.rewards)
