
import time
import numpy as np

def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation,
                            scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation)
    return env

env = make_env('simple_adversary')
obs = env.reset()

for ob in obs:
    print(ob)

print(env.action_space)
while (True):
    # [__, right, left, up, down]
    env.render()
    time.sleep(1)
    env.step(np.array([[0, 0, 0.1, 0, 0], [0, 0.1, 0, 0, 0], [0, 0.1, 0, 0, 0]]))


for agent_ob in obs:
    print(agent_ob)
