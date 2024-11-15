import gym
import gym_penguin
from copy import deepcopy

env = gym.make('penguin_env-v0')

env.reset()
rand_act = env.action_space.sample()
env.step(rand_act)


new_env= gym.make('penguin_env-v0')
# new_env.model = env.model
# new_env.data = env.data
new_model= deepcopy(env.model)
new_data = deepcopy(env.data)
new_env.set_states(new_model,new_data)
print("The above mats should be same")
print(env.get_states())
print(new_env.get_states())


rand_act = new_env.action_space.sample()
new_env.reset()
new_env.step(rand_act)

print("The above mats should be different")
print(env.get_states())
print(new_env.get_states())