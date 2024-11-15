from utils import *
from mcts import *
import gym
# import mujoco

from collections import deque
import matplotlib.pyplot as plt
# from IPython.display import clear_output

import gym_penguin

import numpy as np

import matplotlib.animation as animation
from copy import deepcopy
from math import *
import random

# env = gym.make('penguin_env-v0')

episodes = 10
rewards = []
moving_average = []

def cus_copy(game):

    new_game = gym.make('penguin_env-v0')
    game_model = deepcopy(game.model)
    game_data = deepcopy(game.data)
    new_game.set_states(game_model,game_data)

    return new_game


for e in range(episodes):

    reward_e = 0    
    game = gym.make('penguin_env-v0')
    observation = game.reset() 
    done = False
    
    new_game = cus_copy(game)
    mytree = Node(new_game, False, 0, observation, 0)
    
    print('episode #' + str(e+1))
    
    while not done:
    
        mytree, action = Policy_Player_MCTS(mytree)
        print("Chosen action: ", action)
        
        observation, reward, done, _ = game.step(action)  
                        
        reward_e = reward_e + reward
        
        # game.render('rgb_array') # uncomment this if you want to see your agent in action!
                
        if done:
            print('reward_e ' + str(reward_e))
            game.close()
            break
        
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))
    
plt.plot(rewards)
plt.plot(moving_average)
plt.show()
print('moving average: ' + str(np.mean(rewards[-20:])))