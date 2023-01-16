#instal·lació de llibreries.
import warnings
import time

import random
import sys
from time import time
from collections import deque, defaultdict, namedtuple
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

!pip install gym[box2d]

env = gym.make('LunarLander-v2')
env.seed(0)

print("- Rang de recompenses o llindar de les recompenses: {} ".format(env.reward_range))
print("- Màxim nombre de passos per episodi: {} ".format(env.spec.max_episode_steps)) 
print("- Espai d'accions: {} ".format(env.action_space.n))
print("- Espai d'accions: {} ".format(env.action_space))
print("- Espai d'estats: {} ".format(env.continuous))


""" def play_games(num_games):    
    steps_list = []
    total_reward_list = []    
    for i_game in range(num_games): 
        if i_game % 2 == 0:
            print("\rEpisode {}/{}.".format(i_game, num_games), end="")
            sys.stdout.flush()
            
        total_reward, steps, done = 0, 0, False
        env.reset()
        while not done:
            env.render(mode='rgb_array')
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            if done:    
                steps_list.append(steps)
                total_reward_list.append(total_reward)
    return steps_list, total_reward_list      
          
steps_list, total_reward_list = play_games(1000)
env.close()  """