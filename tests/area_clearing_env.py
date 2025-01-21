"""
An example script for running box pushing or ship ice environment
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np


env = gym.make('area-clearing-v0')
env.reset()

for i in range(1000):

    action = 0
    observation, reward, terminated, truncated, info = env.step(action)

    env.render()

    # print(observation.shape)
