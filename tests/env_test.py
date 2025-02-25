"""
An example script for running box pushing or ship ice environment
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np


# env = gym.make('ship-ice-v0')
# env = gym.make('box-pushing-v0')

# Area clearing. Demo mode is set through a member function.
env = gym.make('area-clearing-v0')
# env = gym.make('maze-NAMO-v0')
env = env.unwrapped
env.activate_demo_mode()

env.reset()

for i in range(500):

    action = 0
    observation, reward, terminated, truncated, info = env.step(action)

    env.render()

    print(observation.shape)
    
    if terminated or truncated:
        break
