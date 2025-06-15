"""
An example script for running box delivery or ship ice environment
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np


# env = gym.make('ship-ice-v0')
env = gym.make('box-delivery-v0')

# Area clearing. Demo mode is set through a member function.
# env = gym.make('area-clearing-v0')
# env = gym.make('maze-NAMO-v0')
env = env.unwrapped
# env.activate_demo_mode()

env.reset()

for i in range(1):

    action = 96*(10 - 1) + 32
    # action = 48
    # input()
    observation, reward, terminated, truncated, info = env.step(action)
    input()

    env.render()
    
    if terminated or truncated:
        break
