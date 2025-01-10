"""
An example script for running baseline planners for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy

env = gym.make('ship-ice-v0')
env = env.unwrapped

planner_type = 'predictive'             # set planner type here. 'lattice' or 'predictive'
policy = PlanningBasedPolicy(planner_type=planner_type, goal=env.goal, conc=env.cfg.concentration, action_scale=env.max_yaw_rate_step)

total_episodes = 5
for eps_idx in range(total_episodes):

    observation, info = env.reset()
    obstacles = info['obs']

    # start a new rollout
    while True:

        action = policy.act(ship_pos=info['state'], observation=(observation / 255).astype(np.float64), obstacles=obstacles)
        observation, reward, terminated, truncated, info = env.step(action)
        obstacles = info['obs']

        env.update_path(policy.path)
        env.render()

        if terminated or truncated:
            policy.reset()
            break
