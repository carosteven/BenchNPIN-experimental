"""
An example script for running baseline policy for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.area_clearing.planning_based.policy import PlanningBasedPolicy

env = gym.make('area-clearing-v0')
env = env.unwrapped

# initialize planning policy
# planner_type = 'lattice'             # set planner type here. 'lattice' or 'predictive'
# policy = PlanningBasedPolicy(planner_type=planner_type)

# initialize RL policy
policy = PlanningBasedPolicy()

total_diff_reward = 0
total_col_reward = 0
total_scaled_col_reward = 0

total_episodes = 5
for eps_idx in range(total_episodes):

    observation, info = env.reset()
    obstacles = info['obs']

    policy.update_boundary_and_obstacles(info['boundary'], info['walls'], info['static_obstacles'])
    print_count = 0

    # start a new rollout
    while True:
        # call planning based policy
        action = policy.act(observation=(observation).astype(np.float64), agent_pos=info['state'], obstacles=obstacles)
        env.update_path(policy.path)

        scaled_action = [action[0] / env.target_speed, action[1] / env.max_yaw_rate_step]

        observation, reward, terminated, truncated, info = env.step(scaled_action)

        obstacles = info['obs']
        env.render()

        # print("reward: ", reward, "; dist reward: ", info['dist reward'], "; col reward: ", info['collision reward'], "; col reward scaled: ", info['scaled collision reward'])
        total_diff_reward += info['diff reward']
        total_col_reward += info['collision reward']
        # total_scaled_col_reward += info['scaled collision reward']

        if terminated or truncated:
            print('Terminated:', terminated)
            print('Truncated:', truncated)
            policy.reset()
            break

print(total_diff_reward, total_col_reward, total_scaled_col_reward)
