"""
An example script for running baseline planners for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.ship_ice_nav.planners.lattice import LatticePlanner
from benchnpin.baselines.ship_ice_nav.planners.predictive import PredictivePlanner
from benchnpin.common.controller.dp import DP

env = gym.make('ship-ice-v0')
env = env.unwrapped

planner_type = 'predictive'             # set planner type here. 'lattice' or 'predictive'

lattice_planner = LatticePlanner()
predictive_planner = PredictivePlanner()

total_episodes = 5
for eps_idx in range(total_episodes):

    observation, info = env.reset()
    obstacles = info['obs']

    # open-loop planning
    if planner_type == 'lattice':
        path = lattice_planner.plan(ship_pos=info['state'], goal=env.goal, obs=obstacles)

    elif planner_type == 'predictive':
        occ_map = observation[0]
        footprint = observation[1]
        path = predictive_planner.plan(ship_pos=info['state'], goal=env.goal, occ_map=occ_map, footprint=footprint, conc=env.cfg.concentration)

    else:
        raise Exception("Invalid planner. Choose a planner between 'lattice' or 'predictive'.")
    env.update_path(path)

    # setup dp controller to track the planned path
    cx = path.T[0]
    cy = path.T[1]
    ch = path.T[2]
    dp = DP(x=info['state'][0], y=info['state'][1], yaw=info['state'][2],
            cx=cx, cy=cy, ch=ch, **lattice_planner.cfg.controller)
    state = dp.state

    # start a new rollout
    t = 0
    while True:

        # call ideal controller to get angular velocity control
        omega, _ = dp.ideal_control(info['state'][0], info['state'][1], info['state'][2])
        observation, reward, terminated, truncated, info = env.step(omega)
        env.render()
        
        # update setpoint
        x_s, y_s, h_s = dp.get_setpoint()
        dp.setpoint = np.asarray([x_s, y_s, np.unwrap([state.yaw, h_s])[1]])

        if terminated or truncated:
            break

        t += 1
