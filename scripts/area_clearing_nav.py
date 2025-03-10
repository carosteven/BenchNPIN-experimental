"""
An example script for running baseline policy for ship ice navigation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
from benchnpin.baselines.area_clearing.planning_based.policy import PlanningBasedPolicy

# initialize RL policy
policy = PlanningBasedPolicy()
policy.evaluate(num_eps=5)