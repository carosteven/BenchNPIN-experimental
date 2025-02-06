"""
An example script for training and evaluating baselines for the maze NAMO task
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import numpy as np
from benchnpin.baselines.maze_NAMO.ppo.policy import MazeNAMOPPO    



""" ============================== Policy Training ========================================"""
ppo_policy = MazeNAMOPPO()
ppo_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)

""" ============================== Policy Benchmark ========================================"""
benchmark_results = []
num_eps = 10
ppo_policy = MazeNAMOPPO()
benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='300000'))