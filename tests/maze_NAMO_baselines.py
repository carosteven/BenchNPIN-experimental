"""
An example script for training and evaluating baselines for the maze NAMO task
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import numpy as np
from benchnpin.baselines.maze_NAMO.ppo.policy import MazeNAMOPPO
from benchnpin.baselines.maze_NAMO.sac.policy import MazeNAMOSAC
from benchnpin.common.metrics.base_metric import BaseMetric
import pickle



""" ============================== Policy Training ========================================"""
# ppo_policy = MazeNAMOPPO()
# ppo_policy.train(total_timesteps=int(15e5), checkpoint_freq=10000)

""" ============================== Policy Benchmark ========================================"""
benchmark_results = []
num_eps = 200

ppo_policy = MazeNAMOPPO()
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='880000'))
benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='1030000'))
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='1290000'))
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='1300000'))

sac_policy = MazeNAMOSAC()
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='60000'))
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='80000'))
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='100000'))
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='120000'))
benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='140000'))

BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')


# save eval results to disk
pickle_dict = {
    'benchmark_results': benchmark_results
}
with open('maze_benchmark_results.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)
