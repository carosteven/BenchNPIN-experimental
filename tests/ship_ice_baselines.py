"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import numpy as np
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC
from benchnpin.baselines.ship_ice_nav.td3.policy import ShipIceTD3
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchnpin.common.metrics.base_metric import BaseMetric


""" ============================== Policy Training ========================================"""
ppo_policy = ShipIcePPO()
ppo_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)

sac_policy = ShipIceSAC()
sac_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)



""" ============================== Policy Benchmark ========================================"""
benchmark_results = []
num_eps = 10

ppo_policy = ShipIcePPO()
sac_policy = ShipIceSAC()
planning_policy = PlanningBasedPolicy(planner_type='lattice')

benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='300000'))
benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='130000'))
# benchmark_results.append(planning_policy.evaluate(num_eps=num_eps))                   # we currently skip planning-based for benchmarking

BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')
