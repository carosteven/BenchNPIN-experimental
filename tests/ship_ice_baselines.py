"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import numpy as np
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchnpin.common.metrics.base_metric import BaseMetric


""" ============================== Policy Training ========================================"""
# ppo_policy = ShipIcePPO()
# ppo_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)

# sac_policy = ShipIceSAC()
# sac_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)



""" ============================== Policy Benchmark ========================================"""
benchmark_results = []
num_eps = 200

ppo_policy = ShipIcePPO()
sac_policy = ShipIceSAC()
lattice_planning_policy = PlanningBasedPolicy(planner_type='lattice')
predictive_planning_policy = PlanningBasedPolicy(planner_type='predictive')

benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps))
benchmark_results.append(sac_policy.evaluate(num_eps=num_eps))
benchmark_results.append(lattice_planning_policy.evaluate(num_eps=num_eps))
benchmark_results.append(predictive_planning_policy.evaluate(num_eps=num_eps))

BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')
