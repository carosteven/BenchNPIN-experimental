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
import pickle


""" ============================== Policy Training ========================================"""
# ppo_policy = ShipIcePPO()
# ppo_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)

# sac_policy = ShipIceSAC()
# sac_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)



""" ============================== Policy Benchmark ========================================"""
benchmark_results = []
num_eps = 200

ppo_policy = ShipIcePPO()
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='300000'))
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='350000'))
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='400000'))
benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='420000'))          # best PPO
# benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps='500000'))


sac_policy = ShipIceSAC()
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='110000'))
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='180000'))
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='200000'))
# benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='220000'))
benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps='240000'))          # best SAC


# planning_policy = PlanningBasedPolicy(planner_type='lattice')
# benchmark_results.append(planning_policy.evaluate(num_eps=num_eps))

# planning_policy = PlanningBasedPolicy(planner_type='predictive')
# benchmark_results.append(planning_policy.evaluate(num_eps=num_eps))

BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')

# save eval results to disk
pickle_dict = {
    'benchmark_results': benchmark_results
}
with open('shipIce_benchmark_results.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)
