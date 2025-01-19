"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import numpy as np
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC
from benchnpin.baselines.ship_ice_nav.td3.policy import ShipIceTD3
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy

# ========================= PPO Policy =====================================
# ppo_policy = ShipIcePPO()
# ppo_policy.train(total_timesteps=500)
# evaluations = ppo_policy.evaluate(num_eps=5, model_eps='300')
# print("PPO Eval: ", evaluations)


# ========================= SAC Policy =====================================
# sac_policy = ShipIceSAC()
# sac_policy.train(total_timesteps=500)
# evaluations = sac_policy.evaluate(num_eps=5, model_eps='300')
# print("PPO Eval: ", evaluations)


# ========================= PPO Policy =====================================
# td3_policy = ShipIceTD3()
# td3_policy.train(total_timesteps=500)
# evaluations = td3_policy.evaluate(num_eps=5, model_eps='latest')
# print("PPO Eval: ", evaluations)


# ========================= Planning-based Policy =====================================
planning_policy = PlanningBasedPolicy(planner_type='lattice')
evaluations = planning_policy.evaluate(num_eps=5)
print("Plannning Based Eval: ", evaluations)
