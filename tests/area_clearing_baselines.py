"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import numpy as np
from benchnpin.baselines.area_clearing.ppo.policy import AreaClearingPPO
from benchnpin.baselines.area_clearing.sac.policy import AreaClearingSAC
from benchnpin.baselines.area_clearing.td3.policy import AreaClearingTD3

import os

# ========================= PPO Policy =====================================
ppo_policy = AreaClearingPPO()
ppo_policy.train(total_timesteps=int(5e5), checkpoint_freq=10000)
evaluations = ppo_policy.evaluate(num_eps=5)

# ppo_policy = AreaClearingPPO(model_path='/Storage2/m5ramesh/git/BenchNPIN/benchnpin/baselines/area_clearing/ppo/final_models/clear_env_small/')
# evaluations = ppo_policy.evaluate(num_eps=5, model_eps='240000') # For small - 280000 Intuitively performing model! For large - V2-260000 is pretty close
print("PPO Eval: ", evaluations)


# ========================= SAC Policy =====================================
# sac_policy = AreaClearingSAC()
# sac_policy.train(total_timesteps=int(2e5), checkpoint_freq=10000)
# evaluations = sac_policy.evaluate(num_eps=5)
# print("SAC Eval: ", evaluations)


# ========================= TD3 Policy =====================================
# td3_policy = AreaClearingTD3()
# td3_policy.train(total_timesteps=500)
# evaluations = td3_policy.evaluate(num_eps=5, model_eps='latest')
# print("PPO Eval: ", evaluations)
