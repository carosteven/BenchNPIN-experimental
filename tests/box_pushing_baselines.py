"""
An example script for training and evaluating baselines for box pushing navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
from benchnpin.baselines.box_pushing.SAM.policy import BoxPushingSAM

# ========================= Spatial Action Map Policy =====================================
sam_policy = BoxPushingSAM()
sam_policy.train()
# evaluations = sam_policy.evaluate(num_eps=5, model_eps='300')
# print("sam Eval: ", evaluations)
