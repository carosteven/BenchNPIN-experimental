"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
import argparse
from benchnpin.baselines.area_clearing.ppo.policy import AreaClearingPPO
from benchnpin.baselines.area_clearing.sac.policy import AreaClearingSAC
from benchnpin.baselines.area_clearing.td3.policy import AreaClearingTD3
from benchnpin.baselines.area_clearing.sam.policy import AreaClearingSAM
from benchnpin.common.utils.utils import DotDict
from os.path import dirname
from benchnpin.common.utils.utils import DotDict
from os.path import dirname

import os

def main(args):
    cfg = DotDict.load_from_file(args.config_file)
    if cfg.train.train_mode:
    #     for cfg_key in ['train', 'env', 'agent', 'rewards']:
    #         info = getattr(cfg, cfg_key).items()
    #         print(f'\n{cfg_key.upper()} CONFIGURATION')
    #         for key, value in info:
    #             print(f'{key}: {value}')

        if cfg.train.resume_training:
            model_name = cfg.train.job_id_to_resume
        else:
            model_name = f'{cfg.train.job_name}_{args.job_id}'
    elif cfg.evaluate.eval_mode:
        model_name = cfg.evaluate.model
    # ========================= PPO Policy =====================================
    # ppo_policy = AreaClearingPPO()
    # ppo_policy.train(total_timesteps=int(4e5), checkpoint_freq=10000)
    # evaluations = ppo_policy.evaluate(num_eps=5)

    # ppo_policy = AreaClearingPPO(model_path='/Storage2/m5ramesh/git/BenchNPIN/benchnpin/baselines/area_clearing/ppo/final_models/clear_env_small/')
    # evaluations = ppo_policy.evaluate(num_eps=5, model_eps='280000') # For small - 280000 Intuitively performing model! For large - V2-260000 is pretty close
    # print("PPO Eval: ", evaluations)


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

    # ========================= SAM Policy =====================================
    sam_policy = AreaClearingSAM(model_name=model_name, cfg=args.config_file)
    # sam_policy.train(job_id=args.job_id, **cfg.train)
    evaluations = sam_policy.evaluate(num_eps=5)
    # evaluations = td3_policy.evaluate(num_eps=5, model_eps='latest')
    # print("PPO Eval: ", evaluations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for box pushing navigation'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to the config file',
        default=f'{dirname(dirname(__file__))}/benchnpin/environments/area_clearing/config.yaml'
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    main(parser.parse_args())
