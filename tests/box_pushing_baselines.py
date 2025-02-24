"""
An example script for training and evaluating baselines for box pushing navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
from benchnpin.baselines.box_pushing.SAM.policy import BoxPushingSAM
from benchnpin.baselines.box_pushing.ppo.policy import BoxPushingPPO
from benchnpin.baselines.box_pushing.sac.policy import BoxPushingSAC
from benchnpin.common.utils.utils import DotDict
from os.path import dirname

def main(args):
    cfg = DotDict.load_from_file(args.config_file)
    if cfg.train.resume_training:
        model_name = cfg.train.job_id_to_resume
    else:
        model_name = f'{cfg.train.job_name}_{args.job_id}'

    if cfg.train.job_type == 'sam':
        # ========================= Spatial Action Map Policy =========================
        sam_policy = BoxPushingSAM(model_name=f'large_divider', cfg=args.config_file)
        # sam_policy.train(job_id=args.job_id, **cfg.train)
        evaluations = sam_policy.evaluate(num_eps=5)
        # print("sam Eval: ", evaluations)

    elif cfg.train.job_type == 'ppo':
        # ================================ PPO Policy =================================
        # ppo_policy = BoxPushingPPO(model_name=f'ppo_model_{args.job_id}', cfg=args.config_file)
        # ppo_policy.train()
        ppo_policy = BoxPushingPPO(model_name=f's15_g1_14952355_60000_steps_60000_steps', cfg=args.config_file)
        evaluations = ppo_policy.evaluate(num_eps=5)
        # print("ppo Eval: ", evaluations)

    elif cfg.train.job_type == 'sac':
        # ================================ SAC Policy =================================
        sac_policy = BoxPushingSAC(model_name=f'sac_model_{args.job_id}', cfg=args.config_file)
        sac_policy.train()
        # sac_policy = BoxPushingSAC(model_name=f'GR2_14875179_80000_steps_80000_steps', cfg=args.config_file)
        # evaluations = sac_policy.evaluate(num_eps=5)
        # print("sac Eval: ", evaluations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for box pushing navigation'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to the config file',
        default=f'{dirname(dirname(__file__))}/benchnpin/environments/box_pushing/config_sac.yaml'
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    main(parser.parse_args())
