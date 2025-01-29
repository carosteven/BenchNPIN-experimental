"""
An example script for training and evaluating baselines for box pushing navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
from benchnpin.baselines.box_pushing.SAM.policy import BoxPushingSAM
from benchnpin.common.utils.utils import DotDict
from os.path import dirname

def main(args):
    # ========================= Spatial Action Map Policy =====================================
    cfg = DotDict.load_from_file(args.config_file)
    sam_policy = BoxPushingSAM()
    # sam_policy.train(job_id=args.job_id, **cfg.train)
    evaluations = sam_policy.evaluate(num_eps=5)
    # print("sam Eval: ", evaluations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for box pushing navigation'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to the config file',
        default=f'{dirname(dirname(__file__))}/benchnpin/environments/box_pushing/config.yaml'
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    main(parser.parse_args())
