"""
An example script for training and evaluating baselines for ship ice navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
from benchnpin.baselines.ship_ice_nav.ppo.policy import ShipIcePPO
from benchnpin.baselines.ship_ice_nav.sac.policy import ShipIceSAC
from benchnpin.baselines.ship_ice_nav.planning_based.policy import PlanningBasedPolicy
from benchnpin.common.metrics.base_metric import BaseMetric
from benchnpin.common.utils.utils import DotDict
from os.path import dirname
import pickle

def main(user_config, job_id):
    cfg = DotDict.load_from_file(f'{dirname(dirname(__file__))}/benchnpin/environments/maze_NAMO/config.yaml')
    # Update the base configuration with the user provided configuration
    for cfg_type in user_config:
        if cfg_type in cfg:
            if type(user_config[cfg_type]) is dict:
                for param in user_config[cfg_type]:
                    cfg[cfg_type][param] = user_config[cfg_type][param]
            else:
                cfg[cfg_type] = user_config[cfg_type]

    if cfg.train.train_mode:

        model_name = cfg.train.job_name

        if cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            ppo_policy = ShipIcePPO(model_name=model_name, cfg=cfg)
            ppo_policy.train(total_timesteps=cfg.train.total_timesteps, checkpoint_freq=cfg.train.checkpoint_freq)

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            sac_policy = ShipIceSAC(model_name=model_name, cfg=cfg)
            sac_policy.train(total_timesteps=cfg.train.total_timesteps, checkpoint_freq=cfg.train.checkpoint_freq)

    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        for policy_type, model_eps in zip(cfg.evaluate.policy_types, cfg.evaluate.model_eps):
            cfg.train.job_type = policy_type

            if policy_type == 'ppo':
                # ================================ PPO Policy =================================    
                ppo_policy = ShipIcePPO(cfg=cfg)
                benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps, model_eps=model_eps))

            elif policy_type == 'sac':
                # ================================ SAC Policy =================================
                sac_policy = ShipIceSAC(cfg=cfg)
                benchmark_results.append(sac_policy.evaluate(num_eps=num_eps, model_eps=model_eps))

            elif policy_type == 'lattice':
                # ================================ Lattice Planning Policy =================================
                lattice_planning_policy = PlanningBasedPolicy(planner_type='lattice')
                benchmark_results.append(lattice_planning_policy.evaluate(num_eps=num_eps))

            elif policy_type == 'predictive':
                # ================================ Predictive Planning Policy =================================
                predictive_planning_policy = PlanningBasedPolicy(planner_type='predictive')
                benchmark_results.append(predictive_planning_policy.evaluate(num_eps=num_eps))
        
        BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./')

        # save eval results to disk
        pickle_dict = {
            'benchmark_results': benchmark_results
        }
        with open('shipIce_benchmark_results.pkl', 'wb') as f:
                pickle.dump(pickle_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for maze navigation'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='path to the config file',
        default=None
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    job_id = parser.parse_args().job_id

    if parser.parse_args().config_file is not None:
        config = DotDict.load_from_file(parser.parse_args().config_file)


    else:
        # High level configuration for the box delivery task
        config={
                'seed': 1,
                'egocentric_obs': False,
                'planner': 'predictive', # options are 'skeleton', 'straight', or 'lattice'
                'render': {
                    'show': True,           # if true show animation plots
                    'show_obs': False,       # if true show observation
                },
                'train': {
                    'train_mode': False,
                    'job_type': 'ppo', # 'ppo', 'sac'
                    'job_name': 'shipice_ppo',
                    'total_timesteps': int(15e5),
                    'checkpoint_freq': 10000,
                },
                'evaluate': {
                    'eval_mode': True,
                    'num_eps': 1,
                    'policy_types': [], # list of policy types to evaluate
                    'action_types': [], # list of action types to evaluate
                    'models': [], # list of model names to evaluate
                }
            }

    main(config, job_id)

