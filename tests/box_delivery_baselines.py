"""
An example script for training and evaluating baselines for box pushing navigation
Uncomment the code blocks to train/evaluate each baseline algorithms
"""
import argparse
from benchnpin.baselines.box_delivery.SAM.policy import BoxDeliverySAM
from benchnpin.baselines.box_delivery.ppo.policy import BoxDeliveryPPO
from benchnpin.baselines.box_delivery.sac.policy import BoxDeliverySAC
from benchnpin.common.metrics.base_metric import BaseMetric
from benchnpin.common.utils.utils import DotDict
from os.path import dirname
import json

def main(user_config, job_id):
    cfg = DotDict.load_from_file(f'{dirname(dirname(__file__))}/benchnpin/environments/box_delivery/config.yaml')
    # Update the base configuration with the user provided configuration
    for key in user_config:
        if key in cfg:
            for sub_key in user_config[key]:
                cfg[key][sub_key] = user_config[key][sub_key]

    if cfg.train.train_mode:
        for cfg_key in ['train', 'env', 'agent', 'rewards']:
            info = getattr(cfg, cfg_key).items()
            print(f'\n{cfg_key.upper()} CONFIGURATION')
            for key, value in info:
                print(f'{key}: {value}')

        if cfg.train.resume_training:
            model_name = cfg.train.job_id_to_resume
        else:
            model_name = f'{cfg.train.job_name}_{job_id}'

        if cfg.train.job_type == 'sam':
            # ========================= Spatial Action Map Policy =========================
            sam_policy = BoxDeliverySAM(model_name=model_name, cfg=cfg)
            sam_policy.train(job_id)

        elif cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            ppo_policy = BoxDeliveryPPO(model_name=model_name, cfg=cfg)
            ppo_policy.train(resume_training=cfg.train.resume_training, n_steps=cfg.train.n_steps, batch_size=cfg.train.batch_size)

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            sac_policy = BoxDeliverySAC(model_name=model_name, cfg=cfg)
            sac_policy.train(resume_training=cfg.train.resume_training, batch_size=cfg.train.batch_size, learning_starts=cfg.train.learning_starts)
    
    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        if cfg.train.job_type == 'sam':
            # ========================= Spatial Action Map Policy =========================
            for model_name, obstacle_config in zip(cfg.evaluate.models, cfg.evaluate.obs_configs):
                sam_policy = BoxDeliverySAM(model_name=model_name, cfg=cfg)
                benchmark_results.append(sam_policy.evaluate(obstacle_config=obstacle_config, num_eps=num_eps))

        elif cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            for model_name in cfg.evaluate.models:
                ppo_policy = BoxDeliveryPPO(model_name=model_name, cfg=cfg)
                benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps))

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            for model_name in cfg.evaluate.models:
                sac_policy = BoxDeliverySAC(model_name=model_name, cfg=cfg)
                benchmark_results.append(sac_policy.evaluate(num_eps=num_eps))
        
        # Save benchmark results to a JSON file
        results_to_save = [
            {
            "efficiency_scores": result[0],
            "effort_scores": result[1],
            "rewards": result[2],
            "algorithm": result[3],
            "success": result[4]
            }
            for result in benchmark_results
        ]

        with open(f'benchmark_results_{benchmark_results[0][3]}.json', 'w') as f:
            json.dump(results_to_save, f, indent=4)

        # Read benchmark results back from the JSON file
        with open(f'benchmark_results_{benchmark_results[0][3]}.json', 'r') as f:
            benchmark_results = json.load(f)
            
        # Convert benchmark results back to a list of tuples
        benchmark_results = [
            (
            result["efficiency_scores"],
            result["effort_scores"],
            result["rewards"],
            result["algorithm"],
            result["success"]
            )
            for result in benchmark_results
        ]
        BaseMetric.plot_algs_scores(benchmark_results, save_fig_dir='./', plot_success=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate baselines for box pushing navigation'
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
        cfg = DotDict.load_from_file(parser.parse_args().config_file)


    else:
        # High level configuration for the box delivery task
        config={
                'boxes': {
                    'num_boxes': 10,
                },
                'agent': {
                    'action_type': 'position', # 'position', 'heading', 'velocity'
                    'step_size': 2.0, # distance travelled per step in heading control
                },
                'env': {
                    'obstacle_config': 'small_empty', # options are small_empty, small_columns, large_columns, large_divider
                    'room_length': 10,
                    'room_width': 5, # 5 for small, 10 for large
                    'local_map_pixel_width': 96, # 96 is all you need for SAM, but if using a vanilla ResNet, we recommend 224
                },
                'misc': {
                    'inactivity_cutoff': 100, # number of steps to run before terminating an episode
                    'seed': 42,
                },
                'rewards': {
                    'partial_rewards_scale': 0.2, # scales distance boxes are pushed to/from goal for reward
                    'goal_reward': 1.0,
                    'collision_penalty': 0.25,
                    'non_movement_penalty': 0.25, # only appllies to position control ()
                },
                'train': {
                    'train_mode': True,
                    'job_type': 'sam', # 'sam', 'ppo', 'sac'
                    'job_name': 'SAM',
                    'resume_training': False,
                    'job_id_to_resume': None,
                },
                'evaluate': {
                    'eval_mode': True,
                    'num_eps': 5,
                    'models': [None], # list of model names to evaluate
                    'obs_configs': [None], # list of obstacle configurations to evaluate
                    # TODO implement obstacle configurations for PPO and SAC
                }
            }

    main(config, job_id)
