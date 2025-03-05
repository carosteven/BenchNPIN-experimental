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

def main(args):
    cfg = DotDict.load_from_file(args.config_file)
    if cfg.train.train_mode:
        for cfg_key in ['train', 'env', 'agent', 'rewards']:
            info = getattr(cfg, cfg_key).items()
            print(f'\n{cfg_key.upper()} CONFIGURATION')
            for key, value in info:
                print(f'{key}: {value}')

        if cfg.train.resume_training:
            model_name = cfg.train.job_id_to_resume
        else:
            model_name = f'{cfg.train.job_name}_{args.job_id}'

        if cfg.train.job_type == 'sam':
            # ========================= Spatial Action Map Policy =========================
            sam_policy = BoxDeliverySAM(model_name=model_name, cfg=args.config_file)
            sam_policy.train(job_id=args.job_id, **cfg.train)
            # evaluations = sam_policy.evaluate(num_eps=5)
            # print("sam Eval: ", evaluations)

        elif cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            ppo_policy = BoxDeliveryPPO(model_name=model_name, cfg=args.config_file)
            ppo_policy.train(resume_training=cfg.train.resume_training, n_steps=cfg.train.n_steps, batch_size=cfg.train.batch_size)
            # evaluations = ppo_policy.evaluate(num_eps=5)
            # print("ppo Eval: ", evaluations)

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            sac_policy = BoxDeliverySAC(model_name=model_name, cfg=args.config_file)
            sac_policy.train(resume_training=cfg.train.resume_training, batch_size=cfg.train.batch_size, learning_starts=cfg.train.learning_starts)
            # sac_policy = BoxDeliverySAC(model_name=f'GR2_14875179_80000_steps_80000_steps', cfg=args.config_file)
            # evaluations = sac_policy.evaluate(num_eps=5)
            # print("sac Eval: ", evaluations)
    
    if cfg.evaluate.eval_mode:
        benchmark_results = []
        num_eps = cfg.evaluate.num_eps
        if cfg.train.job_type == 'sam':
            # ========================= Spatial Action Map Policy =========================
            for model_name, obstacle_config in zip(cfg.evaluate.models, cfg.evaluate.obs_configs):
                sam_policy = BoxDeliverySAM(model_name=model_name, cfg=args.config_file)
                benchmark_results.append(sam_policy.evaluate(obstacle_config=obstacle_config, num_eps=num_eps))

        elif cfg.train.job_type == 'ppo':
            # ================================ PPO Policy =================================    
            for model_name in cfg.evaluate.models:
                ppo_policy = BoxDeliveryPPO(model_name=model_name, cfg=args.config_file)
                benchmark_results.append(ppo_policy.evaluate(num_eps=num_eps))

        elif cfg.train.job_type == 'sac':
            # ================================ SAC Policy =================================
            for model_name in cfg.evaluate.models:
                sac_policy = BoxDeliverySAC(model_name=model_name, cfg=args.config_file)
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
        default=f'{dirname(dirname(__file__))}/benchnpin/environments/box_delivery/config_ppo.yaml'
    )

    parser.add_argument(
        '--job_id',
        type=str,
        help='slurm job id',
        default=None
    )

    main(parser.parse_args())
