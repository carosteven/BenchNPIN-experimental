import pickle

from benchnpin.common.metrics.base_metric import BaseMetric

pickle_files = ['area_clearing_benchmark_results.pkl']

benchmark_results = []

for file in pickle_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        benchmark_results.extend(data['benchmark_results'])

BaseMetric.plot_algs_scores_task_driven(benchmark_results, save_fig_dir='./')
