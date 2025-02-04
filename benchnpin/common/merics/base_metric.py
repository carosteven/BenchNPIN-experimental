from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import os
from typing import List, Tuple


class BaseMetric(ABC):
    """
    A base metric class
    """

    def __init__(self, env, alg_name) -> None:
        self.env = env

        self.rewards = []
        self.efficiency_scores = []
        self.effort_scores = []

        self.alg_name = alg_name

    
    def plot_scores(self, save_fig_dir=None):
        """
        Generate box plots for efficiency scores, effort scores, and rewards on a single algorithm
        """

        if save_fig_dir is None:
            save_fig_dir = self.env.cfg.output_dir

        fig, ax = plt.subplots()

        ax.clear()
        bp_data = [self.efficiency_scores]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Efficiency Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Efficiency Scores")
        fp = os.path.join(save_fig_dir, self.alg_name + '_efficiency.png')
        fig.savefig(fp)

        ax.clear()
        bp_data = [self.effort_scores]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Effort Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Effort Scores")
        fp = os.path.join(save_fig_dir, self.alg_name + '_effort.png')
        fig.savefig(fp)

        ax.clear()
        bp_data = [self.rewards]
        ax.boxplot(bp_data, showmeans=True)
        ax.set_title("Rewards Plot")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Rewards")
        fp = os.path.join(save_fig_dir, self.alg_name + '_rewards.png')
        fig.savefig(fp)

        plt.close('all')



    @staticmethod
    def plot_algs_scores(benchmark_results: List[Tuple[List[float], List[float], List[float], str]], save_fig_dir: str) -> None:
        """
        :param benchmark_results: a list of evaluation tuples, where each tuple is computed from policy.evaluate()
        """

        fig, ax = plt.subplots()

        # parse benchmark results
        efficiency_data = []
        effort_data = []
        reward_data = []
        alg_names = []
        for alg_efficiency, alg_effort, alg_reward, alg_name in benchmark_results:
            efficiency_data.append(alg_efficiency)
            effort_data.append(alg_effort)
            reward_data.append(alg_reward)
            alg_names.append(alg_name)
        
        ax.clear()
        ax.boxplot(efficiency_data, showmeans=True)
        ax.set_xticks(list(range(1, len(efficiency_data) + 1)))
        ax.set_xticklabels(alg_names)
        ax.set_title("Efficiency Benchmark")
        ax.set_xlabel("Efficiency Score")
        ax.set_ylabel("Algorithms")
        fp = os.path.join(save_fig_dir, 'efficiency_benchmark.png')
        fig.savefig(fp)

        ax.clear()
        ax.boxplot(effort_data, showmeans=True)
        ax.set_xticks(list(range(1, len(effort_data) + 1)))
        ax.set_xticklabels(alg_names)
        ax.set_title("Interaction Effort Benchmark")
        ax.set_xlabel("Effort Score")
        ax.set_ylabel("Algorithms")
        fp = os.path.join(save_fig_dir, 'effort_benchmark.png')
        fig.savefig(fp)

        ax.clear()
        ax.boxplot(reward_data, showmeans=True)
        ax.set_xticks(list(range(1, len(reward_data) + 1)))
        ax.set_xticklabels(alg_names)
        ax.set_title("Reward Benchmark")
        ax.set_xlabel("Rewards")
        ax.set_ylabel("Algorithms")
        fp = os.path.join(save_fig_dir, 'reward_benchmark.png')
        fig.savefig(fp)


    @abstractmethod
    def compute_efficiency_score(self):
        """
        Implement this function to compute efficiency score for a trial
        """
        raise NotImplementedError


    @abstractmethod
    def compute_effort_score(self):
        """
        Implement this function to compute interaction effort score for a trial
        """
        raise NotImplementedError


    @abstractmethod
    def step(self, action):
        """
        Implement this function for any accumulative metrics
        """
        raise NotImplementedError


    @abstractmethod
    def reset(self):
        """
        Implement this function to reset any trial-specific values upon starting a new trial
        """
        raise NotImplementedError
    