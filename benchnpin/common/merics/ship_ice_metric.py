from benchnpin.common.merics.base_metric import BaseMetric
import numpy as np


class ShipIceMetric(BaseMetric):
    """
    Reference to paper "Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments"
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8954627
    """

    def __init__(self, env, alg_name) -> None:
        super().__init__(env=env, alg_name=alg_name)

        self.eps_reward = 0

        # NOTE in contrast to the Interactive Gibson Benchmark, for ship ice navigation environment, we keep track of the mass motion distance 
        # instead of displacement. The concept of displacement is to penalize environment disturbance, 
        # which is more applicable to indoor environments, less suitable for an ice field.
        self.total_mass_dist = 0               # \sum_{i=1}^{k}m_il_i

        self.ship_mass = env.cfg.ship.mass          # m_0
        self.total_ship_dist = 0                    # l_0


    def compute_efficiency_score(self):
        """
        Compute 1_{success} * (L / ship_dist)
        """

        if not self.trial_success:
            return 0
        else:
            return self.L / self.total_ship_dist


    def compute_effort_score(self):
        """
        Compute (m_0 * l_0) / (\sum_{i=0}^k m_i * l_i)
        """

        effort = (self.ship_mass * self.total_ship_dist) / (self.ship_mass * self.total_ship_dist + self.total_mass_dist)
        return effort

    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.eps_reward += reward

        self.total_mass_dist = info['total_work']
        self.trial_success = info['trial_success']

        # compute ship motion distance
        ship_state = info['state']
        self.total_ship_dist += np.linalg.norm(np.array(self.ship_state[:2]) - np.array(ship_state[:2]))
        self.ship_state = ship_state
        
        if done or truncated:
            self.rewards.append(self.eps_reward)
            self.efficiency_scores.append(self.compute_efficiency_score())
            self.effort_scores.append(self.compute_effort_score())

        return obs, reward, done, truncated, info


    def reset(self):
        obs, info = self.env.reset()

        self.eps_reward = 0
        self.total_mass_dist = 0
        self.total_ship_dist = 0
        self.trial_success = False

        self.ship_state = info['state']
        self.goal_line = self.env.goal[1]

        # shortest obstacle-free path length for the ship
        self.L = self.goal_line - self.ship_state[1]

        return obs, info