from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from benchnpin.common.metrics.base_metric import BaseMetric

import numpy as np
import networkx as nx

class TaskDrivenMetric(BaseMetric):
    """
    Reference to paper "Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments"
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8954627
    """

    def __init__(self, alg_name, robot_mass) -> None:
        super().__init__(alg_name=alg_name)

        self.eps_reward = 0

        # NOTE in contrast to the Interactive Gibson Benchmark, for ship ice navigation environment, we keep track of the mass motion distance 
        # instead of displacement. The concept of displacement is to penalize environment disturbance, 
        # which is more applicable to indoor environments, less suitable for an ice field.
        self.total_mass_dist = 0               # \sum_{i=1}^{k}m_il_i

        self.robot_mass = robot_mass          # m_0
        self.total_robot_dist = 0                    # l_0
    
    def compute_mst_cost_for_successful_boxes(self):
        """
        Compute the minimum spanning tree for successful boxes
        """

        if not any(self.box_completed_statuses):
            return -1

        # compute the minimum spanning tree for successful boxes
        G = nx.Graph()
        completed_box_ids = [i for i, status in enumerate(self.box_completed_statuses) if status]
        completed_boxes = [self.all_boxes[i] for i in completed_box_ids]
        completed_box_positions = [box.centroid for box in completed_boxes]
        completed_box_positions = [[pos.x, pos.y] for pos in completed_box_positions]

        robot_node_id = len(completed_box_positions)
        goal_node_id = len(completed_box_positions) + 1

        G.add_nodes_from(range(len(completed_box_positions)))
        G.add_node(robot_node_id)
        G.add_node(goal_node_id)
        for i in range(len(completed_box_positions)):
            for j in range(i+1, len(completed_box_positions)):
                G.add_edge(i, j, weight=np.linalg.norm(np.array(completed_box_positions[i]) - np.array(completed_box_positions[j])))

        # add robot node
        for i in range(len(completed_box_positions)):
            G.add_edge(robot_node_id, i, weight=np.linalg.norm(np.array(self.initial_robot_state[:2]) - np.array(completed_box_positions[i])))

        # add goal node
        for i in range(len(completed_box_positions)):
            min_dist = np.inf
            for j in range(len(self.goal_positions)):
                dist = np.linalg.norm(np.array(completed_box_positions[i]) - np.array(self.goal_positions[j]))
                min_dist = min(min_dist, dist)
            G.add_edge(i, goal_node_id, weight=min_dist)

        mst = nx.minimum_spanning_tree(G)

        mst_cost = sum([mst.edges[edge]['weight'] for edge in mst.edges])
        return mst_cost

    def compute_efficiency_score(self, mst_cost):
        """
        Compute 1_{success} * (L / ship_dist)
        """

        success_rate = sum(self.box_completed_statuses) / len(self.box_completed_statuses)

        return success_rate * mst_cost / self.total_robot_dist


    def compute_effort_score(self, mst_cost):
        """
        Compute (m_0 * l_0) / (\sum_{i=0}^k m_i * l_i)
        """

        min_mass_dist = 0

        for obstacle in self.all_boxes:
            area = obstacle.area
            min_dist = np.inf
            for goal in self.goal_positions:
                min_dist = min(min_dist, np.linalg.norm(np.array([goal[0], goal[1]]) - np.array([obstacle.centroid.x, obstacle.centroid.y])))
            min_mass_dist += min_dist * area

        effort = (self.robot_mass * mst_cost) / (self.robot_mass * self.total_robot_dist) + (min_mass_dist / self.total_mass_dist)
        return effort

    
    def update(self, info, reward, eps_complete=False):
        self.eps_reward += reward

        self.total_mass_dist = info['total_work']
        self.box_completed_statuses = info['box_completed_statuses']

        # compute robot motion distance
        cur_robot_state = info['state']
        self.total_robot_dist += np.linalg.norm(np.array(self.robot_state[:2]) - np.array(cur_robot_state[:2]))
        self.robot_state = cur_robot_state
        
        if eps_complete:
            self.rewards.append(self.eps_reward)
            mst_cost = self.compute_mst_cost_for_successful_boxes()
            self.efficiency_scores.append(self.compute_efficiency_score(mst_cost))
            self.effort_scores.append(self.compute_effort_score(mst_cost))


    def reset(self, info):
        self.eps_reward = 0
        self.total_mass_dist = 0
        self.total_robot_dist = 0
        self.trial_success = False

        self.robot_state = info['state']
        self.initial_robot_state = info['state']

        self.all_boxes, self.goal_positions = info['obs'], info['goal_positions']

        self.all_boxes = [Polygon(box) for box in self.all_boxes]
        self.goal_positions = [[goal.x, goal.y] for goal in self.goal_positions]