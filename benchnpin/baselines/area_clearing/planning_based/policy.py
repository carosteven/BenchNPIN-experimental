import benchnpin.environments
import gymnasium as gym
from benchnpin.baselines.base_class import BasePolicy
from benchnpin.common.controller.dp import DP
import numpy as np

from shapely.geometry import Polygon, LineString
from shapely import shortest_line

from benchnpin.baselines.area_clearing.planning_based.GTSPPlanner.transition_graph_lookup import TransitionGraphLookup
from benchnpin.baselines.area_clearing.planning_based.GTSPPlanner.solve_gtsp import GTSPSolver

import numpy as np

EXTEND_BUFFER = 2

def extend_linestring_at_start(line: LineString, distance: float) -> LineString:
    """Extend a LineString by a small distance at the first end."""
    if len(line.coords) < 2:
        raise ValueError("LineString must have at least two points")

    # Get start and end points
    start, second = np.array(line.coords[0]), np.array(line.coords[1])
    end = np.array(line.coords[-1])

    # Compute direction vectors
    start_direction = start - second

    # Normalize and scale
    start_extension = start + (start_direction / np.linalg.norm(start_direction)) * distance

    # Create new extended LineString
    new_coords = [tuple(start_extension)] + list(line.coords)
    return LineString(new_coords)


class PlanningBasedPolicy(BasePolicy):
    """
    A baseline policy for area clearing using . 
    This policy first plans a path using a ship planner and outputs actions to track the planned path.
    """

    def __init__(self) -> None:
        super().__init__()

        self.path = None

        self.boundary_vertices = None
        self.walls = None
        self.static_obstacles = None

        self.boundary_polygon = None
        self.boundary_goals = None

    def update_boundary_and_obstacles(self, boundary, walls, static_obstacles):
        self.boundary_vertices = boundary
        self.walls = walls
        self.static_obstacles = static_obstacles

        self.boundary_polygon = Polygon(self.boundary_vertices)

        self._compute_boundary_goals()

    def _compute_boundary_goals(self):
        if self.boundary_vertices is None:
            return None
        
        boundary_edges = []
        for i in range(len(self.boundary_vertices)):
            boundary_edges.append([self.boundary_vertices[i], self.boundary_vertices[(i + 1) % len(self.boundary_vertices)]])
        
        boundary_linestrings = [LineString(edge) for edge in boundary_edges]

        # remove walls from boundary
        for wall in self.walls:
            wall_polygon = LineString(wall)
            wall_polygon = wall_polygon.buffer(0.1)
            for i in range(len(boundary_linestrings)):
                boundary_linestrings[i] = boundary_linestrings[i].difference(wall_polygon)

        # convert multilinestrings to linestrings
        temp_boundary_linestrings = boundary_linestrings.copy()
        boundary_linestrings = []
        for line in temp_boundary_linestrings:
            if line.geom_type == 'MultiLineString':
                boundary_linestrings.extend(list(line.geoms))
            else:
                boundary_linestrings.append(line)

        self.boundary_goals = boundary_linestrings
    
    def plan_path(self, agent_pos, observation, obstacles=None):
        obstacles_to_push = []
        
        for obstacle in obstacles:
            obstacle_poly = Polygon(obstacle)
            is_in_boundary = self.boundary_polygon.intersects(obstacle_poly)
            if is_in_boundary:
                obstacles_to_push.append(obstacle)

        all_pushing_paths = []
        for obstacle in obstacles_to_push:
            obstacle_poly = Polygon(obstacle)
            # shrink the obstacle by a small amount
            obstacle_poly = obstacle_poly.buffer(-0.4)
            obs_pushing_paths = []
            for goal in self.boundary_goals:
                path = shortest_line(obstacle_poly, goal)
                path = extend_linestring_at_start(path, EXTEND_BUFFER)
                # round to 2 decimal places
                path = LineString(np.round(np.array(path.coords), 7))
                obs_pushing_paths.append(path)
            all_pushing_paths.append(obs_pushing_paths)

        transition_graph = TransitionGraphLookup.compute_transition_graph(agent_pos, obstacles_to_push, all_pushing_paths)

        gtsp_solver = GTSPSolver()
        output_tuple, time_found = gtsp_solver.solve_GTSP_Problem(obstacles_to_push, all_pushing_paths, transition_graph, agent_pos)
        final_nodes, transition_paths, transition_cost, transition_length, total_angle, transition_costs = output_tuple

        self.path = [agent_pos[:2]]
        for i in range(len(final_nodes)):
            self.path.append(final_nodes[i].traversal_route[0])
            self.path.append(final_nodes[i].traversal_route[-1])

    def act(self, observation, **kwargs):

        # parameters for planners
        agent_pos = kwargs.get('agent_pos', [0, 0, np.pi / 2])
        obstacles = kwargs.get('obstacles', None)

        omega = 0.0
        
        # plan a path
        if self.path is None:
            self.plan_path(agent_pos, observation, obstacles)

            # # setup dp controller to track the planned path
            # cx = self.path.T[0]
            # cy = self.path.T[1]
            # ch = self.path.T[2]
            # self.dp = DP(x=agent_pos[0], y=agent_pos[1], yaw=agent_pos[2],
            #         cx=cx, cy=cy, ch=ch, **self.lattice_planner.cfg.controller)
            # self.dp_state = self.dp.state
        
        # # call ideal controller to get angular velocity control
        # omega, _ = self.dp.ideal_control(agent_pos[0], agent_pos[1], agent_pos[2])

        # # update setpoint
        # x_s, y_s, h_s = self.dp.get_setpoint()
        # self.dp.setpoint = np.asarray([x_s, y_s, np.unwrap([self.dp_state.yaw, h_s])[1]])

        return omega


    def evaluate(self, num_eps: int, model_eps: str ='latest') -> list:
        env = gym.make('area-clearing-v0')
        env = env.unwrapped

        rewards_list = []
        for eps_idx in range(num_eps):
            print("Progress: ", eps_idx, " / ", num_eps, " episodes")
            observation, info = env.reset()
            obstacles = info['obs']
            done = truncated = False
            eps_reward = 0.0

            while True:
                action = self.act(observation=(observation / 255).astype(np.float64), ship_pos=info['state'], obstacles=obstacles, 
                                    goal=env.goal,
                                    conc=env.cfg.concentration, 
                                    action_scale=env.max_yaw_rate_step)
                observation, reward, done, truncated, info = env.step(action)
                obstacles = info['obs']
                eps_reward += reward
                if done or truncated:
                    rewards_list.append(eps_reward)
                    break

        env.close()
        return rewards_list
    
    def reset(self):
        self.path = None