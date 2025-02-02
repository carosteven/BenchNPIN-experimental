import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

import random
import pymunk
from pymunk import Vec2d
from matplotlib import pyplot as plt

# ship-ice related imports
from benchnpin.common.cost_map import CostMap
from benchnpin.common.evaluation.metrics import total_work_done
from benchnpin.common.geometry.polygon import poly_area
from benchnpin.common.utils.sim_utils import generate_sim_obs, generate_sim_agent, get_color
from benchnpin.common.geometry.polygon import poly_centroid, create_polygon_from_line
from benchnpin.common.utils.utils import DotDict
from benchnpin.common.occupancy_grid.occupancy_map import OccupancyGrid
from benchnpin.common.types import ObstacleType
from benchnpin.common.utils.renderer import Renderer

from shapely.geometry import Polygon

R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

BOUNDARY_PENALTY = -50
TERMINAL_REWARD = 200

FORWARD = 0
STOP_TURNING = 1
LEFT = 2
RIGHT = 3
STOP = 4
BACKWARD = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7

class AreaClearingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        super(AreaClearingEnv, self).__init__()

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)

        # construct absolute path to the env_config folder
        cfg_file = os.path.join(self.current_dir, 'config.yaml')

        cfg = DotDict.load_from_file(cfg_file)
        self.occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)
        self.cfg = cfg

        env_cfg_file_path = os.path.join(self.current_dir, 'envs/' + cfg.env + '.yaml')

        if not os.path.exists(env_cfg_file_path):
            raise FileNotFoundError(f"Environment config file {env_cfg_file_path} not found")

        self.env_cfg = DotDict.load_from_file(env_cfg_file_path)

        self.env_max_trial = 4000
        self.beta = 500         # amount to scale the collision reward
        self.episode_idx = None
        self.path = None
        self.scatter = False

        self.target_speed = self.cfg.controller.target_speed

        # Define action space
        max_yaw_rate_step = (np.pi/2) / 15        # rad/sec
        print("max yaw rate per step: ", max_yaw_rate_step)
        self.action_space = spaces.Box(low= np.array([-self.target_speed, -max_yaw_rate_step]), 
                                       high=np.array([self.target_speed, max_yaw_rate_step]),
                                       dtype=np.float64)

        # Define observation space
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            if self.cfg.randomize_obstacles:
                self.observation_space = spaces.Box(low=-10, high=30, shape=(self.cfg.num_obstacles * 2,), dtype=np.float64)
            else:
                self.observation_space = spaces.Box(low=-10, high=30, shape=(6,), dtype=np.float64)

        else:
            self.observation_shape = (2, self.occupancy.occ_map_height, self.occupancy.occ_map_width)
            self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float64)

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  
        self.boundary_violation_limit = self.occupancy.occ_map_width / 4       # if the ship is out of boundary more than this limit, terminate and truncate the episode 

        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))

        self.demo_mode = False

        self.boundary_vertices = self.env_cfg.boundary
        self.walls = self.env_cfg.walls if 'walls' in self.env_cfg else []
        self.static_obstacles = self.env_cfg.static_obstacles if 'static_obstacles' in self.env_cfg else []

        self.env_center = (int(self.cfg.occ.map_width / 2), int(self.cfg.occ.map_height / 2))

        # move boundary to the center of the environment
        self.boundary_polygon = Polygon(self.boundary_vertices)

        self.min_x_boundary = min([x for x, y in self.boundary_vertices])
        self.max_x_boundary = max([x for x, y in self.boundary_vertices])
        self.min_y_boundary = min([y for x, y in self.boundary_vertices])
        self.max_y_boundary = max([y for x, y in self.boundary_vertices])

        self.renderer = None

        self.cleared_box_count = 0
    
    def activate_demo_mode(self):
        self.demo_mode = True
        
        self.angular_speed = 0.0
        self.angular_speed_increment = 0.005
        self.linear_speed = 0.0
        self.linear_speed_increment = 0.02

    def init_area_clearing_sim(self):

        # initialize robot clearing environment
        self.steps = self.cfg.sim.steps
        self.t_max = self.cfg.sim.t_max if self.cfg.sim.t_max else np.inf
        self.dt = self.cfg.controller.dt

        # setup pymunk environment
        self.space = pymunk.Space()  # threaded=True causes some issues
        self.space.iterations = self.cfg.sim.iterations
        self.space.gravity = self.cfg.sim.gravity
        self.space.damping = self.cfg.sim.damping

        # keep track of running total of total kinetic energy / total impulse
        # computed using pymunk api call, source code here
        # https://github.com/slembcke/Chipmunk2D/blob/edf83e5603c5a0a104996bd816fca6d3facedd6a/src/cpArbiter.c#L158-L172
        self.system_ke_loss = []   # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_ke
                                # source code in Chimpunk2D cpArbiterTotalKE
        self.total_ke = [0, []]  # keep track of both running total and ke at each collision
        self.total_impulse = [0, []]
        # keep track of running total of work
        self.total_work = [0, []]

        self.total_dis = 0 
        self.prev_state = None   

        # keep track of all the obstacles that collide with ship
        self.clln_obs = set()

        # keep track of contact points
        self.contact_pts = []

        # setup pymunk collision callbacks
        def pre_solve_handler(arbiter, space, data):
            obs_body = arbiter.shapes[1].body
            obs_body.pre_collision_KE = obs_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            self.system_ke_loss.append(arbiter.total_ke)

            self.total_ke[0] += arbiter.total_ke
            self.total_ke[1].append(arbiter.total_ke)

            self.total_impulse[0] += arbiter.total_impulse.length
            self.total_impulse[1].append(list(arbiter.total_impulse))

            if arbiter.is_first_contact:
                self.clln_obs.add(arbiter.shapes[1])

            # max of two sets of points, easy to see with a picture with two overlapping convex shapes
            # find the impact locations in the local coordinates of the ship
            for i in arbiter.contact_point_set.points:
                self.contact_pts.append(list(arbiter.shapes[0].body.world_to_local((i.point_b + i.point_a) / 2)))

        # handler = space.add_default_collision_handler()
        self.handler = self.space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        self.handler.pre_solve = pre_solve_handler
        self.handler.post_solve = post_solve_handler

        if self.cfg.render.show:
            if self.renderer is None:
                self.renderer = Renderer(self.space, env_width=self.cfg.occ.map_width, env_height=self.cfg.occ.map_height, render_scale=20, 
                        background_color=(255, 255, 255), caption="Area Clearing", 
                        centered=True,
                        clearance_boundary=self.boundary_vertices
                        )
            else:
                self.renderer.reset(new_space=self.space)
        
    def init_area_clearing_env(self):

        # generate random start point, if specified
        if self.cfg.random_start:
            x_start = (self.min_x_boundary + 1) + random.random() * ((self.max_x_boundary - self.min_x_boundary) - 2)
            self.start = (x_start, self.min_y_boundary + 1.0, np.pi / 2)
        else:
            mid_x = (self.min_x_boundary + self.max_x_boundary) / 2
            self.start = (mid_x, self.min_y_boundary + 1.0, np.pi / 2)

        self.agent_info = self.cfg.agent
        self.agent_info['start_pos'] = self.start
        self.agent_info['color'] = get_color('red')

        self.obs_dicts = self.generate_obstacles()
        obs_dicts, self.static_obs_shapes = self.generate_static_obstacles()
        self.obs_dicts.extend(obs_dicts)
        obs_dicts, self.wall_shapes = self.generate_walls()
        self.obs_dicts.extend(obs_dicts)
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if (poly_area(ob['vertices']) != 0)]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        # initialize ship sim objects
        self.dynamic_obs = [ob for ob in self.obs_dicts if ob['type'] == ObstacleType.DYNAMIC]
        self.polygons = generate_sim_obs(self.space, self.dynamic_obs, self.cfg.sim.obstacle_density)
        for p in self.polygons:
            p.collision_type = 2

        self.agent = generate_sim_agent(self.space, self.agent_info, body_type=pymunk.Body.KINEMATIC)
        self.agent.collision_type = 1

        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.polygons)

    def generate_static_obstacles(self):
        obs_dict = []
        static_obs_shapes = []
        for obstacle in self.static_obstacles:
            obs_info = {}
            obs_info['type'] = ObstacleType.STATIC
            obs_info['vertices'] = np.array(obstacle)

            shape = pymunk.Poly(self.space.static_body, obstacle, radius=0.1)
            shape.collision_type = 2
            shape.friction = 0.99

            self.space.add(shape)
            obs_dict.append(obs_info)
            static_obs_shapes.append(shape)

        return obs_dict, static_obs_shapes

    def generate_walls(self):
        obs_dict = []
        wall_shapes = []
        for wall_vertices in self.walls:
            # convert line to polygon
            wall_poly = create_polygon_from_line(wall_vertices)

            obs_info = {}
            obs_info['type'] = ObstacleType.BOUNDARY
            obs_info['vertices'] = wall_poly

            # convert np array to list
            wall_poly = [(x, y) for x, y in wall_poly]

            shape = pymunk.Poly(self.space.static_body, wall_poly, radius=0.1)
            shape.collision_type = 2
            shape.friction = 0.99

            self.space.add(shape)
            obs_dict.append(obs_info)
            wall_shapes.append(shape)

        return obs_dict, wall_shapes
    
    def generate_obstacles(self):
        obs_size = self.cfg.obstacle_size
        obstacles = []          # a list storing non-overlappin obstacle centers

        total_obs_required = self.cfg.num_obstacles
        self.num_box = self.cfg.num_obstacles
        obs_min_dist = self.cfg.min_obs_dist
        min_x = self.min_x_boundary + 1
        max_x = self.max_x_boundary - 1
        min_y = self.min_y_boundary + 1
        max_y = self.max_y_boundary - 1

        obs_count = 0
        while obs_count < total_obs_required:
            center_x = random.random() * (max_x - min_x) + min_x
            center_y = random.random() * (max_y - min_y) + min_y

            # loop through previous obstacles to check for overlap
            overlapped = False
            for prev_obs_x, pre_obs_y in obstacles:
                if ((center_x - prev_obs_x)**2 + (center_y - pre_obs_y)**2)**(0.5) <= obs_min_dist:
                    overlapped = True
                    break
            
            if not overlapped:
                obstacles.append([center_x, center_y])
                obs_count += 1
        
        # convert to obs dict
        obs_dict = []
        for obs_x, obs_y in obstacles:
            obs_info = {}
            obs_info['type'] = ObstacleType.DYNAMIC
            obs_info['centre'] = np.array([obs_x, obs_y])
            obs_info['vertices'] = np.array([[obs_x + obs_size, obs_y + obs_size], 
                                    [obs_x - obs_size, obs_y + obs_size], 
                                    [obs_x - obs_size, obs_y - obs_size], 
                                    [obs_x + obs_size, obs_y - obs_size]])
            obs_dict.append(obs_info)
        return obs_dict
        

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state and returns the initial observation."""

        if self.episode_idx is None:
            self.episode_idx = 0
        else:
            self.episode_idx += 1

        self.init_area_clearing_sim()
        self.init_area_clearing_env()

        self.t = 0

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        info = {'state': (round(self.agent.body.position.x, 2),
                                round(self.agent.body.position.y, 2),
                                round(self.agent.body.angle, 2)), 
                'total_work': self.total_work[0], 
                'obs': updated_obstacles, 
                'box_count': 0,
                'boundary': self.boundary_vertices,
                'walls': self.walls,
                'static_obstacles': self.static_obstacles}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)

        else:
            low_level_observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
            info['low_level_observation'] = low_level_observation
            
            observation = self.generate_observation()

        return observation, info
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        if self.demo_mode:

            if action == FORWARD:
                self.linear_speed = 0.3
            elif action == BACKWARD:
                self.linear_speed = -0.3
            elif action == STOP_TURNING:
                self.angular_speed = 0.0

            elif action == LEFT:
                self.angular_speed = 0.1
            elif action == RIGHT:
                self.angular_speed = -0.1

            elif action == SMALL_LEFT:
                self.angular_speed = 0.05
            elif action == SMALL_RIGHT:
                self.angular_speed = -0.05

            elif action == STOP:
                self.linear_speed = 0.0

            if abs(self.linear_speed) >= self.target_speed:
                self.linear_speed = self.target_speed*np.sign(self.linear_speed)

            # apply linear and angular speeds
            global_velocity = R(self.agent.body.angle) @ [self.linear_speed, 0]

            # apply velocity controller
            self.agent.body.angular_velocity = self.angular_speed
            self.agent.body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        else:

            # apply velocity controller
            self.agent.body.angular_velocity = action[1] / 2
            self.agent.body.velocity = action[0].tolist()

        # move simulation forward
        boundary_constraint_violated = False
        boundary_violation_terminal = False      # if out of boundary for too much, terminate and truncate the episode
        collision_with_static_or_walls = False
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)

            # apply boundary constraints
            if self.agent.body.position.x < 0 and abs(self.agent.body.position.x - 0) >= self.boundary_violation_limit:
                boundary_constraint_violated = True
            if self.agent.body.position.x > self.cfg.occ.map_width and abs(self.agent.body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
                boundary_constraint_violated = True

            for obs in self.static_obs_shapes + self.wall_shapes:
                contact_pts = self.agent.shapes_collide(obs)
                if len(contact_pts.points) > 0:
                    collision_with_static_or_walls = True
                    break
                
        if self.agent.body.position.x < 0 and abs(self.agent.body.position.x - 0) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
        if self.agent.body.position.x > self.cfg.occ.map_width and abs(self.agent.body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
            boundary_violation_terminal = True

        for obs in self.static_obs_shapes + self.wall_shapes:
            contact_pts = self.agent.shapes_collide(obs)
            if len(contact_pts.points) > 0:
                collision_with_static_or_walls = True
                break
            
        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        num_completed, all_boxes_completed = self.boxes_completed(updated_obstacles, self.boundary_polygon)
        
        if(self.cleared_box_count < num_completed):
            print("Boxes completed: ", num_completed)
            self.cleared_box_count = num_completed

        # compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        failure = boundary_constraint_violated or collision_with_static_or_walls or boundary_violation_terminal

        # # check episode terminal condition
        if all_boxes_completed:
            terminated = True
        elif failure:
            terminated = True
        else:
            terminated = False

        ### TODO: NEED TO FIGURE OUT WHAT THE REWARD function should be
        # compute reward
        # if self.agent.body.position.y < self.goal[1]:
        #     dist_reward = -1
        # else:
        #     dist_reward = 0
        dist_reward = 0
        collision_reward = -work

        reward = self.beta * collision_reward + dist_reward

        # apply constraint penalty
        if failure:
            reward += BOUNDARY_PENALTY

        # apply terminal reward
        if terminated and not failure:
            reward += TERMINAL_REWARD

        # Optionally, we can add additional info
        info = {'state': (round(self.agent.body.position.x, 2),
                                round(self.agent.body.position.y, 2),
                                round(self.agent.body.angle, 2)), 
                'total_work': self.total_work[0], 
                'collision reward': collision_reward, 
                'scaled collision reward': collision_reward * self.beta, 
                'dist reward': dist_reward, 
                'obs': updated_obstacles,
                'box_count': num_completed}
        
        # generate observation
        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
        else:
            low_level_observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
            info['low_level_observation'] = low_level_observation
            
            observation = self.generate_observation()
        
        return observation, reward, terminated, False, info


    def generate_observation_low_dim(self, updated_obstacles):
        """
        The observation is a vector of shape (num_obstacles * 2) specifying the 2d position of the obstacles
        <obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        observation = np.zeros((len(updated_obstacles) * 2))
        for i in range(len(updated_obstacles)):
            obs = updated_obstacles[i]
            center = np.abs(poly_centroid(obs))
            observation[i * 2] = center[0]
            observation[i * 2 + 1] = center[1]
        return observation


    def update_path(self, new_path):
        self.path = new_path
        self.renderer.update_path(path=self.path)
    

    def generate_observation(self):
        # compute occupancy map observation  (40, 12)
        raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, 
                        ice_binary_w=int(self.occupancy.map_width * self.cfg.occ.m_to_pix_scale), 
                        ice_binary_h=int(self.occupancy.map_height * self.cfg.occ.m_to_pix_scale))
        self.occupancy.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=None)
        occupancy = np.copy(self.occupancy.occ_map)         # (H, W)

        # compute footprint observation  NOTE: here we want unscaled, unpadded vertices
        agent_pose = (self.agent.body.position.x, self.agent.body.position.y, self.agent.body.angle)
        self.occupancy.compute_ship_footprint_planner(ship_state=agent_pose, ship_vertices=self.cfg.agent.vertices)
        footprint = np.copy(self.occupancy.footprint)       # (H, W)

        observation = np.concatenate((np.array([occupancy]), np.array([footprint])))          # (2, H, W)

        return observation

    
    def boxes_completed(self, updated_obstacles, boundary_polygon):
        """
        Returns a tuple: (int: number of boxes completed, bool: whether pushing task is complete)
        """
        completed_count = 0
        completed = False

        for obs in updated_obstacles:
            # if center[1] - self.cfg.obstacle_size >= self.cfg.goal_y:
            if not(boundary_polygon.intersects(Polygon(obs))):
                completed_count += 1
        
        if completed_count == self.num_box:
            completed = True
        
        return completed_count, completed

    def render(self, mode='human', close=False):
        """Renders the environment."""

        if self.t % self.cfg.anim.plot_steps == 0:

            path = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '.png')
            self.renderer.render(save=True, path=path)

            if self.cfg.render.log_obs and not self.low_dim_state:

                # visualize occupancy map
                self.con_ax.clear()
                occ_map_render = np.copy(self.occupancy.occ_map)
                occ_map_render = np.flip(occ_map_render, axis=0)
                self.con_ax.imshow(occ_map_render, cmap='gray')
                self.con_ax.axis('off')
                save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
                fp = os.path.join(save_fig_dir, str(self.t) + '_con.png')
                self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)

                # visualize footprint
                self.con_ax.clear()
                occ_map_render = np.copy(self.occupancy.footprint)
                occ_map_render = np.flip(occ_map_render, axis=0)
                self.con_ax.imshow(occ_map_render, cmap='gray')
                self.con_ax.axis('off')
                save_fig_dir = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
                fp = os.path.join(save_fig_dir, str(self.t) + '_footprint.png')
                self.con_fig.savefig(fp, bbox_inches='tight', transparent=False, pad_inches=0)
        else:
            self.renderer.render(save=False)


    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        pass
