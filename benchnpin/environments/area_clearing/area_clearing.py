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

from benchnpin.environments.area_clearing.utils import round_up_to_even, position_to_pixel_indices
from scipy.ndimage import distance_transform_edt, rotate as rotate_image
from skimage.morphology import disk, binary_dilation
import spfa

from cv2 import fillPoly

R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

BOUNDARY_PENALTY = -50
TERMINAL_REWARD = 200

LOCAL_MAP_PIXEL_WIDTH = 96
LOCAL_MAP_WIDTH = 10 # 10 meters
LOCAL_MAP_PIXELS_PER_METER = LOCAL_MAP_PIXEL_WIDTH / LOCAL_MAP_WIDTH

OBSTACLE_SEG_INDEX = 0
FLOOR_SEG_INDEX = 1
RECEPTACLE_SEG_INDEX = 3
CUBE_SEG_INDEX = 4
ROBOT_SEG_INDEX = 5
MAX_SEG_INDEX = 8

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

        # observation
        self.num_channels = 2
        self.observation = None
        self.global_overhead_map = None
        self.small_obstacle_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None

        # robot state channel
        self.agent_info = self.cfg.agent
        self.robot_radius = ((self.agent_info.length**2 + self.agent_info.width**2)**0.5 / 2) * 1.2
        robot_pixel_width = int(2 * self.robot_radius * LOCAL_MAP_PIXELS_PER_METER)
        self.robot_state_channel = np.zeros((LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32)
        start = int(np.floor(LOCAL_MAP_PIXEL_WIDTH / 2 - robot_pixel_width / 2))
        for i in range(start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                # Circular robot mask
                if (((i + 0.5) - LOCAL_MAP_PIXEL_WIDTH / 2)**2 + ((j + 0.5) - LOCAL_MAP_PIXEL_WIDTH / 2)**2)**0.5 < robot_pixel_width / 2:
                    self.robot_state_channel[i, j] = 1

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
            self.observation_space = spaces.Box(low=-10, high=30, shape=(self.cfg.num_obstacles * 2,), dtype=np.float64)
        else:
            self.observation_shape = (self.num_channels, LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH)
            self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float64)

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  
        self.boundary_violation_limit = self.cfg.occ.map_width / 4       # if the ship is out of boundary more than this limit, terminate and truncate the episode 

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

        self.state_fig, self.state_ax = plt.subplots(1, self.num_channels, figsize=(4 * self.num_channels, 6))
        self.colorbars = [None] * self.num_channels
    
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

        # Initialize configuration space (only need to compute once)
        self.update_configuration_space()

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

        # reset map
        self.global_overhead_map = self.create_padded_room_zeros()
        self.update_global_overhead_map()

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
                if len(contact_pts.points) > 1:
                    collision_with_static_or_walls = True
                    break
                
        if self.agent.body.position.x < 0 and abs(self.agent.body.position.x - 0) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
        if self.agent.body.position.x > self.cfg.occ.map_width and abs(self.agent.body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
            boundary_violation_terminal = True

        for obs in self.static_obs_shapes + self.wall_shapes:
            contact_pts = self.agent.shapes_collide(obs)
            if len(contact_pts.points) > 1:
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
        truncated = self.t >= self.t_max

        # apply constraint penalty
        if failure:
            reward += BOUNDARY_PENALTY

        # apply terminal reward
        if (terminated or truncated) and not failure:
            reward += TERMINAL_REWARD
        
        done = terminated or truncated

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
            
            observation = self.generate_observation(done=done)
            self.observation = observation

        self.update_global_overhead_map()
        
        return observation, reward, terminated, truncated, info


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

    def generate_observation(self, done=False):
        self.update_global_overhead_map()

        if done:
            return None
        
        # Overhead map
        channels = []
        channels.append(self.get_local_map(self.global_overhead_map, self.agent.body.position, self.agent.body.angle))
        channels.append(self.robot_state_channel)
        # channels.append(self.get_local_distance_map(self.create_global_shortest_path_map(self.agent.body.position), self.agent.body.position, self.agent.body.angle))
        try:
            observation = np.stack(channels)
        except Exception as e:
            print(e)
            print(channels[0].shape, channels[1].shape)
            raise e
        return observation
    
    def create_padded_room_zeros(self):
        room_width = self.max_x_boundary - self.min_x_boundary
        room_length = self.max_y_boundary - self.min_y_boundary
        return np.zeros((
            int(2 * np.ceil((room_width * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2)),
            int(2 * np.ceil((room_length * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def update_global_overhead_map(self):
        small_overhead_map = self.small_obstacle_map.copy()

        for vertices in self.obstacles:
            vertices_np = np.array(vertices)

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
            vertices_px[:, 0] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
            vertices_px[:, 1] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
            vertices_px[:, 1] = small_overhead_map.shape[0] - vertices_px[:, 1]

            fillPoly(small_overhead_map, [vertices_px], color=CUBE_SEG_INDEX/MAX_SEG_INDEX)

        start_i, start_j = int(self.global_overhead_map.shape[0] / 2 - small_overhead_map.shape[0] / 2), int(self.global_overhead_map.shape[1] / 2 - small_overhead_map.shape[1] / 2)
        self.global_overhead_map[start_i:start_i + small_overhead_map.shape[0], start_j:start_j + small_overhead_map.shape[1]] = small_overhead_map

    def get_local_distance_map(self, global_map, robot_position, robot_heading):
        local_map = self.get_local_map(global_map, robot_position, robot_heading)
        local_map -= local_map.min() # move the min to 0 to make invariant to size of environment
        return local_map
    
    def get_local_map(self, global_map, robot_position, robot_heading):
        crop_width = round_up_to_even(LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2))
        rotation_angle = 90 - np.degrees(robot_heading)
        pixel_i = int(np.floor(-robot_position[1] * LOCAL_MAP_PIXELS_PER_METER + global_map.shape[0] / 2))
        pixel_j = int(np.floor(robot_position[0] * LOCAL_MAP_PIXELS_PER_METER + global_map.shape[1] / 2))
        crop = global_map[pixel_i - crop_width // 2:pixel_i + crop_width // 2, pixel_j - crop_width // 2:pixel_j + crop_width // 2]
        rotated_crop = rotate_image(crop, rotation_angle, order=0)
        local_map = rotated_crop[
            rotated_crop.shape[0] // 2 - LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[0] // 2 + LOCAL_MAP_PIXEL_WIDTH // 2,
            rotated_crop.shape[1] // 2 - LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[1] // 2 + LOCAL_MAP_PIXEL_WIDTH // 2
        ]
        return local_map
    
    def get_local_overhead_map(self):
        rotation_angle = -np.degrees(self.agent.body.angle) + 90
        pos_y = int(np.floor(self.global_overhead_map.shape[0] / 2 - self.agent.body.position.y * LOCAL_MAP_PIXELS_PER_METER))
        pos_x = int(np.floor(self.global_overhead_map.shape[1] / 2 + self.agent.body.position.x * LOCAL_MAP_PIXELS_PER_METER))
        mask = rotate_image(np.zeros((LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32), rotation_angle, order=0)
        y_start = pos_y - int(mask.shape[0] / 2)
        y_end = y_start + mask.shape[0]
        x_start = pos_x - int(mask.shape[1] / 2)
        x_end = x_start + mask.shape[1]
        crop = self.global_overhead_map[y_start:y_end, x_start:x_end]
        crop = rotate_image(crop, rotation_angle, order=0)
        y_start = int(crop.shape[0] / 2 - LOCAL_MAP_PIXEL_WIDTH / 2)
        y_end = y_start + LOCAL_MAP_PIXEL_WIDTH
        x_start = int(crop.shape[1] / 2 - LOCAL_MAP_PIXEL_WIDTH / 2)
        x_end = x_start + LOCAL_MAP_PIXEL_WIDTH
        return crop[y_start:y_end, x_start:x_end]
    
    def create_global_shortest_path_map(self, robot_position):
        pixel_i, pixel_j = position_to_pixel_indices(robot_position[0], robot_position[1], self.configuration_space.shape, LOCAL_MAP_PIXELS_PER_METER)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        global_map, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        global_map /= LOCAL_MAP_PIXELS_PER_METER
        global_map /= (np.sqrt(2) * LOCAL_MAP_PIXEL_WIDTH) / LOCAL_MAP_PIXELS_PER_METER
        # global_map *= self.cfg.env.shortest_path_channel_scale
        return global_map
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]
    
    def update_configuration_space(self):
        """
        Obstacles are dilated based on the robot's radius to define a collision-free space
        """

        obstacle_map = self.create_padded_room_zeros()
        small_obstacle_map = np.zeros((LOCAL_MAP_PIXEL_WIDTH+20, LOCAL_MAP_PIXEL_WIDTH+20), dtype=np.float32)

        for vertices in self.obstacles:
            # get world coordinates of vertices
            # vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            # vertices_np = np.array([[v.x, v.y] for v in vertices])
            vertices_np = np.array(vertices)

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
            vertices_px[:, 0] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
            vertices_px[:, 1] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
            vertices_px[:, 1] = small_obstacle_map.shape[0] - vertices_px[:, 1]

            fillPoly(small_obstacle_map, [vertices_px], color=1)

            # # draw the boundary on the small_obstacle_map
            # if poly.label in ['wall', 'divider', 'column', 'corner']:
            #     fillPoly(small_obstacle_map, [vertices_px], color=1)
        
        start_i, start_j = int(obstacle_map.shape[0] / 2 - small_obstacle_map.shape[0] / 2), int(obstacle_map.shape[1] / 2 - small_obstacle_map.shape[1] / 2)
        obstacle_map[start_i:start_i + small_obstacle_map.shape[0], start_j:start_j + small_obstacle_map.shape[1]] = small_obstacle_map

        # Dilate obstacles and walls based on robot size
        robot_pixel_width = int(2 * self.robot_radius * LOCAL_MAP_PIXELS_PER_METER)
        selem = disk(np.floor(robot_pixel_width / 2))
        self.configuration_space = 1 - binary_dilation(obstacle_map, selem).astype(np.float32)
        
        selem_thin = disk(np.floor(robot_pixel_width / 4))
        self.configuration_space_thin = 1 - binary_dilation(obstacle_map, selem_thin).astype(np.float32)

        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        self.small_obstacle_map = 1 - small_obstacle_map

    
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

                for ax, i in zip(self.state_ax, range(self.num_channels)):
                    ax.clear()
                    ax.set_title(f'Channel {i}')
                    im = ax.imshow(self.observation[i,:,:], cmap='hot', interpolation='nearest')
                    if self.colorbars[i] is not None:
                        self.colorbars[i].update_normal(im)
                    else:
                        self.colorbars[i] = self.state_fig.colorbar(im, ax=ax)
                
                self.state_fig.savefig(os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '_obs.png'))
        else:
            self.renderer.render(save=False)


    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        pass
