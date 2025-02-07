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
from benchnpin.common.evaluation.metrics import total_work_done, obs_to_goal_difference
from benchnpin.common.geometry.polygon import poly_area
from benchnpin.common.utils.sim_utils import generate_sim_obs, generate_sim_agent, get_color
from benchnpin.common.geometry.polygon import poly_centroid, create_polygon_from_line
from benchnpin.common.utils.utils import DotDict
from benchnpin.common.types import ObstacleType
from benchnpin.common.utils.renderer import Renderer

from shapely.geometry import Polygon, LineString

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
TRUNCATION_PENALTY = -25
TERMINAL_REWARD = 100
BOX_CLEARED_REWARD = 10
BOX_PUSHING_REWARD_MULTIPLIER = 1.5
TIME_PENALTY = -0.1

LOCAL_MAP_PIXEL_WIDTH = 144
LOCAL_MAP_WIDTH = 20 #  meters
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
        self.num_channels = 4
        self.observation = None
        self.global_overhead_map = None
        self.small_obstacle_map = None
        self.configuration_space = None
        self.configuration_space_thin = None
        self.closest_cspace_indices = None
        self.goal_point_global_map = None

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
        
        self.action_space = spaces.Box(low= np.array([-1, -1]), 
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        self.max_yaw_rate_step = max_yaw_rate_step

        # Define observation space
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            self.observation_space = spaces.Box(low=-10, high=30, shape=(self.cfg.num_obstacles * 2,), dtype=np.float32)
        else:
            self.observation_shape = (self.num_channels, LOCAL_MAP_PIXEL_WIDTH, LOCAL_MAP_PIXEL_WIDTH)
            self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  

        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))

        self.demo_mode = False

        self.outer_boundary_vertices = self.env_cfg.outer_boundary
        self.boundary_vertices = self.env_cfg.boundary
        self.walls = self.env_cfg.walls if 'walls' in self.env_cfg else []
        self.static_obstacles = self.env_cfg.static_obstacles if 'static_obstacles' in self.env_cfg else []

        # move boundary to the center of the environment
        self.boundary_polygon = Polygon(self.boundary_vertices)

        self.min_x_boundary = min([x for x, y in self.boundary_vertices])
        self.max_x_boundary = max([x for x, y in self.boundary_vertices])
        self.min_y_boundary = min([y for x, y in self.boundary_vertices])
        self.max_y_boundary = max([y for x, y in self.boundary_vertices])

        self.min_x_outer = min([x for x, y in self.outer_boundary_vertices])
        self.max_x_outer = max([x for x, y in self.outer_boundary_vertices])
        self.min_y_outer = min([y for x, y in self.outer_boundary_vertices])
        self.max_y_outer = max([y for x, y in self.outer_boundary_vertices])

        self.map_width = self.max_x_outer - self.min_x_outer
        self.map_height = self.max_y_outer - self.min_y_outer

        self.renderer = None

        self.cleared_box_count = 0

        self.state_fig, self.state_ax = plt.subplots(1, self.num_channels, figsize=(4 * self.num_channels, 6))
        self.colorbars = [None] * self.num_channels

        self.boundary_goals, self.goal_points = self._compute_boundary_goals()

    def _compute_boundary_goals(self, interpolated_points=10):
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
                boundary_linestrings.extend([ls for ls in list(line.geoms) if ls.length > 0.1])
            elif line.geom_type == 'LineString':
                if line.length > 0.1:
                    boundary_linestrings.append(line)
            else:
                raise ValueError("Invalid geometry type to handle")

        boundary_goals = boundary_linestrings
        
        # get 5 evenly spaced points on each boundary goal line
        goal_points = []
        for line in boundary_goals:
            line_length = line.length
            for i in range(int(interpolated_points)):
                goal_points.append(line.interpolate(((i + 1/2) / interpolated_points) * line_length))

        return boundary_goals, goal_points

    
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
        
        def robot_boundary_pre_solve(arbiter, space, data):
            self.robot_hit_obstacle = self.prevent_boundary_intersection(arbiter)
            return True
        
        def cube_boundary_pre_solve(arbiter, space, data):
            self.prevent_boundary_intersection(arbiter)
            return True

        # handler = space.add_default_collision_handler()
        self.handler = self.space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        self.handler.pre_solve = pre_solve_handler
        self.handler.post_solve = post_solve_handler

        self.robot_boundary_handler = self.space.add_collision_handler(1, 3)
        self.robot_boundary_handler.pre_solve = robot_boundary_pre_solve
        
        self.cube_boundary_handler = self.space.add_collision_handler(2, 3)
        self.cube_boundary_handler.pre_solve = cube_boundary_pre_solve

        if self.cfg.render.show:
            if self.renderer is None:
                self.renderer = Renderer(self.space, env_width=self.map_width + 2, env_height=self.map_height + 2, render_scale=20, 
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

        for b in self.static_obs_shapes + self.wall_shapes:
            b.collision_type = 3
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if (poly_area(ob['vertices']) != 0)]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        # initialize ship sim objects
        self.dynamic_obs = [ob for ob in self.obs_dicts if ob['type'] == ObstacleType.DYNAMIC]
        self.box_shapes = generate_sim_obs(self.space, self.dynamic_obs, self.cfg.sim.obstacle_density)
        for p in self.box_shapes:
            p.collision_type = 2

        self.agent = generate_sim_agent(self.space, self.agent_info, body_type=pymunk.Body.KINEMATIC)
        self.agent.collision_type = 1

        # Initialize configuration space (only need to compute once)
        self.update_configuration_space()

        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.box_shapes)

    def prevent_boundary_intersection(self, arbiter):
        collision = False
        normal = arbiter.contact_point_set.normal
        current_velocity = arbiter.shapes[0].body.velocity
        reflection = current_velocity - 2 * current_velocity.dot(normal) * normal

        elasticity = 0.5
        new_velocity = reflection * elasticity

        penetration_depth = arbiter.contact_point_set.points[0].distance
        if penetration_depth < 0:
            collision = True
        correction_vector = normal * penetration_depth
        arbiter.shapes[0].body.position += correction_vector

        arbiter.shapes[0].body.velocity = new_velocity

        return collision

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
        outer_boundary_walls = []
        for i in range(len(self.outer_boundary_vertices)):
            outer_boundary_walls.append([self.outer_boundary_vertices[i], self.outer_boundary_vertices[(i + 1) % len(self.outer_boundary_vertices)]])
        for wall_vertices in self.walls + outer_boundary_walls:
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
        self.global_overhead_map = self.create_padded_room_ones()
        self.update_global_overhead_map()

        self.goal_point_global_map = self.create_global_shortest_path_to_goal_points()

        self.t = 0

        self.cleared_box_count = 0

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.box_shapes)
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
            self.agent.body.angular_velocity = self.max_yaw_rate_step * action[1] / 2

            # apply linear and angular speeds
            scaled_vel = self.target_speed * np.sign(action[0])
            global_velocity = R(self.agent.body.angle) @ [scaled_vel, 0]
            self.agent.body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        # move simulation forward
        collision_with_static_or_walls = False
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)

            for obs in self.static_obs_shapes + self.wall_shapes:
                contact_pts = self.agent.shapes_collide(obs)
                if len(contact_pts.points) > 1:
                    collision_with_static_or_walls = True
                    break

        for obs in self.static_obs_shapes + self.wall_shapes:
            contact_pts = self.agent.shapes_collide(obs)
            if len(contact_pts.points) > 1:
                collision_with_static_or_walls = True
                break
            
        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.box_shapes)
        num_completed, all_boxes_completed = self.boxes_completed(updated_obstacles, self.boundary_polygon)
        
        diff_reward = obs_to_goal_difference(self.prev_obs, updated_obstacles, self.goal_points, self.boundary_polygon) * BOX_PUSHING_REWARD_MULTIPLIER
        movement_reward = 0 if abs(diff_reward) > 0 else TIME_PENALTY

        box_completion_reward = 0
        if(self.cleared_box_count < num_completed):
            print("Boxes completed: ", num_completed)
            self.cleared_box_count = num_completed
            box_completion_reward = (num_completed - self.cleared_box_count) * BOX_CLEARED_REWARD

        ### compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        collision_reward = -work

        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        failure = collision_with_static_or_walls

        # # check episode terminal condition
        if all_boxes_completed:
            terminated = True
        elif failure:
            terminated = True
        else:
            terminated = False

        reward = box_completion_reward + movement_reward
        truncated = self.t >= self.t_max

        # apply constraint penalty
        if failure:
            reward += BOUNDARY_PENALTY
        elif truncated:
            reward += TRUNCATION_PENALTY
        # apply terminal reward
        elif terminated and not failure:
            reward += TERMINAL_REWARD
        
        done = terminated or truncated

        # Optionally, we can add additional info
        info = {'state': (round(self.agent.body.position.x, 2),
                                round(self.agent.body.position.y, 2),
                                round(self.agent.body.angle, 2)), 
                'total_work': self.total_work[0], 
                'collision reward': collision_reward, 
                'diff reward': diff_reward,
                'box completed reward': box_completion_reward, 
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
        if(self.renderer):
            self.renderer.update_path(path=self.path)

    def generate_observation(self, done=False):
        self.update_global_overhead_map()

        if done:
            return None
        
        # Overhead map
        channels = []
        obs_array_1 = self.get_local_map(self.global_overhead_map, self.agent.body.position, self.agent.body.angle)
        channels.append(self.scale_obs_to_image_space(obs_array_1))
        
        obs_array_2 = self.robot_state_channel.copy()
        channels.append(self.scale_obs_to_image_space(obs_array_2))

        obs_array_3 = self.get_local_distance_map(self.create_global_shortest_path_map(self.agent.body.position), self.agent.body.position, self.agent.body.angle)
        channels.append(self.scale_obs_to_image_space(obs_array_3))

        obs_array_4 = self.get_local_distance_map(self.goal_point_global_map, self.agent.body.position, self.agent.body.angle)
        channels.append(self.scale_obs_to_image_space(obs_array_4))

        try:
            observation = np.stack(channels).astype(np.uint8)
        except Exception as e:
            print(channels[0].shape, channels[1].shape)
            raise e
        return observation
    
    def scale_obs_to_image_space(self, obs_array):
        obs_array = (obs_array * 255).astype(np.uint8)
        return obs_array
    
    def create_padded_room_zeros(self):
        return np.zeros((
            int(2 * np.ceil((self.map_width * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.map_height * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def create_padded_room_ones(self):
        return np.ones((
            int(2 * np.ceil((self.map_width * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2)),
            int(2 * np.ceil((self.map_height * LOCAL_MAP_PIXELS_PER_METER + LOCAL_MAP_PIXEL_WIDTH * np.sqrt(2)) / 2))
        ), dtype=np.float32)
    
    def update_global_overhead_map(self):
        small_overhead_map = self.small_obstacle_map.copy()
        small_overhead_map[small_overhead_map == 1] = FLOOR_SEG_INDEX/MAX_SEG_INDEX
        self.global_overhead_map[self.global_overhead_map == 1] = FLOOR_SEG_INDEX/MAX_SEG_INDEX

        for poly in self.box_shapes:
            vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            vertices_np = np.array([[v.x, v.y] for v in vertices])

            # convert world coordinates to pixel coordinates
            vertices_px = (vertices_np * LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
            vertices_px[:, 0] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
            vertices_px[:, 1] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
            vertices_px[:, 1] = small_overhead_map.shape[0] - vertices_px[:, 1]

            # draw the boundary on the small_overhead_map
            fillPoly(small_overhead_map, [vertices_px], color=CUBE_SEG_INDEX/MAX_SEG_INDEX)
        
        vertices = [self.agent.body.local_to_world(v) for v in self.agent.get_vertices()]
        robot_vertices = np.array([[v.x, v.y] for v in vertices])
        robot_vertices_px = (robot_vertices * LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
        robot_vertices_px[:, 0] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
        robot_vertices_px[:, 1] += int(LOCAL_MAP_WIDTH * LOCAL_MAP_PIXELS_PER_METER / 2) + 10
        robot_vertices_px[:, 1] = small_overhead_map.shape[0] - robot_vertices_px[:, 1]
        
        fillPoly(small_overhead_map, [robot_vertices_px], color=ROBOT_SEG_INDEX/MAX_SEG_INDEX)

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
    
    def create_global_shortest_path_map(self, robot_position):
        pixel_i, pixel_j = position_to_pixel_indices(robot_position[0], robot_position[1], self.configuration_space.shape, LOCAL_MAP_PIXELS_PER_METER)
        pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
        global_map, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
        global_map /= LOCAL_MAP_PIXELS_PER_METER
        global_map /= (np.sqrt(2) * LOCAL_MAP_PIXEL_WIDTH) / LOCAL_MAP_PIXELS_PER_METER
        # global_map *= self.cfg.env.shortest_path_channel_scale
        return global_map
    
    def create_global_shortest_path_to_goal_points(self):
        global_map = self.create_padded_room_zeros() + np.inf
        for point in self.goal_points:
            rx, ry = point.x, point.y
            pixel_i, pixel_j = position_to_pixel_indices(rx, ry, self.configuration_space.shape, LOCAL_MAP_PIXELS_PER_METER)
            pixel_i, pixel_j = self.closest_valid_cspace_indices(pixel_i, pixel_j)
            shortest_path_image, _ = spfa.spfa(self.configuration_space, (pixel_i, pixel_j))
            shortest_path_image /= LOCAL_MAP_PIXELS_PER_METER
            global_map = np.minimum(global_map, shortest_path_image)
        global_map /= (np.sqrt(2) * LOCAL_MAP_PIXEL_WIDTH) / LOCAL_MAP_PIXELS_PER_METER

        # global_map *= self.cfg.env.shortest_path_channel_scale

        global_map += 1 - self.configuration_space

        return global_map
    
    def closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]
    
    def update_configuration_space(self):
        """
        Obstacles are dilated based on the robot's radius to define a collision-free space
        """

        obstacle_map = self.create_padded_room_zeros()
        small_obstacle_map = np.zeros((LOCAL_MAP_PIXEL_WIDTH+20, LOCAL_MAP_PIXEL_WIDTH+20), dtype=np.float32)

        for poly in self.wall_shapes + self.static_obs_shapes:
            # get world coordinates of vertices
            vertices = [poly.body.local_to_world(v) for v in poly.get_vertices()]
            vertices_np = np.array([[v.x, v.y] for v in vertices])

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
        selem = disk(np.floor(robot_pixel_width / 4))
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
            if(self.renderer):
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
            if(self.renderer):
                self.renderer.render(save=False)


    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        pass
