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
from benchnpin.common.ship import Ship as Robot
from benchnpin.common.utils.plot_pushing import Plot
from benchnpin.common.utils.sim_utils import generate_sim_obs
from benchnpin.common.geometry.polygon import poly_centroid
from benchnpin.common.utils.utils import DotDict
from benchnpin.common.occupancy_grid.occupancy_map import OccupancyGrid

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
OTHER = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7


class ObjectPushing(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        super(ObjectPushing, self).__init__()

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)

        # construct absolute path to the env_config folder
        cfg_file = os.path.join(self.current_dir, 'config.yaml')

        cfg = DotDict.load_from_file(cfg_file)
        self.occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)
        self.cfg = cfg

        self.env_max_trial = 4000

        self.beta = 500         # amount to scale the collision reward

        self.episode_idx = None

        self.path = None
        self.scatter = False

        # Define action space
        max_yaw_rate_step = (np.pi/2) / 15        # rad/sec
        print("max yaw rate per step: ", max_yaw_rate_step)
        self.action_space = spaces.Box(low=-max_yaw_rate_step, high=max_yaw_rate_step, dtype=np.float64)

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
        self.boundary_violation_limit = 0.0       # if the ship is out of boundary more than this limit, terminate and truncate the episode 

        self.plot = None
        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))


        if self.cfg.demo_mode:
            self.angular_speed = 0.0
            self.angular_speed_increment = 0.005
            self.linear_speed = 0.0
            self.linear_speed_increment = 0.02

        plt.ion()  # Interactive mode on
        

    def init_ship_ice_sim(self):

        # initialize ship-ice environment
        self.steps = self.cfg.sim.steps
        self.t_max = self.cfg.sim.t_max if self.cfg.sim.t_max else np.inf
        self.horizon = self.cfg.a_star.horizon
        self.replan = self.cfg.a_star.replan
        self.dt = self.cfg.controller.dt
        self.target_speed = self.cfg.controller.target_speed

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

        # setup a collision callback to keep track of total ke
        # def pre_solve_handler(arbiter, space, data):
        #     nonlocal ship_ke
        #     ship_ke = arbiter.shapes[0].body.kinetic_energy
        #     print('ship_ke', ship_ke, 'mass', arbiter.shapes[0].body.mass, 'velocity', arbiter.shapes[0].body.velocity)
        #     return True
        # # http://www.pymunk.org/en/latest/pymunk.html#pymunk.Body.each_arbiter

        # setup pymunk collision callbacks
        def pre_solve_handler(arbiter, space, data):
            ice_body = arbiter.shapes[1].body
            ice_body.pre_collision_KE = ice_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            # nonlocal self.total_ke, self.system_ke_loss, self.total_impulse, self.clln_obs
            ship_shape, ice_shape = arbiter.shapes

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
        
        
    def init_ship_ice_env(self):

        # generate random start point, if specified
        if self.cfg.random_start:
            x_start = 1 + random.random() * (self.cfg.start_x_range - 1)    # [1, start_x_range]
            self.start = (x_start, 1.0, np.pi / 2)
        else:
            self.start = (5, 1.0, np.pi / 2)

        # if self.cfg.randomize_obstacles:
        #     self.randomize_obstacles()

        self.obs_dicts = self.generate_obstacles()
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if poly_area(ob['vertices']) != 0]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        self.goal = (0, self.cfg.goal_y)

        # initialize ship sim objects
        self.polygons = generate_sim_obs(self.space, self.obs_dicts, self.cfg.sim.obstacle_density)
        for p in self.polygons:
            p.collision_type = 2

        self.ship_body, self.ship_shape = Robot.sim(self.cfg.ship.vertices, self.start)
        self.ship_shape.collision_type = 1
        self.space.add(self.ship_body, self.ship_shape)
        # run initial simulation steps to let environment settle
        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.polygons)

    
    def generate_obstacles(self):
        obs_size = self.cfg.obstacle_size
        obstacles = []          # a list storing non-overlappin obstacle centers

        if self.cfg.randomize_obstacles:
            total_obs_required = self.cfg.num_obstacles
            self.num_box = self.cfg.num_obstacles
            obs_min_dist = self.cfg.min_obs_dist
            min_x = self.cfg.min_obs_x
            max_x = self.cfg.max_obs_x
            min_y = self.cfg.min_obs_y
            max_y = self.cfg.max_obs_y

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
        
        else:
            obstacles.append([8, 5.5])
            obstacles.append([3, 7])
            obstacles.append([13, 8])
            self.num_box = 3
        
        # convert to obs dict
        obs_dict = []
        for obs_x, obs_y in obstacles:
            obs_info = {}
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

        self.init_ship_ice_sim()
        self.init_ship_ice_env()

        self.t = 0
        
        # close figure before opening new ones
        if self.plot is not None:
            self.plot.close()

        self.plot = Plot(
                np.zeros((self.cfg.costmap.m, self.cfg.costmap.n)), self.obs_dicts,
                ship_pos=self.start, ship_vertices=np.asarray(self.ship_shape.get_vertices()),
                map_figsize=None, y_axis_limit=self.cfg.plot.y_axis_limit, inf_stream=False, goal=self.goal[1], 
                path=np.zeros((3, 50))
            )

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        info = {'state': (round(self.ship_body.position.x, 2),
                                round(self.ship_body.position.y, 2),
                                round(self.ship_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'obs': updated_obstacles, 
                'box_count': 0}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)

        else:
            observation = self.generate_observation()
        return observation, info
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        if self.cfg.demo_mode:

            if action == FORWARD:
                self.linear_speed = 0.01
            elif action == STOP_TURNING:
                self.angular_speed = 0.0

            elif action == LEFT:
                self.angular_speed = 0.01
            elif action == RIGHT:
                self.angular_speed = -0.01

            elif action == SMALL_LEFT:
                self.angular_speed = 0.005
            elif action == SMALL_RIGHT:
                self.angular_speed = -0.005

            elif action == STOP:
                self.linear_speed = 0.0
                self.angular_speed = 0.0

            # check speed boundary
            if self.linear_speed <= 0:
                self.linear_speed = 0
            elif self.linear_speed >= self.target_speed:
                self.linear_speed = self.target_speed

            # apply linear and angular speeds
            global_velocity = R(self.ship_body.angle) @ [self.linear_speed, 0]

            # apply velocity controller
            self.ship_body.angular_velocity = self.angular_speed
            self.ship_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        else:

            # constant forward speed in global frame
            global_velocity = R(self.ship_body.angle) @ [self.target_speed, 0]

            # apply velocity controller
            self.ship_body.angular_velocity = action
            self.ship_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        # move simulation forward
        boundary_constraint_violated = False
        boundary_violation_terminal = False      # if out of boundary for too much, terminate and truncate the episode
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)

            # apply boundary constraints
            if self.ship_body.position.x < 0 or self.ship_body.position.x > self.cfg.occ.map_width:
                boundary_constraint_violated = True
        if self.ship_body.position.x < 0 and abs(self.ship_body.position.x - 0) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
        if self.ship_body.position.x > self.cfg.occ.map_width and abs(self.ship_body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
            
        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        num_completed, all_boxes_completed = self.boxes_completed(updated_obstacles=updated_obstacles)

        # compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        # check episode terminal condition
        if all_boxes_completed:
            terminated = True
        elif boundary_violation_terminal:
            terminated = True
        else:
            terminated = False

        # compute reward
        if self.ship_body.position.y < self.goal[1]:
            dist_reward = -1
        else:
            dist_reward = 0
        collision_reward = -work

        reward = self.beta * collision_reward + dist_reward

        # apply constraint penalty
        if boundary_constraint_violated:
            reward += BOUNDARY_PENALTY

        # apply terminal reward
        if terminated and not boundary_violation_terminal:
            reward += TERMINAL_REWARD

        # Optionally, we can add additional info
        info = {'state': (round(self.ship_body.position.x, 2),
                                round(self.ship_body.position.y, 2),
                                round(self.ship_body.angle, 2)), 
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


    def update_path(self, new_path, scatter=False):
        if scatter:
            self.scatter = True
        self.path = new_path
    

    def generate_observation(self):
        # compute occupancy map observation  (40, 12)
        if self.occupancy.map_height == 40:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, ice_binary_w=235, ice_binary_h=774)

        elif self.occupancy.map_height == 20 and self.occupancy.map_width == 12:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, ice_binary_w=235, ice_binary_h=387)

        elif self.occupancy.map_height == 20 and self.occupancy.map_width == 6:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, ice_binary_w=118, ice_binary_h=387)

        elif self.occupancy.map_height == 10 and self.occupancy.map_width == 6:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, ice_binary_w=118, ice_binary_h=192)
        else:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.obstacles, ice_binary_w=235, ice_binary_h=1355)
        self.occupancy.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=None)
        occupancy = np.copy(self.occupancy.occ_map)         # (H, W)

        # compute footprint observation  NOTE: here we want unscaled, unpadded vertices
        ship_pose = (self.ship_body.position.x, self.ship_body.position.y, self.ship_body.angle)
        self.occupancy.compute_ship_footprint_planner(ship_state=ship_pose, ship_vertices=self.cfg.ship.vertices)
        footprint = np.copy(self.occupancy.footprint)       # (H, W)

        observation = np.concatenate((np.array([occupancy]), np.array([footprint])))          # (2, H, W)
        return observation

    
    def boxes_completed(self, updated_obstacles):
        """
        Returns a tuple: (int: number of boxes completed, bool: whether pushing task is complete)
        """
        completed_count = 0
        completed = False

        for obs in updated_obstacles:
            center = np.abs(poly_centroid(obs))
            if center[1] - self.cfg.obstacle_size >= self.cfg.goal_y:
                completed_count += 1
        
        if completed_count == self.num_box:
            completed = True
        
        return completed_count, completed



    def render(self, mode='human', close=False):
        """Renders the environment."""

        # update animation
        if self.path is not None:

            if not self.scatter:
                self.plot.update_path(full_path=self.path.T)
            else:
                self.plot.update_path_scatter(full_path=self.path.T)

        self.plot.update_ship(self.ship_body, self.ship_shape, move_yaxis_threshold=self.cfg.anim.move_yaxis_threshold)
        self.plot.update_obstacles(obstacles=CostMap.get_obs_from_poly(self.polygons))
        # get updated obstacles
        self.plot.animate_sim(save_fig_dir=os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx))
                        if (self.cfg.anim.save and self.cfg.output_dir) else None, suffix=self.t)

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


    def close(self):
        """Optional: close any resources or cleanup if necessary."""
        pass
