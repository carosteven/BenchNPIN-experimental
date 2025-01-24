import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

import pickle
import random
import pymunk
from pymunk import Vec2d
from matplotlib import pyplot as plt

# maze NAMO specific imports
from benchnpin.common.cost_map import CostMap
from benchnpin.common.evaluation.metrics import total_work_done
from benchnpin.common.geometry.polygon import poly_area
from benchnpin.common.robot import Robot
from benchnpin.common.utils.renderer import Renderer
from benchnpin.common.utils.sim_utils import generate_sim_obs, generate_sim_maze
from benchnpin.common.geometry.polygon import poly_centroid
from benchnpin.common.utils.utils import DotDict
from benchnpin.common.occupancy_grid.occupancy_map import OccupancyGrid


R = lambda theta: np.asarray([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

YAW_CONSTRAINT_PENALTY = 0
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
BACKWARD = 8

class MazeNAMO(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        super(MazeNAMO, self).__init__()

        # get current directory of this script
        self.current_dir = os.path.dirname(__file__)

        # construct absolute path to the env_config folder
        cfg_file = os.path.join(self.current_dir, 'config.yaml')

        cfg = DotDict.load_from_file(cfg_file)
        self.occupancy = OccupancyGrid(grid_width=cfg.occ.grid_size, grid_height=cfg.occ.grid_size, map_width=cfg.occ.map_width, map_height=cfg.occ.map_height, ship_body=None)
        self.cfg = cfg

        self.beta = 500         # amount to scale the collision reward

        self.episode_idx = None     # the increment of this index is handled in reset()

        self.path = None
        self.scatter = False

        self.low_dim_state = self.cfg.low_dim_state

        
        self.env_max_trial = 4000

        # Define action space
        max_linear_speed = 1.0
        max_yaw_rate_step = (np.pi/2) / 15        # rad/sec
        print("max yaw rate per step: ", max_yaw_rate_step)
        self.action_space = spaces.Box(low= np.array([0, -max_yaw_rate_step]), 
                                       high=np.array([max_linear_speed, max_yaw_rate_step]),
                                       dtype=np.float64)
        
        # Define observation space
        self.low_dim_state = self.cfg.low_dim_state
        if self.low_dim_state:
            #low dimensional observation space comprises of the 2D positions of each obstacle in addition to the robot
            self.fixed_trial_idx = self.cfg.fixed_trial_idx
            if self.cfg.randomize_obstacles:
                self.observation_space = spaces.Box(low=-10, high=30, shape=((self.cfg.num_obstacles+1) * 2,), dtype=np.float64) 
            else:
                self.observation_space = spaces.Box(low=-10, high=30, shape=(8,), dtype=np.float64) # 8 for 3 obstacles and the robot
        
        else:
            #high dimensional observation space comprises of the occupancy grid map with 4 channels
            #channel 1 - occupancy grid map with static obstacles
            #channel 2 - occupancy grid map with movable obstacles
            #channel 3 - occupancy grid map with robot footprint
            #channel 4 - occupancy grid map with goal location
            self.observation_shape = (4, self.occupancy.occ_map_height, self.occupancy.occ_map_width)
            self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float64)

        self.yaw_lim = (0, np.pi)       # lower and upper limit of ship yaw  
        self.boundary_violation_limit = 0.0       # if the ship is out of boundary more than this limit, terminate and truncate the episode 

        self.renderer = None
        self.con_fig, self.con_ax = plt.subplots(figsize=(10, 10))


        if self.cfg.demo_mode:
            self.angular_speed = 0.0
            self.angular_speed_increment = 0.005
            self.linear_speed = 0.0
            self.linear_speed_increment = 0.02

        #robot and obstacles occupancy grid
        self.occupancy_plot = plt
        self.occupancy_plot.ion()
        
   
    def init_maze_NAMO_sim(self):

        # initialize maze environment
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
            obstacle_body = arbiter.shapes[1].body
            obstacle_body.pre_collision_KE = obstacle_body.kinetic_energy  # hacky, adding a field to pymunk body object
            return True

        def post_solve_handler(arbiter, space, data):
            # nonlocal self.total_ke, self.system_ke_loss, self.total_impulse, self.clln_obs
            robot_shape, obstacle_shape = arbiter.shapes

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

        if self.renderer is None:
            self.renderer = Renderer(self.space, env_width=self.cfg.occ.map_width, env_height=self.cfg.occ.map_height, render_scale=40, 
                    background_color=(28, 107, 160), caption="ASV Navigation", goal_point= (self.cfg.goal_x, self.cfg.goal_y))
        else:
            self.renderer.reset(new_space=self.space)
        
    def init_maze_NAMO_env(self):

        #initialize the maze walls in a list (temporary)
        self.construct_maze_walls()
        #add maze walls to pymunk
        generate_sim_maze(self.space, self.maze_walls)
        
        # generate random start point, if specified and avoid maze walls
        if self.cfg.random_start:
            #x_start = 1 + random.random() * (self.cfg.start_x_range - 1)    # [1, start_x_range]
            while True:
                x_start = 1 + random.random() * (self.cfg.start_x_range - 1)
                y_start = 1 + random.random() * (self.cfg.start_y_range - 1)
                #check if the start and goal points are not in the maze walls
                min_dist = self.cfg.robot.min_obstacle_dist
                if not self.space.point_query((x_start, y_start), min_dist, pymunk.ShapeFilter()): 
                    print("start point: ", x_start, y_start)
                    break

            self.start = (x_start, y_start,np.pi*3/2)
        else:
            self.start = (2, 2,np.pi*3/2)

        # if self.cfg.randomize_obstacles:
        #     self.randomize_obstacles()

        self.obs_dicts = self.generate_obstacles()
        
        # filter out obstacles that have zero area
        self.obs_dicts[:] = [ob for ob in self.obs_dicts if poly_area(ob['vertices']) != 0]
        self.obstacles = [ob['vertices'] for ob in self.obs_dicts]

        self.goal = (self.cfg.goal_x, self.cfg.goal_y)
        
        # initialize ship sim objects
        self.polygons = generate_sim_obs(self.space, self.obs_dicts, self.cfg.sim.obstacle_density, color=(173, 216, 230, 255))
        for p in self.polygons:
            p.collision_type = 2

        self.robot_body, self.robot_shape = Robot.sim(self.cfg.robot.vertices, self.start, body_type=pymunk.Body.DYNAMIC, color=(64, 64, 64, 255))
        self.robot_shape.collision_type = 1
        self.space.add(self.robot_body, self.robot_shape)
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

                # loop through previous obstacles to check for overlap with other obstacles or maze walls
                overlapped = False
                for prev_obs_x, pre_obs_y in obstacles:
                    if ((center_x - prev_obs_x)**2 + (center_y - pre_obs_y)**2)**(0.5) <= obs_min_dist:
                        overlapped = True
                        break
                    if self.space.point_query((center_x, center_y), obs_min_dist, pymunk.ShapeFilter()):
                        overlapped = True
                        break
                
                if not overlapped:
                    obstacles.append([center_x, center_y])
                    obs_count += 1
        
        else:
            obstacles.apself.startpend([8, 5.5])
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

        self.init_maze_NAMO_sim()
        self.init_maze_NAMO_env()

        self.t = 0

        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        info = {'state': (round(self.robot_body.position.x, 2),
                                round(self.robot_body.position.y, 2),
                                round(self.robot_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'obs': updated_obstacles, 
                'box_count': 0}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)

        else:
            observation = self.generate_observation()
        return observation, info

          
    def construct_maze_walls(self):
        self.length = self.cfg.env.length
        self.width = self.cfg.env.width
        self.maze_version = self.cfg.env.maze_version
        if self.maze_version == 1:
            self.maze_walls = [[(0,0),(self.width,0)] , [(0,0),(0,self.length)],
                    [(self.width,0),(self.width,self.length)], 
                    [(0,self.length),(self.width,self.length)],
                     [(2*self.width/2,self.length),(2*self.width/2,5)],
                    [(self.width/2,0),(self.width/2,self.length - self.length/3)]]
        elif self.maze_version == 2:
            self.maze_walls = [[(0,0),(self.width,0)] , [(0,0),(0,self.length)],
                    [(self.width,0),(self.width,self.length)], 
                    [(0,self.length),(self.width,self.length)],
                    [(self.width/3,0),(self.width/3,2*self.length/3)], [(2*self.width/3,self.length),(2*self.width/3, self.length/3)]]
        else:
            #abort the program
            print("Invalid maze version")
            exit(1)

    def randomize_obstacles(self):
        """
        NOTE this function is called only when using low-dimensional observation
        """
        for obs in self.obs_dicts:
            prev_centre = np.array(obs['centre'])

            rand_x = 0.5 + random.random() * (self.cfg.max_obs_x - 0.5)
            rand_y = 1 + random.random() * (self.cfg.max_obs_y - 1)
            new_centre = np.array([rand_x, rand_y])

            # translate vertices and reset center
            obs['vertices'] = obs['vertices'] - prev_centre + new_centre
            obs['centre'] = new_centre
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        if self.cfg.demo_mode:

            if action == FORWARD:
                self.linear_speed = 1
            elif action == STOP_TURNING:
                self.angular_speed = 0
            elif action == BACKWARD:
                self.linear_speed = -1
            elif action == LEFT:
                self.angular_speed = 0.5
            elif action == RIGHT:
                self.angular_speed = -0.5

            elif action == SMALL_LEFT:
                self.angular_speed = 0.05
            elif action == SMALL_RIGHT:
                self.angular_speed = -0.05

            elif action == STOP:
                self.linear_speed = 0.0
                self.angular_speed = 0.0

            # check speed boundary
           # if self.linear_speed <= 0:
            #    self.linear_speed = 0
            if self.linear_speed >= self.target_speed:
                self.linear_speed = self.target_speed

            # apply linear and angular speeds
            global_velocity = R(self.robot_body.angle) @ [self.linear_speed, 0]

            # apply velocity controller
            self.robot_body.angular_velocity = self.angular_speed
            self.robot_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        else:

            # apply linear velocity
            global_velocity = R(self.robot_body.angle) @ [action[0], 0]

            # apply velocity controller
            self.robot_body.angular_velocity = action[1]
            self.robot_body.velocity = Vec2d(global_velocity[0], global_velocity[1])

        # move simulation forward
        boundary_constraint_violated = False
        boundary_violation_terminal = False      # if out of boundary for too much, terminate and truncate the episode
        for _ in range(self.steps):
            self.space.step(self.dt / self.steps)

            # apply boundary constraints
            if self.robot_body.position.x < 0 or self.robot_body.position.x > self.cfg.occ.map_width:
                boundary_constraint_violated = True
        if self.robot_body.position.x < 0 and abs(self.robot_body.position.x - 0) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
        if self.robot_body.position.x > self.cfg.occ.map_width and abs(self.robot_body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
            boundary_violation_terminal = True
        
        # get updated obstacles
        updated_obstacles = CostMap.get_obs_from_poly(self.polygons)
        

        # compute work done
        work = total_work_done(self.prev_obs, updated_obstacles)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_obs = updated_obstacles
        self.obstacles = updated_obstacles

        # check episode terminal condition
        if self.goal_is_reached():
            terminated = True
        else:
            terminated = False

        # compute reward
        if self.robot_body.position.y < self.goal[1]:
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
        info = {'state': (round(self.robot_body.position.x, 2),
                                round(self.robot_body.position.y, 2),
                                round(self.robot_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'collision reward': collision_reward, 
                'scaled collision reward': collision_reward * self.beta, 
                'dist reward': dist_reward, 
                'obs': updated_obstacles, 
               }    
        
        # generate observation
        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_obstacles=updated_obstacles)
        else:
            observation = self.generate_observation()
        
        return observation, reward, terminated, False, info


    def generate_observation_low_dim(self, updated_obstacles):
        """
        The observation is a vector of shape (num_obstacles * 2)+ 2 specifying the 2d position of the obstacles and the robot
        <robot_x, robot_y, obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        # print("num obs: ", len(updated_obstacles))
        observation = np.zeros(((len(updated_obstacles)+1) * 2))
        #robot position
        observation[0] = self.robot_body.position.x
        observation[1] = self.robot_body.position.y
        #obstacle positions
        for i in range(1,len(updated_obstacles)):
            obs = updated_obstacles[i]
            center = np.abs(poly_centroid(obs))
            observation[i * 2] = center[0]
            observation[i * 2 + 1] = center[1]
        return observation


    def update_path(self, new_path, scatter=False):
        if scatter:
            self.scatter = True
        self.path = new_path
        self.renderer.update_path(path=self.path)
    

    def generate_observation(self):
        #Compute Binary Occupancy Grids
        robot_pose = (self.robot_body.position.x, self.robot_body.position.y, self.robot_body.angle)
        self.occupancy.compute_ship_footprint_planner(ship_state=robot_pose, ship_vertices=self.cfg.robot.vertices)
        robot_occ = np.copy(self.occupancy.footprint)         # (H, W)
        movable_obstacles = self.occupancy.compute_occ_img(obstacles=self.obstacles, 
                        ice_binary_w=int(self.occupancy.map_width * self.cfg.occ.m_to_pix_scale), 
                        ice_binary_h=int(self.occupancy.map_height * self.cfg.occ.m_to_pix_scale))
        fixed_obstacles = self.occupancy.compute_occ_img(obstacles=self.maze_walls, 
                        ice_binary_w=int(self.occupancy.map_width * self.cfg.occ.m_to_pix_scale), 
                        ice_binary_h=int(self.occupancy.map_height * self.cfg.occ.m_to_pix_scale))
        self.occupancy.compute_goal_image(goal_y=self.goal[1])
        goal_img = np.copy(self.occupancy.goal_img)               # (H, W)
        
        occupancy = np.copy(self.occupancy.occ_map)         # (H, W)

        # compute footprint observation  NOTE: here we want unscaled, unpadded vertices
       # robot_pose = (self.robot_body.position.x, self.robot_body.position.y, self.robot_body.angle)
       # self.occupancy.compute_ship_footprint_planner(ship_state=robot_pose, ship_vertices=self.cfg.robot.vertices)
       # footprint = np.copy(self.occupancy.footprint)       # (H, W)

        # compute goal observation
        # self.occupancy.compute_goal_image(goal_y=self.goal[1])
        # goal_img = np.copy(self.occupancy.goal_img)               # (H, W)
        # observation = np.concatenate((np.array([occupancy]), np.array([footprint]), np.array([goal_img])))          # (3, H, W)
        print("Robot: ", robot_occ.shape)
        print("movable obstacles: ", movable_obstacles.shape)
        print("fixed obstacles: ", fixed_obstacles.shape)
        print("goal img: ", goal_img.shape)
        observation = np.concatenate((np.array([robot_occ]), np.array([movable_obstacles]), 
                                      np.array([fixed_obstacles]), np.array([goal_img])))          # (4, H, W)
        return observation

    def goal_is_reached(self):
        #check if the goal is within the robot's dimensions
        robot_x = self.robot_body.position.x
        robot_y = self.robot_body.position.y
        robot_angle = self.robot_body.angle
        robot_vertices = self.robot_shape.get_vertices()
        robot_transformed_vertices = [R(robot_angle) @ vertex + np.array([robot_x, robot_y]) for vertex in robot_vertices]
        #check if the goal is within the robot's dimensions (cross-product method)
        cross_product = []
        for i in range(len(robot_transformed_vertices)):
            vertex1 = robot_transformed_vertices[i]
            vertex2 = robot_transformed_vertices[(i+1)%len(robot_transformed_vertices)]
            cross_product.append(np.cross(vertex2 - vertex1, self.goal - vertex1))
        if all([cross >= 0 for cross in cross_product]) or all([cross <= 0 for cross in cross_product]):
            print("goal reached")
            return True
            

    
    def render(self, mode='human', close=False):
        """Renders the environment."""

        if self.t % self.cfg.anim.plot_steps == 0:

            path = os.path.join(self.cfg.output_dir, 't' + str(self.episode_idx), str(self.t) + '.png')
            self.renderer.render(save=True, path=path)

            # whether to also log occupancy observation
            if self.cfg.render.log_obs and not self.low_dim_state:
                print("Rendering occupancy observation")
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
