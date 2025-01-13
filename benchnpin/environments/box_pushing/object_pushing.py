# TODO rename from ship to robot (can't do it yet - 
#   TypeError: __init__() got an unexpected keyword argument 'robot_body' was raised from the environment creator for object-pushing-v0 with kwargs ({}))

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
from benchnpin.common.utils.sim_utils import generate_sim_cubes, generate_sim_bounds, generate_sim_corners
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
BACKWARD = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7

ROBOT_HALF_WIDTH = 0.03
CUBE_MASS = 0.01
WALL_THICKNESS = 1.4*10

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
            if self.cfg.randomize_cubes:
                self.observation_space = spaces.Box(low=-10, high=30, shape=(self.cfg.num_cubes * 2,), dtype=np.float64)
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
        self.normal_cancelled_velocity = Vec2d(0, 0)

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

        # keep track of cubes pushed into receptacle
        self.cumulative_cubes = 0

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

        def boundary_collision_begin(arbiter, space, data):
            return True
        
        def cube_boundary_pre_solve(arbiter, space, data):
            cube_shape = arbiter.shapes[0]
            self.prevent_cube_boundary_intersection(cube_shape)
            return True
        
        def recept_collision_begin(arbiter, space, data):
            return False

        # handler = space.add_default_collision_handler()
        self.handler = self.space.add_collision_handler(1, 2)
        # from pymunk docs
        # post_solve: two shapes are touching and collision response processed
        self.handler.pre_solve = pre_solve_handler
        self.handler.post_solve = post_solve_handler

        self.boundary_handler = self.space.add_collision_handler(1, 3)
        self.boundary_handler.begin = boundary_collision_begin
        
        self.cube_boundary_handler = self.space.add_collision_handler(2, 3)
        self.cube_boundary_handler.pre_solve = cube_boundary_pre_solve

        self.robot_recept_handler = self.space.add_collision_handler(1, 4)
        self.robot_recept_handler.begin = recept_collision_begin

        self.cube_recept_handler = self.space.add_collision_handler(2, 4)
        self.cube_recept_handler.begin = recept_collision_begin

    def is_point_inside_polygon(self, point, polygon_vertices):
        # Ray-casting algorithm to check if a point is inside a polygon
        x, y = point
        n = len(polygon_vertices)
        inside = False

        p1x, p1y = polygon_vertices[0][:2]  # Unpack only the first two elements
        for i in range(n + 1):
            p2x, p2y = polygon_vertices[i % n][:2]  # Unpack only the first two elements
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
    
    def prevent_cube_boundary_intersection(self, cube_shape):
        cube_body = cube_shape.body
        # Check if any vertex of the cube intersects with the boundary
        cube_vertices = [cube_body.local_to_world(v) for v in cube_shape.get_vertices()]
        new_position = pymunk.Vec2d(cube_body.position.x, cube_body.position.y)

        # Check if the cube intersects with any dividers or columns
        for obstacle in self.boundary_dicts:
            if obstacle['type'] == 'divider' or obstacle['type'] == 'column' or obstacle['type'] == 'wall':
                obstacle_vertices = obstacle['vertices']
                for vertex in cube_vertices:
                    if self.is_point_inside_polygon(vertex, obstacle_vertices):
                        # Adjust the position of the cube to prevent intersection with the obstacle
                        for vertex in cube_vertices:
                            if self.is_point_inside_polygon(vertex, obstacle_vertices):
                                # Move the cube away from the obstacle
                                obstacle_center = np.mean(obstacle_vertices, axis=0)
                                direction = pymunk.Vec2d(vertex[0], vertex[1]) - pymunk.Vec2d(obstacle_center[0], obstacle_center[1])
                                direction = direction.normalized()
                                new_position += direction * 0.1  # Move the cube slightly away from the obstacle
                        cube_body.position = new_position
                        break

    def init_ship_ice_env(self):

        # generate random start point, if specified
        if self.cfg.random_start:
            length = self.cfg.ship.length
            width = self.cfg.ship.width
            size = max(length, width)
            x_start = random.uniform(-self.cfg.env.room_length / 2 + size, self.cfg.env.room_length / 2 - size)
            y_start = random.uniform(-self.cfg.env.room_width / 2 + size, self.cfg.env.room_width / 2 - size)
            heading = random.uniform(0, 2 * np.pi)
            self.start = (x_start, y_start, heading)
        else:
            self.start = (5, 1.5, np.pi*3/2)

        self.boundary_dicts = self.generate_boundary()

        # if self.cfg.randomize_cubes:
        #     self.randomize_cubes()

        self.cubes_dicts = self.generate_cubes()
        
        # filter out cubes that have zero area NOTE probably not needed
        self.cubes_dicts[:] = [c for c in self.cubes_dicts if poly_area(c['vertices']) != 0]
        self.cubes = [c['vertices'] for c in self.cubes_dicts]

        self.goal = (0, self.cfg.goal_y)

        # initialize ship sim objects
        self.polygons = generate_sim_cubes(self.space, self.cubes_dicts, self.cfg.sim.cube_density)
        self.boundaries = generate_sim_bounds(self.space, self.boundary_dicts, density=1)
        for p in self.polygons:
            p.collision_type = 2
        for b in self.boundaries:
            b.collision_type = 3
            if b.label == 'receptacle':
                b.collision_type = 4

        # Get vertices of corners (after they have been moved to proper spots)
        corner_dicts = [obstacle for obstacle in self.boundary_dicts if obstacle['type'] == 'corner']
        corner_polys = [shape for shape in self.boundaries if getattr(shape, 'label', None) == 'corner']
        for dict in corner_dicts:
            dict['vertices'] = []
            for i in range(3):
                vs = corner_polys[0].get_vertices()
                transformed_vertices = [corner_polys[0].body.local_to_world(v) for v in vs]
                dict['vertices'].append(np.array([[v.x, v.y] for v in transformed_vertices]))
                corner_polys.pop(0)


        self.ship_body, self.ship_shape = Robot.sim(self.cfg.ship.vertices, self.start, body_type=pymunk.Body.DYNAMIC)
        self.ship_shape.collision_type = 1
        self.space.add(self.ship_body, self.ship_shape)
        # run initial simulation steps to let environment settle
        for _ in range(1000):
            self.space.step(self.dt / self.steps)
        self.prev_obs = CostMap.get_obs_from_poly(self.polygons)

    def get_receptacle_position_and_size(self):
        size = self.cfg.env.receptacle_width
        return [self.cfg.env.room_length / 2 - size / 2, self.cfg.env.room_width / 2 - size / 2, size]

    def generate_boundary(self):
        boundary_dicts = []
        # generate walls
        for x, y, length, width in [
            (-self.cfg.env.room_length / 2 - WALL_THICKNESS / 2, 0, WALL_THICKNESS, self.cfg.env.room_width),
            (self.cfg.env.room_length / 2 + WALL_THICKNESS / 2, 0, WALL_THICKNESS, self.cfg.env.room_width),
            (0, -self.cfg.env.room_width / 2 - WALL_THICKNESS / 2, self.cfg.env.room_length + 2 * WALL_THICKNESS, WALL_THICKNESS),
            (0, self.cfg.env.room_width / 2 + WALL_THICKNESS / 2, self.cfg.env.room_length + 2 * WALL_THICKNESS, WALL_THICKNESS),
            ]:

            boundary_dicts.append(
                {'type': 'wall',
                 'vertices': np.array([
                    [x - length / 2, y - width / 2],  # bottom-left
                    [x + length / 2, y - width / 2],  # bottom-right
                    [x + length / 2, y + width / 2],  # top-right
                    [x - length / 2, y + width / 2],  # top-left
                ])
            })
        
        # generate receptacle
        x, y, size = self.get_receptacle_position_and_size()
        boundary_dicts.append(
            {'type': 'receptacle',
             'position': (x, y),
             'vertices': np.array([
                [x - size / 2, y - size / 2],  # bottom-left
                [x + size / 2, y - size / 2],  # bottom-right
                [x + size / 2, y + size / 2],  # top-right
                [x - size / 2, y + size / 2],  # top-left
            ]),
            'length': size,
            'width': size
        })
        
        def add_random_columns(obstacles, max_num_columns):
            num_columns = random.randint(1, max_num_columns) # NOTE need to be able to manually set seed
            column_length = 1
            column_width = 1
            buffer_width = 0.8
            col_min_dist = 1.2
            cols_dict = []

            new_cols = []
            for _ in range(num_columns):
                for _ in range(100): # try 100 times to generate a column that doesn't overlap with existing columns
                    x = random.uniform(-self.cfg.env.room_length / 2 + 2 * buffer_width + column_length / 2,
                                        self.cfg.env.room_length / 2 - 2 * buffer_width - column_length / 2)
                    y = random.uniform(-self.cfg.env.room_width / 2 + 2 * buffer_width + column_width / 2,
                                        self.cfg.env.room_width / 2 - 2 * buffer_width - column_width / 2)
                    
                    overlapped = False
                    rx, ry, size = self.get_receptacle_position_and_size()
                    if ((x - rx)**2 + (y - ry)**2)**(0.5) <= col_min_dist / 2 + size / 2:
                        overlapped = True
                        break
                    for prev_col in new_cols:
                        if ((x - prev_col[0])**2 + (y - prev_col[1])**2)**(0.5) <= col_min_dist:
                            overlapped = True
                            break

                    if not overlapped:
                        new_cols.append([x, y])
                        break

            for x, y in new_cols:
                cols_dict.append({'type': 'column',
                                  'position': (x, y),
                                  'vertices': np.array([
                                      [x - column_length / 2, y - column_width / 2],  # bottom-left
                                      [x + column_length / 2, y - column_width / 2],  # bottom-right
                                      [x + column_length / 2, y + column_width / 2],  # top-right
                                      [x - column_length / 2, y + column_width / 2],  # top-left
                                      ]),
                                  'length': column_length,
                                  'width': column_width
                                })
            return cols_dict
        
        def add_random_horiz_divider():
            divider_length = 8
            divider_width = 0.5
            buffer_width = 3.5

            new_divider = []
            for _ in range(100): # try 100 times to generate a divider that doesn't overlap with existing obstacles
                if len(new_divider) == 1:
                    break

                x = self.cfg.env.room_length / 2 - divider_length / 2
                y = random.uniform(-self.cfg.env.room_width / 2 + buffer_width + divider_width / 2,
                                    self.cfg.env.room_width / 2 - buffer_width - divider_width / 2)

                new_divider.append([x, y])
            
            divider_dicts = []
            for x, y in new_divider:
                divider_dicts.append({'type': 'divider',
                                      'position': (x, y),
                                      'vertices': np.array([
                                          [x - divider_length / 2, y - divider_width / 2],  # bottom-left
                                          [x + divider_length / 2, y - divider_width / 2],  # bottom-right
                                          [x + divider_length / 2, y + divider_width / 2],  # top-right
                                          [x - divider_length / 2, y + divider_width / 2],  # top-left
                                          ]),
                                        'length': divider_length,
                                        'width': divider_width
                                    })
            return divider_dicts
                    
        
        # generate obstacles
        if self.cfg.env.obstacle_config == 'small_empty':
            pass
        elif self.cfg.env.obstacle_config == 'small_columns':
            boundary_dicts.extend(add_random_columns(boundary_dicts, 3))
        elif self.cfg.env.obstacle_config == 'large_columns':
            boundary_dicts.extend(add_random_columns(boundary_dicts, 8))
        elif self.cfg.env.obstacle_config == 'large_divider':
            boundary_dicts.extend(add_random_horiz_divider())
        else:
            raise ValueError(f'Invalid obstacle config: {self.cfg.env.obstacle_config}')
        
        # generate corners
        for i, (x, y) in enumerate([
            (-self.cfg.env.room_length / 2, self.cfg.env.room_width / 2),
            (self.cfg.env.room_length / 2, self.cfg.env.room_width / 2),
            (self.cfg.env.room_length / 2, -self.cfg.env.room_width / 2),
            (-self.cfg.env.room_length / 2, -self.cfg.env.room_width / 2),
            ]):
            if i == 1: # Skip the receptacle corner
                continue
            heading = -np.radians(i * 90)
            boundary_dicts.append(
                {'type': 'corner',
                 'position': (x, y),
                 'heading': heading,
                })
            
        # generate corners for divider
        for obstacle in boundary_dicts:
            if obstacle['type'] == 'divider':
                (x, y), length, width = obstacle['position'], obstacle['length'], obstacle['width']
                corner_positions = [(self.cfg.env.room_length / 2, y - width / 2), (self.cfg.env.room_length / 2, y + width / 2)]
                corner_headings = [-90, 180]
                for position, heading in zip(corner_positions, corner_headings):
                    heading = np.radians(heading)
                    boundary_dicts.append(
                        {'type': 'corner',
                        'position': position,
                        'heading': heading,
                        })

        return boundary_dicts

    def generate_cubes(self):
        cubes_size = self.cfg.cube_size / 2
        cubes = []          # a list storing non-overlapping cube centers

        if self.cfg.randomize_cubes:
            total_cubes_required = self.cfg.num_cubes
            self.num_box = self.cfg.num_cubes
            cube_min_dist = self.cfg.min_cube_dist
            min_x = self.cfg.min_cube_x
            max_x = self.cfg.max_cube_x
            min_y = self.cfg.min_cube_y
            max_y = self.cfg.max_cube_y

            cube_count = 0
            while cube_count < total_cubes_required:
                center_x = random.uniform(min_x, max_x)
                center_y = random.uniform(min_y, max_y)
                heading = random.uniform(0, 2 * np.pi)

                # loop through previous cubes to check for overlap
                overlapped = False
                for obstacle in self.boundary_dicts:
                    if obstacle['type'] == 'corner' or obstacle['type'] == 'wall':
                        continue
                    elif obstacle['type'] == 'divider':
                        # just check y distance
                        if abs(center_y - obstacle['position'][1]) <= (cube_min_dist / 2 + obstacle['width'] / 2) * 1.2:
                            overlapped = True
                            break
                    elif ((center_x - obstacle['position'][0])**2 + (center_y - obstacle['position'][1])**2)**(0.5) <= (cube_min_dist / 2 + obstacle['width'] / 2) * 1.2:
                        overlapped = True
                        break
                for prev_cube_x, pre_cube_y, _ in cubes:
                    if ((center_x - prev_cube_x)**2 + (center_y - pre_cube_y)**2)**(0.5) <= cube_min_dist:
                        overlapped = True
                        break
                
                if not overlapped:
                    cubes.append([center_x, center_y, heading])
                    cube_count += 1
        
        else:
            cubes.append([3, 2, 0])
            cubes.append([3, 1.5, 0])
            cubes.append([3, 1, 0])
            cubes.append([3, 3, 0])
            self.num_box = 4
        
        # convert to cubes dict
        cubes_dict = []
        for i, [cubes_x, cubes_y, cubes_heading] in enumerate(cubes):
            cubes_info = {}
            cubes_info['centre'] = np.array([cubes_x, cubes_y])
            cubes_info['vertices'] = np.array([[cubes_x + cubes_size, cubes_y + cubes_size], 
                                    [cubes_x - cubes_size, cubes_y + cubes_size], 
                                    [cubes_x - cubes_size, cubes_y - cubes_size], 
                                    [cubes_x + cubes_size, cubes_y - cubes_size]])
            cubes_info['heading'] = cubes_heading
            cubes_info['idx'] = i
            cubes_dict.append(cubes_info)
        return cubes_dict

    def cube_position_in_receptacle(self, cube_vertices):
        x, y, size = self.get_receptacle_position_and_size()
        receptacle_min_x = x - size / 2
        receptacle_max_x = x + size / 2
        receptacle_min_y = y - size / 2
        receptacle_max_y = y + size / 2
    
        for vertex in cube_vertices:
            if not (receptacle_min_x <= vertex[0] <= receptacle_max_x and receptacle_min_y <= vertex[1] <= receptacle_max_y):
                return False
        return True

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
                np.zeros((self.cfg.costmap.m, self.cfg.costmap.n)), self.cubes_dicts,
                ship_pos=self.start, ship_vertices=np.asarray(self.ship_shape.get_vertices()),
                map_figsize=None, y_axis_limit=self.cfg.plot.y_axis_limit, inf_stream=False, goal=self.goal[1], 
                path=np.zeros((3, 50)), boundaries=self.boundary_dicts
            )

        # get updated cubes
        updated_cubes = CostMap.get_obs_from_poly(self.polygons)
        info = {'state': (round(self.ship_body.position.x, 2),
                                round(self.ship_body.position.y, 2),
                                round(self.ship_body.angle, 2)), 
                'total_work': self.total_work[0], 
                'cubes': updated_cubes, 
                'box_count': 0}

        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_cubes=updated_cubes)

        else:
            observation = self.generate_observation()
        return observation, info
    

    def step(self, action):
        """Executes one time step in the environment and returns the result."""
        self.t += 1

        if self.cfg.demo_mode:

            if action == FORWARD:
                self.linear_speed = 0.01
            elif action == BACKWARD:
                self.linear_speed = -0.01
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
                # self.angular_speed = 0.0

            # check speed boundary
            # if self.linear_speed <= 0:
            #     self.linear_speed = 0
            if abs(self.linear_speed) >= self.target_speed:
                self.linear_speed = self.target_speed*np.sign(self.linear_speed)

            # apply linear and angular speeds
            global_velocity = R(self.ship_body.angle) @ [self.linear_speed, 0]

            # apply velocity controller
            self.ship_body.angular_velocity = self.angular_speed * 200
            self.ship_body.velocity = Vec2d(global_velocity[0], global_velocity[1]) * 200

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
        #     if self.ship_body.position.x < 0 or self.ship_body.position.x > self.cfg.occ.map_width:
        #         boundary_constraint_violated = True
        # if self.ship_body.position.x < 0 and abs(self.ship_body.position.x - 0) >= self.boundary_violation_limit:
        #     boundary_violation_terminal = True
        # if self.ship_body.position.x > self.cfg.occ.map_width and abs(self.ship_body.position.x - self.cfg.occ.map_width) >= self.boundary_violation_limit:
        #     boundary_violation_terminal = True
            
        # get updated cubes
        all_boxes_completed = self.boxes_completed()
        updated_cubes = CostMap.get_obs_from_poly(self.polygons)

        # compute work done
        work = total_work_done(self.prev_obs, updated_cubes)
        self.total_work[0] += work
        self.total_work[1].append(work)
        self.prev_obs = updated_cubes
        self.cubes = updated_cubes

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
                'cubes': updated_cubes, 
                'box_count': self.cumulative_cubes}
        
        # generate observation
        if self.low_dim_state:
            observation = self.generate_observation_low_dim(updated_cubes=updated_cubes)
        else:
            observation = self.generate_observation()
        
        return observation, reward, terminated, False, info


    def generate_observation_low_dim(self, updated_cubes):
        """
        The observation is a vector of shape (num_cubes * 2) specifying the 2d position of the cubes
        <obs1_x, obs1_y, obs2_x, obs2_y, ..., obsn_x, obsn_y>
        """
        observation = np.zeros((len(updated_cubes) * 2))
        for i in range(len(updated_cubes)):
            obs = updated_cubes[i]
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
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.cubes, ice_binary_w=235, ice_binary_h=774)

        elif self.occupancy.map_height == 20 and self.occupancy.map_width == 12:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.cubes, ice_binary_w=235, ice_binary_h=387)

        elif self.occupancy.map_height == 20 and self.occupancy.map_width == 6:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.cubes, ice_binary_w=118, ice_binary_h=387)

        elif self.occupancy.map_height == 10 and self.occupancy.map_width == 6:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.cubes, ice_binary_w=118, ice_binary_h=192)
        else:
            raw_ice_binary = self.occupancy.compute_occ_img(obstacles=self.cubes, ice_binary_w=235, ice_binary_h=1355)
        self.occupancy.compute_con_gridmap(raw_ice_binary=raw_ice_binary, save_fig_dir=None)
        occupancy = np.copy(self.occupancy.occ_map)         # (H, W)

        # compute footprint observation  NOTE: here we want unscaled, unpadded vertices
        ship_pose = (self.ship_body.position.x, self.ship_body.position.y, self.ship_body.angle)
        self.occupancy.compute_ship_footprint_planner(ship_state=ship_pose, ship_vertices=self.cfg.ship.vertices)
        footprint = np.copy(self.occupancy.footprint)       # (H, W)

        observation = np.concatenate((np.array([occupancy]), np.array([footprint])))          # (2, H, W)
        return observation

    
    def boxes_completed(self):
        """
        Returns a tuple: (int: number of boxes completed, bool: whether pushing task is complete)
        """
        completed = False

        for cube, vertices in zip(self.polygons, self.cubes):
            if self.cube_position_in_receptacle(vertices):
                self.cumulative_cubes += 1
                self.space.remove(cube, cube.body)
                self.polygons.remove(cube)
                self.plot.update_obstacles(obstacles=CostMap.get_obs_from_poly(self.polygons), obs_idx=cube.idx, update_patch=True)

        
        if self.cumulative_cubes == self.num_box:
            completed = True
        
        return completed



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
