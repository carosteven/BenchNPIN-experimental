from typing import List

import numpy as np
import pymunk

from benchnpin.common.ship import Ship

def create_agent(space, vertices: List, start_pos: Tuple[float, float, float], body_type: int):

        x, y, heading = start_pos
        # setup for pymunk
        body = pymunk.Body(body_type=body_type)  # mass and moment ignored when kinematic body type
        body.position = (x, y)
        body.angle = heading  # rotation of the body in radians
        dummy_shape = pymunk.Poly(None, vertices)
        centre_of_g = dummy_shape.center_of_gravity
        vs = [(x - centre_of_g[0], y - centre_of_g[1]) for x, y in vertices]

        shape = pymunk.Poly(body, vs, radius=0.08)
        shape.mass = 10.0
        shape.elasticity = 0.01
        shape.friction = 1.0
        shape.label = 'agent'
        space.add(body, shape)
        return shape

def generate_sim_agent(space, agent: dict, body_type=pymunk.Body.DYNAMIC):
    return create_agent(space, agent['vertices'], agent['start_pos'], body_type)

def create_static(space, boundary):
    body = space.static_body
    # body.position = (x, y)
    # dummy_shape = pymunk.Poly(None, vertices)
    # centre_of_g = dummy_shape.center_of_gravity
    # vs = [(x - centre_of_g[0], y - centre_of_g[1]) for x, y in vertices]

    vs = [(x, y) for x, y in boundary['vertices']]
    shape = pymunk.Poly(body, vs, radius=0.02)
    shape.elasticity = 0.01
    shape.friction = 1.0
    shape.label = boundary['type']
    space.add(shape)
    return shape

def generate_sim_bounds(space, bounds: List[dict]):
    shapes = [create_static(space, bound)
              for bound in bounds if bound['type'] != 'corner']
    shapes.extend(generate_sim_corners(space, [bound for bound in bounds if bound['type'] == 'corner']))
    return shapes

def create_corners(space, corner):
    shapes = []
    vs1 = [(0, 0),
           (0, -1),
           (0.5*np.sin(22.5*np.pi/180), -1 + 0.5*np.cos(22.5*np.pi/180)),
           ]
    vs2 = [(0, 0),
           (0.5*np.sin(22.5*np.pi/180), -1 + 0.5*np.cos(22.5*np.pi/180)),
           (1 - 0.5*np.cos(22.5*np.pi/180), -0.5*np.sin(22.5*np.pi/180)),
           ]
    vs3 = [(0, 0),
           (1, 0),
           (1 - 0.5*np.cos(22.5*np.pi/180), -0.5*np.sin(22.5*np.pi/180)),
           ]
    
    for vs in [vs1, vs2, vs3]:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = corner['position']
        body.angle = corner['heading']

        shape = pymunk.Poly(body, vs, radius=0.02)
        shape.elasticity = 0.01
        shape.friction = 1.0
        shape.label = 'corner'
        space.add(body, shape)
        shapes.append(shape)
    return shapes

def generate_sim_corners(space, corners: List[dict]):
    corner_shapes = []
    for corner in corners:
        cs = create_corners(space, corner)
        for shape in cs:
            corner_shapes.append(shape)
    return corner_shapes

def create_polygon(space, vertices, x, y, density, heading=0, label='poly', idx=None, radius=0.02, color=None):
    body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
    body.position = (x, y)
    dummy_shape = pymunk.Poly(None, vertices)
    centre_of_g = dummy_shape.center_of_gravity
    vs = [(x - centre_of_g[0], y - centre_of_g[1]) for x, y in vertices]

    shape = pymunk.Poly(body, vs, radius=0.02)
    shape.density = density
    shape.elasticity = 0.01
    shape.friction = 1.0
    space.add(body, shape)

    if color is not None:
        shape.color = color
    return shape


def generate_sim_obs(space, obstacles: List[dict], density, color=None):
    return [
        create_polygon(
            space, (obs['vertices'] - np.array(obs['centre'])).tolist(),
            *obs['centre'], density=density, color=color
        )
        for obs in obstacles
    ]

def generate_sim_maze(space, maze_walls):
    for pos in maze_walls:
        print(pos)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, pos[0], pos[1], 0.5)
        shape.elasticity = 0.5
        shape.friction = 0.5
        space.add(body, shape)

def simulate_ship_ice_collision(path, ship_vertices, obs_dicts, ship_vel=1, dt=0.25, steps=10):
    """
    Simulate collision between ship and ice as ship travels along path
    Assumes only 1 collision between ship and obstacle can occur
    """
    assert path.shape[1] == 3

    # make a new pymunk env with ship and obstacle
    space = pymunk.Space()
    static_body = space.static_body  # create a static body for friction constraints

    # init ship sim objects
    # ship dynamics will not change in response to collision if body type is KINEMATIC
    start_pose = path[0]
    ship_body, ship_shape = Ship.sim(ship_vertices, start_pose, body_type=pymunk.Body.KINEMATIC,
                                     velocity=(np.cos(start_pose[2]) * ship_vel, np.sin(start_pose[2]) * ship_vel))
    space.add(ship_body, ship_shape)
    ship_shape.collision_type = 1

    # init polygon objects
    for ob in obs_dicts:
        poly = create_polygon(
            space, (ob['vertices'] - np.array(ob['centre'])).tolist(),
            *ob['centre'], density=10
        )
        poly.collision_type = 2  # this will identify the obstacle in the collision shape pair object (i.e. arbiter)

    # flag to keep sim running until collision
    collision_ob = None
    initial_ob_pos = None

    # collision handler
    def collide(arbiter, space, data):
        nonlocal collision_ob, initial_ob_pos
        # print('collide!')
        if collision_ob is None:
            collision_ob = arbiter.shapes[1]  # keep a reference of obstacle so we can get velocity
            initial_ob_pos = collision_ob.body.position
            assert arbiter.shapes[1].collision_type == poly.collision_type
        return True

    handler = space.add_collision_handler(1, 2)
    # from pymunk docs
    # separate: two shapes have just stopped touching for the first time
    # begin: two shapes just started touching for the first time
    handler.begin = collide

    # assert abs(dt * ship_vel - (path_length(path[:, :2] / len(path)))) < 0.05 * (dt * ship_vel)
    for p in path:  #  FIXME: resolution of path, velocity, dt, and steps are all related
        # the amount that sim moves ship forward should be equal to resolution of path
        for _ in range(steps):  # this slows down function quite a bit but increases sim accuracy
            space.step(dt / steps)
        # update velocity and angle such that ship follows path exactly
        ship_body.velocity = (np.cos(p[2]) * ship_vel, np.sin(p[2]) * ship_vel)
        ship_body.angle = -p[2]

    if collision_ob is None:
        return None  # this means there were no collision
    return (
        tuple(collision_ob.body.velocity),
        tuple(collision_ob.body.position - initial_ob_pos),
        tuple(ship_body.velocity)
    )
