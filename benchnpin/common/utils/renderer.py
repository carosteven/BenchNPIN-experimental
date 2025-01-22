import pygame
import pymunk 
import pymunk.pygame_util
import math
import numpy as np
import os

class Renderer():

    def __init__(self, space, env_width, env_height, render_scale=35, background_color=(255, 255, 255), caption="Renderer", **kwargs):
        """
        :param space: Pymunk space to be rendered
        :param env_width: the width of the environment in Pymunk unit (in our case, meter)
        :param env_height: the height of the envionment in Pymunk unit (in our case, meter)
        :param render_scale: the scale for transformation from pymunk unit to pygame pixel
        """

        # get parameters
        self.goal_line = kwargs.get('goal_line', None)
        self.goal_point = kwargs.get('goal_point', None)
        self.clearance_boundary = kwargs.get('clearance_boundary', None)

        # scale to convert from pymunk meter unit to pygame pixel unit
        self.render_scale = render_scale
        self.background_color = background_color

        # Initialize Pygame
        pygame.init()
        self.pygame_w, self.pygame_h = env_width * self.render_scale, env_height * self.render_scale
        self.window = pygame.display.set_mode((self.pygame_w, self.pygame_h))
        pygame.display.set_caption(caption)

        # set pygame y-axis same as pymunk y-axis
        pymunk.pygame_util.positive_y_is_up = True

        self.draw_options = pymunk.pygame_util.DrawOptions(self.window)

        # disable the draw collision point flag
        self.draw_options.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS

        # convert from pymunk meter unit to pygame pixel unit
        self.draw_options.transform = pymunk.Transform.scaling(self.render_scale)

        self.space = space

        self.path = None


    def to_pygame(self, pymunk_point):
        """
        Convert Pymunk world coorniates (meter unit) to Pygame screen coordinates (pixel unit)
        """
        x = pymunk_point[0]
        y = pymunk_point[1]
        return int(x * self.render_scale), int(self.pygame_h - y * self.render_scale)

    
    def update_path(self, path):
        """
        Update the planned path for display for planning-based policies
        """
        self.path = path

    
    def reset(self, new_space):
        """
        Reset the renderer to display a new Pymunk space
        """
        self.space = new_space

    
    def display_planned_path(self):
        """
        Display the planned path given from a planner
        """
        pygame.draw.lines(
                self.window, (0, 255, 0), False,  # green color, not a closed shape
                [self.to_pygame(point) for point in self.path],  # Convert trajectory to Pygame coordinates
                3,  # Line thickness
            )

    
    def display_goal_line(self):
        """
        Display the goal line for ship ice navigation
        """
        pygame.draw.line(
            self.window,
            (255, 255, 255),  # Line color (white)
            (0, self.goal_line),                # Start point (x1, y1)
            (self.pygame_w, self.goal_line),    # End point (x2, y2)
            6               # Line width
        )

    

    def display_goal_region(self):
        """
        Display goal regions for navigation tasks
        """
        raise NotImplementedError

    def display_goal_point(self):
        """
        Display goal point for navigation tasks
        """
        pygame.draw.circle(
            self.window,
            (255, 255, 255),  # Circle color (white)
            self.to_pygame(self.goal_point),  # Circle center
            5,  # Circle radius
            0   # Circle thickness
        )
    
    def display_clearance_boundary(self):
        """
        Display clearance boundary for object pushing tasks
        """
        pygame.draw.polygon(
            self.window,
            (0, 255, 0),  # Line color (green)
            [self.to_pygame(point) for point in self.clearance_boundary],  # Convert boundary to Pygame coordinates
            2  # Line width
        )

    def render(self, save=False, path=None):
        self.window.fill(self.background_color)
        self.space.debug_draw(self.draw_options)

        if self.path is not None:
            self.display_planned_path()
        
        if self.goal_line is not None:
            self.display_goal_line()
        
        if self.clearance_boundary is not None:
            self.display_clearance_boundary()

        ### could add goal region display here
        ###

        ### Goal point display
        if self.goal_point is not None:
            self.display_goal_point()

        pygame.display.update()

        if save:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

            pygame.image.save(self.window, path)


    
    def close(self):
        pygame.quit()
