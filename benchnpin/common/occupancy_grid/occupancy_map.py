import numpy as np
import cv2
import math
from skimage.draw import draw
from skimage.measure import block_reduce
from benchnpin.common.geometry.polygon import poly_centroid

class OccupancyGrid:

    def __init__(self, grid_width, grid_height, map_width, map_height, local_width=6, local_height=6, ship_body=None, meter_to_pixel_scale=50) -> None:
        """
        grid_width, grid_height, map_width, map_height are in meter units
        ship body info at starting position
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.map_width = map_width
        self.map_height = map_height
        self.occ_map_width = int(self.map_width / self.grid_width)         # number of grids in x-axis
        self.occ_map_height = int(self.map_height / self.grid_height)      # number of grids in y-axis
        self.meter_to_pixel_scale = meter_to_pixel_scale

        self.local_width = local_width
        self.local_height = local_height
        self.local_window_height = int(self.local_height / self.grid_height)            # local window height (unit: cell)
        self.local_window_width = int(self.local_width / self.grid_width)            # local window width (unit: cell)

        # print("Occupancy map resolution: ", grid_width, "; occupancy map dimension: ", (self.occ_map_width, self.occ_map_height))

        self.occ_map = np.zeros((self.occ_map_height, self.occ_map_width))
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        self.obstacle_centroids = np.zeros((self.occ_map_height, self.occ_map_width))
        self.swath = np.zeros((self.occ_map_height, self.occ_map_width))


    def compute_occ_img(self, obstacles, ice_binary_w=235, ice_binary_h=774, local_range=None, ship_state=None):
        meter_to_pixel_scale = self.meter_to_pixel_scale

        raw_ice_binary = np.zeros((ice_binary_h, ice_binary_w))

        for obstacle in obstacles:

            if local_range is not None:
                robot_x, robot_y = ship_state[:2]
                range_x, range_y = local_range
                center_x, center_y = np.abs(poly_centroid(obstacle))
                if abs(robot_x - center_x) > range_x or abs(robot_y - center_y) > range_y:
                    continue

            obstacle = np.asarray(obstacle) * meter_to_pixel_scale

            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(obstacle[:, 1], obstacle[:, 0], shape=raw_ice_binary.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            raw_ice_binary[rr, cc] = 1.0
        
        # occ_val = np.sum(raw_ice_binary) / (raw_ice_binary.shape[0] * raw_ice_binary.shape[1])
        # print("occ concentration: ", occ_val)
        
        return raw_ice_binary
    
    def compute_occ_img_walls(self, walls, width , height, wall_radius=0.5):
        meter_to_pixel_scale = height / self.map_height
        raw_wall_binary = np.zeros((height, width))
        for wall in walls:
            vertices = []
            
            direction_vector = np.array(wall[1]) - np.array(wall[0])
            line_length = np.linalg.norm(direction_vector)
            unit_direction_vector = direction_vector / line_length
            perpendicular_unit_vector = np.array([-unit_direction_vector[1], unit_direction_vector[0]])
            
            #Create the four vertices of the wall 
            vertices.append(wall[0] + wall_radius * perpendicular_unit_vector - wall_radius * unit_direction_vector)
            vertices.append(wall[0] - wall_radius * perpendicular_unit_vector - wall_radius * unit_direction_vector)
            vertices.append(wall[1] - wall_radius * perpendicular_unit_vector + wall_radius * unit_direction_vector)
            vertices.append(wall[1] + wall_radius * perpendicular_unit_vector + wall_radius * unit_direction_vector)
            vertices = np.array(vertices) * meter_to_pixel_scale

            print("vertices: ", vertices)
            print("shape: ", vertices.shape)
            # get pixel coordinates on costmap that are contained inside obstacle/polygon
            rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0], shape=raw_wall_binary.shape)

            # skip if 0 area
            if len(rr) == 0 or len(cc) == 0:
                continue

            raw_wall_binary[rr, cc] = 1.0

        return raw_wall_binary

    # NOTE Old version, keep here for a reference temporarily
    # def compute_con_gridmap(self, raw_ice_binary, local_range=None, ship_state=None, save_fig_dir=None):
    #     """
    #     Compute concentration grid map
    #     """
    #     meter_to_pixel_scale_y = self.meter_to_pixel_scale
    #     meter_to_pixel_scale_x = self.meter_to_pixel_scale

    #     if local_range is not None:
    #         robot_x, robot_y = ship_state[:2]
    #         range_x, range_y = local_range

    #     for i in range(self.occ_map_height):

    #         if local_range is not None:
    #             grid_y = i * self.grid_height
    #             if abs(grid_y - robot_y) > range_y:
    #                 continue

    #         y_low = int(i * self.grid_height * meter_to_pixel_scale_y)
    #         y_high = int((i + 1) * self.grid_height * meter_to_pixel_scale_y)
    #         if y_high >= raw_ice_binary.shape[0]:
    #             y_high = raw_ice_binary.shape[0] - 1

    #         for j in range(self.occ_map_width):

    #             if local_range is not None:
    #                 grid_x = j * self.grid_width
    #                 if abs(grid_x - robot_x) > range_x:
    #                     continue

    #             x_low = int(j * self.grid_width * meter_to_pixel_scale_x)
    #             x_high = int((j + 1) * self.grid_width * meter_to_pixel_scale_x)
    #             if x_high >= raw_ice_binary.shape[1]:
    #                 x_high = raw_ice_binary.shape[1] - 1
    #             cropped_region = raw_ice_binary[y_low:y_high, x_low:x_high]

    #             self.occ_map[i, j] = np.mean(cropped_region)
    #     return self.occ_map



    def compute_con_gridmap(self, raw_ice_binary, save_fig_dir=None):
        """
        Compute concentration grid map
        """
        meter_to_pixel_scale_y = self.meter_to_pixel_scale
        meter_to_pixel_scale_x = self.meter_to_pixel_scale

        
        block_size = (int(self.grid_height * meter_to_pixel_scale_y), int(self.grid_width * meter_to_pixel_scale_x))
        occ_map = block_reduce(raw_ice_binary, block_size, np.mean)

        self.occ_map = occ_map
        return occ_map


    def eagle_view_obstacle_map(self, raw_ice_binary, ship_state, vertical_shift):

        global_obstacle_map = self.compute_con_gridmap(raw_ice_binary=raw_ice_binary)

        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        local_obstacle_map = np.zeros((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_obstacle_map.shape[0] or global_j < 0 or global_j >= global_obstacle_map.shape[1]:
                    continue

                local_obstacle_map[local_i, local_j] = global_obstacle_map[global_i, global_j]
        
        self.local_obstacle_map = local_obstacle_map  
        return local_obstacle_map


    def compute_ship_footprint(self, body, ship_vertices, padding=0.25):
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(body.position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)

        rr, cc = draw.polygon(r=r, c=c)
        self.footprint[rr, cc] = 1.0


    def compute_goal_image(self, goal_y):
        self.goal_img = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_y = self.occ_map_height / self.map_height
        goal_y_idx = int(goal_y * meter_to_grid_scale_y)
        self.goal_img[goal_y_idx] = 1.0

    def compute_goal_point_image(self, goal):
        goal_x, goal_y = goal
        self.goal_img = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height
        goal_x_idx = int(goal_x * meter_to_grid_scale_x)
        goal_y_idx = int(goal_y * meter_to_grid_scale_y)
        self.goal_img[goal_y_idx, goal_x_idx] = 1.0
    
    def compute_ship_footprint_planner(self, ship_state, ship_vertices, padding=0.25):
        """
        NOTE this function computes current ship footprint similarily to self.compute_ship_footprint()
        but is intended for generating observations for planners 
        :param ship_state: (x, y, theta) where x, y are in meter and theta in radian
        :param ship_vertices: original unscaled, unpadded ship vertices
        """
        self.footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # # apply padding as in a* search
        # ship_vertices = np.asarray(
        #     [[np.sign(a) * (abs(a) + padding), np.sign(b) * (abs(b) + padding)] for a, b in ship_vertices]
        # )

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            # print("ship outside the costmap!!!")
            return
        
        rr, cc = draw.polygon(r=r, c=c)
        
        self.footprint[rr, cc] = 1.0



    def _compute_global_footprint(self, ship_state, ship_vertices, padding=0.25):
        """
        This function computes a global footprint map. 
        Values correspondences: free space 0.5, robot 1.0, out-of-bound 0.0
        """
        global_footprint = np.zeros((self.occ_map_height, self.occ_map_width))
        global_footprint = global_footprint + 0.5                   # free space 0.5, robot 1.0, out-of-bound 0.0
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vertices = np.asarray(ship_vertices) @ R.T + np.asarray(position)

        r = []
        c = []
        for x, y in vertices:
            grid_x = x * meter_to_grid_scale_x
            grid_y = y * meter_to_grid_scale_y
            if grid_y < 0 or grid_y >= self.occ_map_height or grid_x < 0 or grid_x >= self.occ_map_width:
                continue
            r.append(grid_y)
            c.append(grid_x)
        
        # it is possible that the ship state is outside of the grid map
        if len(r) == 0 or len(c) == 0:
            return None
        
        rr, cc = draw.polygon(r=r, c=c)
        
        global_footprint[rr, cc] = 1.0
        return global_footprint


    def eagle_view_footprint(self, ship_state, ship_vertices, vertical_shift=2): 
        """
        This function computes a eagle-centric footprint crop, based on the global footprint from _compute_global_footprint()
        Values correspondences: free space 0.5, robot 1.0, out-of-bound 0.0
        """
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        local_footprint = np.zeros((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        global_footprint = self._compute_global_footprint(ship_state, ship_vertices)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_footprint.shape[0] or global_j < 0 or global_j >= global_footprint.shape[1]:
                    continue

                local_footprint[local_i, local_j] = global_footprint[global_i, global_j]
        
        self.local_footprint = local_footprint  
        return local_footprint

    
    def global_goal_dist_transform(self, goal_y):
        """
        Compute a global normalized goal-line distance transform image
        """
        global_edt = np.zeros((self.occ_map_height, self.occ_map_width))
        grid_to_meter_scale_y = self.map_height / self.occ_map_height

        for i in range(self.occ_map_height):
            for j in range(self.occ_map_width):
                
                # compute distance to goal in meters
                dist_to_goal = goal_y - i * grid_to_meter_scale_y
                if dist_to_goal < 0: 
                    dist_to_goal = 0

                # normalize to prevent covariance shift
                nomalized_dist_to_goal = dist_to_goal / goal_y

                global_edt[i, j] = nomalized_dist_to_goal

        return global_edt


    
    def eagle_view_goal_dist_transform(self, goal_y, ship_state, vertical_shift=2):
        """
        Compute an eagle-centric local crop of a global goal-line distance transform from global_goal_dist_transform()
        """
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        global_edt = self.global_goal_dist_transform(goal_y=goal_y)
        local_edt = np.ones((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_edt.shape[0] or global_j < 0 or global_j >= global_edt.shape[1]:
                    continue

                local_edt[local_i, local_j] = global_edt[global_i, global_j]
        
        self.local_edt = local_edt  
        return local_edt


    def global_orientation_map(self, ship_state, head, tail):
        """
        Compute a global orientation map. This is a grayscale map with a single line indicating the orientation of the ship
        """
        global_orientation = np.zeros((self.occ_map_height, self.occ_map_width))
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        position = ship_state[:2]
        angle = ship_state[2]

        # ship vertices in meter
        heading = angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])

        # get global position for ship head and tail
        head_pos = np.array(head) @ R.T + np.array(position)
        tail_pos = np.array(tail) @ R.T + np.array(position)

        # convert to grid coordinate
        head_pix = np.array([head_pos[0] * meter_to_grid_scale_x, head_pos[1] * meter_to_grid_scale_y]).astype(np.uint8)       # (x, y)
        tail_pix = np.array([tail_pos[0] * meter_to_grid_scale_x, tail_pos[1] * meter_to_grid_scale_y]).astype(np.uint8)       # (x, y)

        cv2.line(global_orientation, head_pix, tail_pix, color=0.5, thickness=1)
        global_orientation[head_pix[1], head_pix[0]] = 1.0              # mark the head

        return global_orientation


    def eagle_view_orientation_map(self, ship_state, head, tail, vertical_shift):
        """
        Compute an eagle-centric local crop of the global orientation map, computed from global_orientation_map()
        """
        meter_to_grid_scale_x = self.occ_map_width / self.map_width
        meter_to_grid_scale_y = self.occ_map_height / self.map_height

        local_orientation = np.zeros((self.local_window_height, self.local_window_width))

        robot_x, robot_y = ship_state[:2]
        window_x, window_y = robot_x, robot_y + vertical_shift      # shifting the local window upward

        window_x = int(window_x * meter_to_grid_scale_x)                              # center of local window on global window (unit: cell)
        window_y = int(window_y * meter_to_grid_scale_y)

        global_orientation = self.global_orientation_map(ship_state, head, tail)

        for local_i in range(self.local_window_height):
            for local_j in range(self.local_window_width):

                global_i = int(local_i + window_y - (self.local_window_height / 2))
                global_j = int(local_j + window_x - (self.local_window_width / 2))

                # check out-of-bound
                if global_i < 0 or global_i >= global_orientation.shape[0] or global_j < 0 or global_j >= global_orientation.shape[1]:
                    continue

                local_orientation[local_i, local_j] = global_orientation[global_i, global_j]
        
        self.local_orientation = local_orientation  
        return local_orientation