from benchnpin.baselines.ship_ice_nav.planning_based.planners.lattice import LatticePlanner
from benchnpin.baselines.ship_ice_nav.planning_based.planners.predictive import PredictivePlanner
from benchnpin.common.controller.dp import DP
import numpy as np


class PlanningBasedPolicy():
    """
    A baseline policy for autonomous ship navigation in ice-covered waters. 
    This policy first plans a path using a ship planner and outputs actions to track the planned path.
    """

    def __init__(self, planner_type, goal, conc) -> None:

        if planner_type not in ['predictive', 'lattice']:
            raise Exception("Invalid planner type. Choose a planner between 'lattice' or 'predictive'.")
        self.planner_type = planner_type

        self.lattice_planner = LatticePlanner()
        self.predictive_planner = PredictivePlanner()

        self.goal = goal
        self.path = None
        self.conc = conc

    
    def plan_path(self, ship_pos, goal, observation, obstacles=None):
        if self.planner_type == 'lattice':
            self.path = self.lattice_planner.plan(ship_pos=ship_pos, goal=goal, obs=obstacles)

        elif self.planner_type == 'predictive':
            occ_map = observation[0]
            footprint = observation[1]
            self.path = self.predictive_planner.plan(ship_pos=ship_pos, goal=goal, occ_map=occ_map, footprint=footprint, conc=self.conc)


    def act(self, ship_pos, observation, obstacles=None):
        
        # plan a path
        if self.path is None:
            self.plan_path(ship_pos, self.goal, observation, obstacles)

            # setup dp controller to track the planned path
            cx = self.path.T[0]
            cy = self.path.T[1]
            ch = self.path.T[2]
            self.dp = DP(x=ship_pos[0], y=ship_pos[1], yaw=ship_pos[2],
                    cx=cx, cy=cy, ch=ch, **self.lattice_planner.cfg.controller)
            self.dp_state = self.dp.state
        
        # call ideal controller to get angular velocity control
        omega, _ = self.dp.ideal_control(ship_pos[0], ship_pos[1], ship_pos[2])

        # update setpoint
        x_s, y_s, h_s = self.dp.get_setpoint()
        self.dp.setpoint = np.asarray([x_s, y_s, np.unwrap([self.dp_state.yaw, h_s])[1]])

        return omega

    
    def reset(self):
        self.path = None

