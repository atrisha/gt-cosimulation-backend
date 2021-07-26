'''
Created on Apr 14, 2021

@author: Atrisha
'''
import sqlite3
from shapely.geometry import LineString, Point
from planners.trajectory_planner import WaitTrajectoryConstraints, ProceedTrajectoryConstraints
import constants
import numpy as np
import math
import code_utils.utils as rg_utils
from all_utils import utils

class AgentState:
    
    def __init__(self,attrib_map):
        for k,v in attrib_map.items():
            setattr(self, k, v)
    '''   
    @property
    def x(self):
        return self.x
    
    @property
    def y(self):
        return self.y
    
    @property
    def velocity(self):
        return self.velocity
    
    @property
    def waypoints(self):
        return self.waypoints
    '''
        
class VehicleState(AgentState):
    
    def __init__(self,attrib_map):
        super().__init__(attrib_map)


class PedestrianState(AgentState):
    
    def __init__(self,attrib_map):
        super().__init__(attrib_map)
        

class UnsupportedManeuverException(Exception):
    
    def __init__(self,message):
        super().__init__(message)

class TrajectoryConstraintsFactory:
    
    @staticmethod
    def get_constraint_object(maneuver,ag_obj,lead_ag_obj):
        if maneuver == 'track_speed':
            vel_pts_proc = [(ag_obj.velocity,)] + [(None,) if i != len(np.arange(1,len(ag_obj.waypoints)-1))//2 else rg_utils.get_reasonable_velocities(ag_obj.waypoint_segments[i], ag_obj.direction, ag_obj) for i in np.arange(1,len(ag_obj.waypoints)-1)] + [rg_utils.get_reasonable_velocities(ag_obj.waypoint_segments[-1], ag_obj.direction, ag_obj)]
            constr = ProceedTrajectoryConstraints(waypoints=ag_obj.waypoints,waypoint_vel_sampling_range=vel_pts_proc)
        elif maneuver in constants.WAIT_ACTIONS:
            if ag_obj.velocity <= 1:
                constr = WaitTrajectoryConstraints(init_vel=ag_obj.velocity,waypoints=ag_obj.waypoints,stop_horizon_dist_sampling_range=(0,5),stop_horizon_time_sampling_range=(0,3))
            else:
                est_stop_dist_range = (max(0,(0.1*ag_obj.velocity - 0.03)), max(5,(3*ag_obj.velocity-4.5)))
                constr = WaitTrajectoryConstraints(init_vel=ag_obj.velocity,waypoints=ag_obj.waypoints,stop_horizon_dist_sampling_range=est_stop_dist_range,stop_horizon_time_sampling_range=(0,3))
        elif maneuver == 'proceed-turn':
            vel_pts_proc = [(ag_obj.velocity,)] + [(None,) if i != len(np.arange(1,len(ag_obj.waypoints)-1))//2 else rg_utils.get_reasonable_velocities(ag_obj.waypoint_segments[i], ag_obj.direction, ag_obj) for i in np.arange(1,len(ag_obj.waypoints)-1)] + [rg_utils.get_reasonable_velocities(ag_obj.waypoint_segments[-1], ag_obj.direction, ag_obj)]
            constr = ProceedTrajectoryConstraints(waypoints=ag_obj.waypoints,waypoint_vel_sampling_range=vel_pts_proc)
        elif maneuver in ['follow_lead','follow_lead_into_intersection']:   
            _min_dist_2_lead = [math.hypot(lead_ag_obj.waypoints[0][0]-x[0],lead_ag_obj.waypoints[0][1]-x[1]) for x in ag_obj.waypoints]
            _mindist = min(_min_dist_2_lead)
            _mindist_idx = _min_dist_2_lead.index(_mindist)
            lead_vel = lead_ag_obj.velocity
            if lead_ag_obj.velocity > 2:
                ''' lead vehicle is moving forward now'''
                if _mindist_idx > 0:
                    vel_pts_proc = [(ag_obj.velocity,)] + [(None,) if i != _mindist_idx else (lead_vel-1,lead_vel+1) for i in np.arange(1,len(ag_obj.waypoints)-1)] + [(lead_vel-1,lead_vel+1)]
                else:
                    vel_pts_proc = [(ag_obj.velocity,)] + [(None,) if i != 1 else (lead_vel-1,lead_vel+1) for i in np.arange(1,len(ag_obj.waypoints)-1)] + [(lead_vel-1,lead_vel+1)]
                constr = ProceedTrajectoryConstraints(waypoints=ag_obj.waypoints,waypoint_vel_sampling_range=vel_pts_proc)
            else:
                ''' lead vehicle is slow, better to wait '''
                _dist2lead = LineString(ag_obj.waypoints).project(Point(lead_ag_obj.x,lead_ag_obj.y)) - LineString(ag_obj.waypoints).project(Point(ag_obj.x,ag_obj.y))
                stop_horizon_dist_sampling_range = (min(_dist2lead-10,0),max(_dist2lead,0.1))
                stop_horizon_time_sampling_range = (max(0,(0.1*ag_obj.velocity - 0.03)), max(5,(3*ag_obj.velocity-4.5)))
                constr = WaitTrajectoryConstraints(init_vel=ag_obj.velocity,waypoints=ag_obj.waypoints,stop_horizon_dist_sampling_range=stop_horizon_dist_sampling_range,stop_horizon_time_sampling_range=stop_horizon_time_sampling_range)
        elif maneuver == 'cut-in':
            _mididx = int(len(ag_obj.waypoints)//2)
            _newpath = utils.add_parallel(ag_obj.waypoints[_mididx:], 3)[0]
            ag_obj.waypoints = ag_obj.waypoints[:_mididx] + _newpath
            vel_pts_proc = [(ag_obj.velocity,)] + [(None,) if i != len(np.arange(1,len(ag_obj.waypoints)-1))//2 else rg_utils.get_reasonable_velocities(ag_obj.waypoint_segments[i], ag_obj.direction, ag_obj) for i in np.arange(1,len(ag_obj.waypoints)-1)] + [rg_utils.get_reasonable_velocities(ag_obj.waypoint_segments[-1], ag_obj.direction, ag_obj)]
            constr = ProceedTrajectoryConstraints(waypoints=ag_obj.waypoints,waypoint_vel_sampling_range=vel_pts_proc)
        else:   
            raise UnsupportedManeuverException(maneuver)
        constr.lateral_path_sampling = True
        return constr
























            