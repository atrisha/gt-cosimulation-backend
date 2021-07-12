'''
Created on Mar 5, 2021

@author: Atrisha
'''

import math

class NYCMapInfo:
    
    vehicle_lane = [(0,4), (13.5,4), (13.5,24), (6.5,24), (6.5,12), (0,12), (0,4)]
    ped_lane = [(3,4), (6.5,4), (6.5,12), (3,12), (3,4)]
    veh_centerline = [(8.25,24), (8.54,13.0),(7.46,11.09), (5.38,10.37), (0,10)]
    ped_centerline = [(4.75,3),(4.75,8),(4.75,16)]
    
    def get_max_stop_dist(self,p):
        
        return math.hypot(p[0]-NYCMapInfo.veh_centerline[2][0],p[1]-NYCMapInfo.veh_centerline[2][1])



class IntersectionClearanceMapInfo:
    
    st2_waypoints = [(538796.97, 4814055.18), (538814.18, 4814030.12), (538818.07, 4814025.05), (538820.96, 4814020.6), (538846.4, 4813982.36)]
    st2_waypoint_segments = ['ln_n_3', 'ln_n_3', 'ln_n_3', 'l_n_s_r', 'ln_s_-2']
    st2_on_intersection_distance = (45,70)
    
    st1_waypoints = [(538810.51, 4814034.38), (538822.99, 4814017.66), (538824.65, 4814015.29), (538826.91, 4814012.06), (538829.43, 4814008.4), (538831.76, 4814004.96), (538834.21, 4814001.28), (538836.99, 4813997.07), (538839.34, 4813993.49), (538841.51, 4813990.19), (538843.84, 4813986.63)]
    st1_waypoint_segments = ['ln_n_3', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'ln_s_-2', 'ln_s_-2']
    st1_on_intersection_distance = (20,45)
    
    lt_waypoints = [(538841.75, 4814001.62), (538840.82, 4814003), (538839.06, 4814005.3), (538835.9, 4814007.9), (538831.19, 4814009.78), (538824.85, 4814009.83), (538816.67, 4814007.44), (538807.52, 4814003.24), (538797.61, 4813998.1), (538787.83, 4813992.66)]
    lt_waypoint_segments = ['prep-turn_s', 'prep-turn_s', 'exec-turn_s', 'exec-turn_s', 'exec-turn_s', 'exec-turn_s', 'exec-turn_s', 'ln_w_-1', 'ln_w_-1', 'ln_w_-1']
    
    