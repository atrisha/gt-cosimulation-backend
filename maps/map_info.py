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

