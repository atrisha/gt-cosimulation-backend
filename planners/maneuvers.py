'''
Created on Mar 5, 2021

@author: Atrisha
'''
from maps.map_info import NYCMapInfo
import numpy as np

def right_turn_wait(v0,parms):
    
    vel_map = { '1': (0,None),
                '2': (0,0),
                '3:': (0,0) }
    vel_pts = []
    for i in np.arange(len(NYCMapInfo.veh_centerline)):
        if i == 0:
            vel_pts.append(v0)
        else:
            if str(i) in vel_map:
                vel_pts.append(vel_map[str(i)][0])
            elif str(i)+':' in vel_map:
                for j in np.arange(i,len(NYCMapInfo.veh_centerline)):
                    vel_pts.append(vel_map[str(i)+':'][0])
                break
    return vel_pts
                
                
                