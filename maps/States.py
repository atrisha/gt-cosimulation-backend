'''
Created on Mar 31, 2021

@author: Atrisha
'''
import sqlite3
import numpy as np
from numpy import linalg as LA
import itertools

act_dict = {'pedestrian':{
                'maneuvers' : ['wait','walk'],
                'manv_modes' : ['aggressive','normal']
            },
            'vehicle':{
                'maneuvers' : ['wait','turn'],
                'manv_modes' : ['aggressive','normal']
            }
            }
class States:

    def build_states(self):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        for t in [2,4]:
            print('---------------------',t,'secs ---------------------------------')
            for ag in ['pedestrian','vehicle']:
                tot_states = 0
                for manv in act_dict[ag]['maneuvers']:
                    
                        
                    q_string = "SELECT MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X),ABS(INIT_POS_Y-Y) FROM TRAJECTORY_METADATA \
                                INNER JOIN TRAJECTORIES on TRAJECTORY_METADATA.TRAJ_ID = TRAJECTORIES.TRACK_ID \
                                    WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag+"' AND TRAJECTORIES.TIME="+str(t)+" \
                                    AND MANEUVER='"+manv+"'"
                    c.execute(q_string)
                    res = c.fetchall()
                    if len(res) == 0:
                        continue
                    speed = np.array([row[2] for row in res])
                    traj_l = np.array([LA.norm([x[3],x[4]]) for x in res])
                    print(ag,manv)
                    speed_states = np.arange(np.amin(speed),np.amax(speed)+.3,.3)
                    trajs_l_states =  np.arange(np.amin(traj_l), np.amax(traj_l)+.5,.5)
                    num_states = len(speed_states)*len(trajs_l_states)
                    print('num states',num_states)
                    if num_states > 100:
                        brk = 1
                    tot_states += num_states
                print('total',ag,'states: ',tot_states)

s = States()
s.build_states()
        