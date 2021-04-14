'''
Created on Mar 31, 2021

@author: Atrisha
'''
import sqlite3
import numpy as np
from numpy import linalg as LA
from maps.map_info import NYCMapInfo
import itertools
import math
import matplotlib.pyplot as plt
from repeated_game import Actions, Utilities


act_dict = {'pedestrian':{
                'maneuvers' : ['wait','walk'],
                'manv_modes' : ['aggressive','normal']
            },
            'vehicle':{
                'maneuvers' : ['wait','turn'],
                'manv_modes' : ['aggressive','normal']
            }
            }

WAIT_MANEUVERS = ['wait']


def find_index_in_list(s_sum, dist_from_origin):
    if s_sum > dist_from_origin[-1]:
        return len(dist_from_origin)-1
    idx = None
    for i,x in enumerate(list(zip(dist_from_origin[:-1],dist_from_origin[1:]))):
        if x[0] <= s_sum <= x[1]:
            idx = i
            break
    return idx

def construct_centerline(dist,ag,v0):
    path = NYCMapInfo.veh_centerline if ag == 'vehicle' else NYCMapInfo.ped_centerline
    waypoint_velocity = [(v0,),(None,),(1,5),(None,),(5,10)] if ag == 'vehicle' else [(v0,),(.5,1.8),(.5,1.8)]
    dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(path[:-1],path[1:]))]
    dist_from_origin = [sum(dist_from_origin[:i]) for i in np.arange(1,len(dist_from_origin))]
    path_idx = find_index_in_list(dist, dist_from_origin)
    if path_idx is None:
        print(dist)
        print(dist_from_origin)
        raise IndexError("path_idx is None")
    if path_idx >= len(path)-1:
        path_idx = path_idx - 1
    if path_idx >= len(dist_from_origin)-1:
        underflow = None
    else:
        underflow = dist_from_origin[path_idx+1] - dist
    print(len(dist_from_origin),path_idx)
    overflow = dist - dist_from_origin[path_idx]
    r = overflow/math.hypot(path[path_idx+1][0]-path[path_idx][0], path[path_idx+1][1]-path[path_idx][1]) if overflow != 0 else 0
    point_x = path[path_idx][0] + r*(path[path_idx+1][0] - path[path_idx][0])
    point_y = path[path_idx][1] + r*(path[path_idx+1][1] - path[path_idx][1])
    point = (point_x,point_y)
    if underflow is not None and underflow < 2:
        if path_idx+2 <= len(path)-1 :
            new_path = [point] + path[path_idx+2:]
            new_waypt_vel = [(v0,)] + waypoint_velocity[path_idx+2:]
        else:
            new_path = [point] + path[path_idx+1:]
            new_waypt_vel = [(v0,)] + waypoint_velocity[path_idx+1:]
    else:
        new_path = [point] + path[path_idx+1:]
        new_waypt_vel = [(v0,)] + waypoint_velocity[path_idx+1:]
    return new_path,new_waypt_vel

class TreeBuilder:


    def build_initial_reachability_states(self):
        init_time = 0
        veh_init_vel = 5
        ped_init_vel = 1.38
        time_horizon = 6
        acts = Actions()
        acts.generate_actions(init_time,veh_init_vel,ped_init_vel,time_horizon,True)
        acts.insert_interaction_data()
        
    def get_current_states(self,time_intervals):
        state_lattice = dict()
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        for t in time_intervals:
            print('---------------------',t,'secs ---------------------------------')
            state_lattice[t[0]+t[1]] = dict()
            for ag in ['pedestrian','vehicle']:
                tot_states = 0
                state_lattice[t[0]+t[1]][ag] = dict()
                print("--",ag,'init states (s,x,y)',"--")
                for manv in act_dict[ag]['maneuvers']:
                    q_string = "SELECT MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X),ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA \
                                INNER JOIN TRAJECTORIES on TRAJECTORY_METADATA.TRAJ_ID = TRAJECTORIES.TRACK_ID \
                                    WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag+"' AND TRAJECTORIES.TIME="+str(t[1])+" AND TRAJECTORY_METADATA.INIT_TIME="+str(t[0])+"\
                                    AND MANEUVER='"+manv+"'"
                    c.execute(q_string)
                    res = c.fetchall()
                    if len(res) == 0:
                        continue
                    speed = np.array([row[2] for row in res])
                    traj_l = np.array([(LA.norm([x[3],x[4]]),x[5],x[6],x[7],x[2],x[8]) for x in res])
                    print(ag,manv)
                    speed_states = np.arange(np.amin(speed),np.amax(speed)+.3,.3)
                    trajs_l_states = []
                    for i,t_p in enumerate(traj_l):
                        if i == 0:
                            trajs_l_states.append((t_p[1],t_p[2],t_p[0],t_p[4],t_p[5]))
                        else:
                            if abs(traj_l[i-1][0] - t_p[0]) >= 0.5 or abs(traj_l[i-1][4] - t_p[4]) >= 0.3:
                                trajs_l_states.append((t_p[1],t_p[2],t_p[0],t_p[4],t_p[5]))
                        
                    #num_states = len(speed_states)*len(trajs_l_states)
                    num_states = len(trajs_l_states)
                    #all_init_states = list(itertools.product(speed_states,trajs_l_states))
                    all_init_states = trajs_l_states
                    state_lattice[t[0]+t[1]][ag][manv] = list(all_init_states)
                    '''
                    for i_s in all_init_states:
                        print(i_s)
                    '''
                    print('num states',num_states)
                    if num_states > 100:
                        brk = 1
                    tot_states += num_states
                print('total',ag,'states: ',tot_states)
        return state_lattice
    
    def insert_trajs_into_db(self,trajs, ag, init_time, parent_traj_id):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        i_string = 'INSERT INTO TRAJECTORIES VALUES (?,?,?,?,?,?,?,?,?)'
        i_string_tj_mtdata = 'INSERT INTO TRAJECTORY_METADATA VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        q_string = "select count(*) from trajectory_METADATA"
        c.execute(q_string)
        res = c.fetchone()
        traj_id = res[0]+1
        for idx,traj_det_dict in enumerate([trajs]):
            ag_type = ag
            trajs, traj_metadata = [],[]
            for traj_manv,tm_v in traj_det_dict.items():
                for traj_mode,tmd_v in tm_v.items():
                    for trj in tmd_v:
                        traj_entry = [(traj_id,float(x[1]),float(x[2]),float(x[3]),float(x[4]),x[6],x[0],x[7],None) for x in trj] 
                        traj_mtdt_entry = [(traj_id,float(trj[0][1]),float(trj[0][2]),float(trj[0][3]),float(trj[0][4]),float(trj[-1][3]),traj_manv,traj_mode,ag_type,init_time,parent_traj_id)]
                        traj_metadata.extend(traj_mtdt_entry)
                        trajs.extend(traj_entry)
                        traj_id += 1
            c.executemany(i_string,trajs)
            c.executemany(i_string_tj_mtdata,traj_metadata)
            print('agent trajectories inserted')           
        
        
        conn.commit()
        conn.close()
        
    
    def build_final_trajectories(self):
        #time_interval_axes = [[(0,2)], [(0,4),(2,2)]]
        time_interval_axes = [[(0,4)]]
        for time_intervals in time_interval_axes:
            state_lattics = self.get_current_states(time_intervals)
            for ts,ts_v in state_lattics.items():
                for ag,ag_v in ts_v.items():
                    for manv,manv_v in ag_v.items():
                        ct,N = 0,len(manv_v)
                        for init_st in manv_v:
                            ct += 1
                            x = init_st[0]
                            y = init_st[1]
                            v = init_st[3]
                            if ct ==7:
                                brk = 1
                            
                            waypt,waypt_vel = construct_centerline(init_st[2],ag,v)
                            
                            act = Actions()
                            generating_manv = [x for x in act_dict[ag]['maneuvers'] if x != manv][0]
                            horizon = 6 - ts
                            parent_traj_id = init_st[4]
                            trajs = act.generate_agent_action(ts, v, waypt, waypt_vel, generating_manv, ag, horizon)
                            self.insert_trajs_into_db(trajs, ag, ts, parent_traj_id)
                            print('generating',ag,'time',ts,'manv',generating_manv,ct,'/',N)
                
       
    def build_complete_tree(self):
        self.build_initial_reachability_states()
        self.build_final_trajectories()

'''
def show_all_trajectories():
    q_string = "SELECT MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X),ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA \
                                INNER JOIN TRAJECTORIES on TRAJECTORY_METADATA.TRAJ_ID = TRAJECTORIES.TRACK_ID \
                                    WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag+"' AND TRAJECTORIES.TIME="+str(t[1])+" AND TRAJECTORY_METADATA.INIT_TIME="+str(t[0])+"\
                                    AND MANEUVER='"+manv+"'"
                    c.execute(q_string)
                    res = c.fetchall()
'''
#s = TreeBuilder()
#s.build_final_trajectories()

                
                
        