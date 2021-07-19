'''
Created on Apr 14, 2021

@author: Atrisha
'''


import numpy as np
import sqlite3
import itertools
from planners.planning_objects import VehicleState, PedestrianState
import time
from equilibrium.equilibria_calculation import *
import copy
from maps.map_info import NYCMapInfo
from mpl_toolkits.mplot3d import Axes3D
from planners.trajectory_planner import VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner, WaitTrajectoryConstraints, ProceedTrajectoryConstraints
from equilibrium.utilities import Utilities
from numpy import linalg as LA
import math
from maps.States import ScenarioDef
import constants
import csv
from all_utils.utils import pickle_dump_to_dir
import rg_constants
from code_utils.utils import get_all_level_nodes, get_nearest_node,\
    lighten_color
import os
import traceback
import sys
from equilibrium.range_estimation import MinDistanceGapModel
from equilibrium.gametree_objects import TrajectoryCache, TrajectoryFragment, UnsupportedAgentObservationException,UnsupportedLatticeException
from code_utils.code_util_objects import RunContext
import all_utils
import code_utils.utils as rg_utils
import copy
from os import listdir
import matplotlib.pyplot as plt
import ast
from rg_visualizer import UniWeberAnalytics
from os.path import isfile, join


log = constants.common_logger
from equilibrium.automata_strategies import *

show_plots = False


        

'''
act_dict = {'pedestrian':{
                'maneuvers' : ['wait','walk'],
                'manv_modes' : ['aggressive','normal']
            },
            'vehicle':{
                'maneuvers' : ['wait','turn'],
                'manv_modes' : ['aggressive','normal']
            }
            }
'''


modes = ['normal','aggressive']


class AssignDistRanges:
        
    
    def assign_distranges(self,node,last_decision_level,model):
        if node.level == last_decision_level:
            for c in node.children:
                self.assign_distranges(c,last_decision_level,model)
                
            self.predict_dist_ranges(node, model, last_decision_level)
        else:
            if not node.is_leaf: 
                for c in node.children:
                    self.assign_distranges(c,last_decision_level,model)
                if not node.is_root:
                    self.predict_dist_ranges(node, model, last_decision_level)
                else:
                    node.automata_strategy_info = None
            else:
                #node.automata_strategy_info = None
                self.predict_dist_ranges(node, model, last_decision_level)
    
    def predict_dist_ranges(self, node, model, last_decision_level):
        node.automata_strategy_info = dict()
        for ag in ['agent_1','agent_2']:
            obs_trajl = [node.path_from_root[ag].length]
            obs_manvs = [node.path_from_root[ag].manv]
            obs_manvs_mode = [node.path_from_root[ag].manv_mode]
            distgap_model = model.distgap_model_agent_1 if ag == 'agent_1' else model.distgap_model_agent_2
            wait_manv = 'wait'
            proceed_manv = 'turn' if ag == 'agent_1' else 'track_speed'
            predicted_distgap_range = distgap_model.predict(list(zip(obs_trajl,obs_manvs)), {'wait':wait_manv,'proceed':proceed_manv})
            automata_strat_info = [(obs_manvs[-1],predicted_distgap_range)]
            if node.level >= last_decision_level and node.level > 2:
                obs_trajl.append(obs_trajl[-1]+node.path_from_root[ag].next_fragment.length)
                obs_manvs.append(node.path_from_root[ag].next_fragment.manv)
                obs_manvs_mode.append(node.path_from_root[ag].next_fragment.manv_mode)
                predicted_distgap_range = distgap_model.predict(list(zip(obs_trajl,obs_manvs)), {'wait':wait_manv,'proceed':proceed_manv})
                automata_strat_info.append((obs_manvs[-1],predicted_distgap_range))
            if node.level == last_decision_level+2:
                obs_trajl.append(obs_trajl[-1]+node.path_from_root[ag].next_fragment.next_fragment.length)
                obs_manvs.append(node.path_from_root[ag].next_fragment.next_fragment.manv)
                obs_manvs_mode.append(node.path_from_root[ag].next_fragment.next_fragment.manv_mode)
                predicted_distgap_range = distgap_model.predict(list(zip(obs_trajl,obs_manvs)), {'wait':wait_manv,'proceed':proceed_manv})
                automata_strat_info.append((obs_manvs[-1],predicted_distgap_range))
            ac_auto = AccommodatingGamma(wait_manv,proceed_manv)
            ac_auto_gamma = ac_auto.accepts(automata_strat_info)
            nac_auto = NonAccomodatingGamma(wait_manv,proceed_manv)
            nac_auto_gamma = nac_auto.accepts(automata_strat_info)
            
            node.automata_strategy_info[ag] = {'obs_trajl':obs_trajl,
                                               'obs_manvs':obs_manvs,
                                               'obs_manvs_mode':obs_manvs_mode,
                                               'predicted_distgap_range':predicted_distgap_range,
                                               'ac_auto_gamma':ac_auto_gamma,
                                               'nac_auto_gamma':nac_auto_gamma}
            
            
        
        
class Actions:
    
    def __init__(self,maneuver_constraints):
        self.maneuver_constraints = maneuver_constraints
    
    def generate_agent_action(self,init_time,init_veh_vel,waypoint,waypoint_vels,manv,ag,horizon,parent_trajectory_arcl=0):
        trajs = dict()
        if manv in ['wait']:
            if ag == 'agent_1':
                stop_horizon_dist_sampling_range = (1,5)
                stop_horizon_time_sampling_range = (1,4)
            else:
                if rg_constants.SCENE_TYPE[0] == 'REAL':
                    stop_horizon_dist_sampling_range = (self.maneuver_constraints[ag]['maneuvers']['wait'].stop_horizon_dist_sampling_range[0]-parent_trajectory_arcl,self.maneuver_constraints[ag]['maneuvers']['wait'].stop_horizon_dist_sampling_range[1]-parent_trajectory_arcl)
                    stop_horizon_time_sampling_range = (1,4)
                elif rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['test','intersection_clearance']:
                    if self.maneuver_constraints[ag]['agent_state'].id == 2:
                        stop_horizon_dist_sampling_range = (IntersectionClearanceMapInfo.st1_on_intersection_distance[0]-5,IntersectionClearanceMapInfo.st1_on_intersection_distance[0]) 
                    else:
                        stop_horizon_dist_sampling_range = (IntersectionClearanceMapInfo.st2_on_intersection_distance[0]-5,IntersectionClearanceMapInfo.st2_on_intersection_distance[0])
                    stop_horizon_time_sampling_range = (1,4)
                else:
                    stop_horizon_dist_sampling_range = (10,100)
                    stop_horizon_time_sampling_range = (1,10)
            manv_constr = WaitTrajectoryConstraints(init_vel=init_veh_vel,waypoints=waypoint,stop_horizon_dist_sampling_range=stop_horizon_dist_sampling_range,stop_horizon_time_sampling_range=stop_horizon_time_sampling_range)
        else:
            manv_constr = ProceedTrajectoryConstraints(waypoints=waypoint, waypoint_vel_sampling_range=waypoint_vels)
        manv_constr.set_limit_constraints()
        manv_constr.parent_trajectory_arcl = parent_trajectory_arcl
        agent_motion = VehicleTrajectoryPlanner(traj_constr_obj=manv_constr, maneuver=manv, mode=None, horizon=horizon)
        agent_motion.generate_trajectory(True)
        trajs[manv] = agent_motion.all_trajectories
        return trajs
                
        
    
    def generate_actions(self,init_time,agent1_init_vel,agent2_init_vel,horizon, insert_into_db = False):
        agent1_trajs,agent2_trajs = dict(), dict()
        agent1_trajs,agent2_trajs = dict(), dict()
        maneuver_constraints = self.maneuver_constraints
        maneuver_constraints['motion_info'] = dict()
        for manv,manv_constr in maneuver_constraints['agent_1']['maneuvers'].items():
            manv_constr.set_limit_constraints()
            agent1_motion = VehicleTrajectoryPlanner(traj_constr_obj=manv_constr,maneuver= manv, mode=None, horizon=horizon)
            agent1_motion.generate_trajectory(True)
            agent1_trajs[manv] = agent1_motion.all_trajectories
            if 'agent_1' not in maneuver_constraints['motion_info']:
                maneuver_constraints['motion_info']['agent_1'] = dict()
            maneuver_constraints['motion_info']['agent_1'][manv] = agent1_motion
        
        for manv,manv_constr in maneuver_constraints['agent_2']['maneuvers'].items():
            manv_constr.set_limit_constraints()
            agent2_motion = VehicleTrajectoryPlanner(traj_constr_obj=manv_constr,maneuver= manv, mode=None, horizon=horizon)
            agent2_motion.generate_trajectory(True)
            agent2_trajs[manv] = agent2_motion.all_trajectories
            if 'agent_2' not in maneuver_constraints['motion_info']:
                maneuver_constraints['motion_info']['agent_2'] = dict()
            maneuver_constraints['motion_info']['agent_2'][manv] = agent2_motion
        if rg_constants.SCENE_TYPE[0] == 'REAL':
            file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        else:
            file_id =  rg_constants.CURRENT_RG_FILE_ID
        '''
        plt.plot([x[0] for x in maneuver_constraints['agent_1']['agent_state'].waypoints],[x[1] for x in maneuver_constraints['agent_1']['agent_state'].waypoints],color='blue',marker='x')
        plt.plot([x[0] for x in maneuver_constraints['agent_2']['agent_state'].waypoints],[x[1] for x in maneuver_constraints['agent_2']['agent_state'].waypoints],color='red',marker='x')
        plt.show()
        '''
        path = maneuver_constraints['agent_1']['agent_state'].waypoints
        dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(path[:-1],path[1:]))]
        dist_from_origin = np.cumsum(dist_from_origin)
        path = maneuver_constraints['agent_2']['agent_state'].waypoints
        dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(path[:-1],path[1:]))]
        dist_from_origin = np.cumsum(dist_from_origin)
        f=1
        
        if insert_into_db:
            parent_traj_id = None
            conn = sqlite3.connect(rg_constants.get_rg_db_path(file_id))
            c = conn.cursor()
            i_string = 'INSERT INTO TRAJECTORIES VALUES (?,?,?,?,?,?,?,?,?)'
            i_string_tj_mtdata = 'INSERT INTO TRAJECTORY_METADATA VALUES (?,?,?,?,?,?,?,?,?,?,?)'
            traj_id = 1
            for ag_type_idx,traj_det_dict in enumerate([agent1_trajs,agent2_trajs]):
                ag_type = 'agent_1' if ag_type_idx == 0 else 'agent_2'
                trajs, traj_metadata = [],[]
                for traj_manv,tm_v in traj_det_dict.items():
                    for traj_mode,tmd_v in tm_v.items():
                        for trj in tmd_v:
                            traj_entry = [(traj_id,float(x[1]),float(x[2]),float(x[3]),float(x[4]),x[6],x[0],x[7],x[8]) for x in trj] 
                            traj_mtdt_entry = [(traj_id,float(trj[0][1]),float(trj[0][2]),float(trj[0][3]),float(trj[0][4]),float(trj[-1][3]),traj_manv,traj_mode,ag_type,init_time,parent_traj_id)]
                            traj_metadata.extend(traj_mtdt_entry)
                            trajs.extend(traj_entry)
                            traj_id += 1
                c.executemany(i_string,trajs)
                c.executemany(i_string_tj_mtdata,traj_metadata)
                print('agent trajectories inserted')           
            conn.commit()
            conn.close()        
            
        return agent1_trajs, agent2_trajs
    
    def insert_interaction_data(self):
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\right_turn_data.db')
        c = conn.cursor()
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='agent_1'"
        c.execute(q_string)
        veh_trajids = c.fetchall()
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='agent_2'"
        c.execute(q_string)
        ped_trajids = c.fetchall()
        all_trajs = dict()
        q_string = "SELECT * FROM TRAJECTORIES ORDER BY TRACK_ID,TIME"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[0] not in all_trajs:
                all_trajs[row[0]] = []
            all_trajs[row[0]].append(row[1:7])
        utils = Utilities()
        interac_id = 1
        interac_data = []
        ct,N = 0, len(veh_trajids)*len(ped_trajids)
        for vt_id,pt_id in itertools.product(veh_trajids,ped_trajids):
            ct += 1
            
            vt = all_trajs[vt_id[0]]
            pt = all_trajs[pt_id[0]]
            d_g = utils.calc_dist_gap(vt, pt)
            vt_l = utils.calc_traj_length(vt)
            pt_l = utils.calc_traj_length(pt)
            interac_entry = [(interac_id,vt_id[0],pt_id[0],d_g,None,vt_l,pt_l)]
            interac_data.extend(interac_entry)
            print('added',ct,'/',N,interac_entry)
            interac_id += 1
        i_string = 'INSERT INTO TRAJ_INTERACTIONS VALUES (?,?,?,?,?,?,?)'
        c.executemany(i_string,interac_data)
        conn.commit()
        conn.close()    
        
    
    def calc_dist_gaps(self):
        init_veh_vel,init_ped_vel = 5, 1.38
        veh_trajs, ped_trajs = self.generate_actions(init_veh_vel,init_ped_vel)
        utils = Utilities()
        util_info = dict()
        dist_gap_map = dict()
        vtidx,ptidx = 0,0
        for veh_m in veh_maneuvers:
            for ped_m in ped_maneuver:
                #Xp,Yp,Xv,Yv = [],[],[],[]
                for m in itertools.product(modes,modes):
                    if m[0] not in veh_trajs[veh_m] or m[1] not in ped_trajs[ped_m]:
                        continue  
                    v_tjs = veh_trajs[veh_m][m[0]]
                    p_tjs = ped_trajs[ped_m][m[1]]
                    
                    for vt in v_tjs:
                        vtidx += 1
                        for pt in p_tjs:
                            ptidx += 1
                            if (veh_m,ped_m) not in dist_gap_map:
                                dist_gap_map[(veh_m,ped_m)] = []
                            d_g = utils.generate_safety_utils(pt, vt)
                            
                            if veh_m == 'turn' and ped_m == 'walk' and d_g > 5:
                                #simulation_sketch.simulate_trajectory(pt, vt,veh_m,ped_m)
                                #plt.plot([x[0] for x in vt],[x[3] for x in vt])
                                #plt.show()
                                pass
                            dist_gap_map[(veh_m,ped_m)].append((vtidx,ptidx,d_g,math.hypot(vt[0][1]-vt[-1][1], vt[0][2]-vt[-1][2]),math.hypot(pt[0][1]-pt[-1][1], pt[0][2]-pt[-1][2])))
                            '''
                            Xv.append(utils.generate_safety_utils(pt,vt))
                            Yv.append(utils.progress_payoff_dist(math.hypot(vt[0][1]-vt[-1][1], vt[0][2]-vt[-1][2]), 'veh'))
                            Xp.append(utils.generate_safety_utils(pt,vt))
                            Yp.append(utils.progress_payoff_dist(math.hypot(pt[0][1]-pt[-1][1], pt[0][2]-pt[-1][2]), 'ped'))
                            '''
                '''
                plt.figure()
                plt.plot(Xv,Yv,'.')
                plt.title('veh '+str(veh_m)+str(ped_m))
                plt.figure()
                plt.plot(Xp,Yp,'.')
                plt.title('ped '+str(veh_m)+str(ped_m))
                plt.show()
                '''     
        for k,v in dist_gap_map.items():
            u_range = [min([x[2] for x in v]),max([x[2] for x in v])]
            print(k," : ",str(u_range))
            util_info[k] = u_range
        utils.util_info = util_info
        self.utils = utils
        self.dist_gap_map = dist_gap_map
        
        '''           
        for ped_traj in ped_motion.all_trajectories:
            for veh_traj in veh_motion.all_trajectories:
                dist_gap = utils.generate_safety_utils(ped_traj, veh_traj)
                if (veh_m,ped_m) not in util_info:
                    util_info[(veh_m,ped_m)] = []
                util_info[(veh_m,ped_m)].append(dist_gap)
        print('utils ranges (dist gaps) ------------- ')
        for k,v in util_info.items():
            u_range = [min(v),max(v)]
            print(k," : ",str(u_range))   
        '''
    
    '''
    utils = Utilities()    
    util_info = {('turn', 'walk')  :  [1.453167769746284, 2.430105932669038],
                ('turn', 'wait')  :  [4.383906901195861, 5.968458852783996],
                ('wait', 'walk')  :  [2.87153787003464, 9.258464652266861],
                ('wait', 'wait')  :  [6.546044963774584, 15.797224181634268]}
    utils.util_info = util_info
    '''               





def find_index_in_list(s_sum, dist_from_origin):
    if s_sum > dist_from_origin[-1]:
        return len(dist_from_origin)-1
    idx = None
    for i,x in enumerate(list(zip(dist_from_origin[:-1],dist_from_origin[1:]))):
        if x[0] <= s_sum <= x[1]:
            idx = i
            break
    return idx



class TreeBuilder:
    
    def construct_centerline(self,dist,ag,v0,maneuver_constraints,point):
        path = maneuver_constraints[ag]['agent_state'].waypoints
        if rg_constants.SCENE_TYPE == ('synthetic','merge_before_intersection'):
            waypoint_velocity = maneuver_constraints[ag]['maneuvers']['turn'].waypoint_vel_sampling_range
        else:
            waypoint_velocity = maneuver_constraints[ag]['maneuvers']['turn'].waypoint_vel_sampling_range if ag == 'agent_1' else maneuver_constraints[ag]['maneuvers']['track_speed'].waypoint_vel_sampling_range
        dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(path[:-1],path[1:]))]
        dist_from_origin = np.cumsum(dist_from_origin)
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
        overflow = dist - dist_from_origin[path_idx]
        r = overflow/math.hypot(path[path_idx+1][0]-path[path_idx][0], path[path_idx+1][1]-path[path_idx][1]) if overflow != 0 else 0
        point_x = path[path_idx][0] + r*(path[path_idx+1][0] - path[path_idx][0])
        point_y = path[path_idx][1] + r*(path[path_idx+1][1] - path[path_idx][1])
        #point = (point_x,point_y)
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
        if len(new_path) != len(new_waypt_vel):
            f=1
        return new_path,new_waypt_vel


    def build_initial_reachability_states(self, maneuver_constraints):
        init_time = 0
        if rg_constants.SCENE_TYPE[0] == 'REAL':
            self.file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        else:
            self.file_id = rg_constants.CURRENT_RG_FILE_ID
        agent1_init_vel = maneuver_constraints['agent_1']['agent_state'].velocity
        agent2_init_vel = maneuver_constraints['agent_2']['agent_state'].velocity
        time_horizon = self.horizon
        acts = Actions(maneuver_constraints)
        acts.generate_actions(init_time,agent1_init_vel,agent2_init_vel,time_horizon,self.initialize_db)
        #acts.insert_interaction_data()
        
    def get_current_states(self,time_intervals,maneuver_constraints):
        state_lattice = dict()
        conn = sqlite3.connect(rg_constants.get_rg_db_path(self.file_id))
        c = conn.cursor()
        for t in time_intervals:
            print('---------------------',t,'secs ---------------------------------')
            state_lattice[t[0]+t[1]] = dict()
            for ag in ['agent_1','agent_2']:
                tot_states = 0
                state_lattice[t[0]+t[1]][ag] = dict()
                lattice_dist_step = 0.5 if ag == 'agent_1' else 5
                
                print("--",ag,'init states (s,x,y)',"--")
                for manv in maneuver_constraints[ag]['maneuvers'].keys():
                    q_string = "SELECT MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X),ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.TRAJ_ID,ARC_LENGTH FROM TRAJECTORY_METADATA \
                                INNER JOIN TRAJECTORIES on TRAJECTORY_METADATA.TRAJ_ID = TRAJECTORIES.TRACK_ID \
                                    WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag+"' AND TRAJECTORIES.TIME="+str(t[1])+" AND TRAJECTORY_METADATA.INIT_TIME="+str(t[0])+"\
                                    AND MANEUVER='"+manv+"'"
                    c.execute(q_string)
                    res = c.fetchall()
                    if len(res) == 0:
                        continue
                    speed = np.array([row[2] for row in res])
                    traj_l = np.array([(LA.norm([x[3],x[4]]),x[5],x[6],x[7],x[2],x[8],x[9]) for x in res])
                    print(ag,manv)
                    speed_states = np.arange(np.amin(speed),np.amax(speed)+.3,.3)
                    trajs_l_states = []
                    for i,t_p in enumerate(traj_l):
                        if i == 0:
                            trajs_l_states.append((t_p[1],t_p[2],t_p[0],t_p[4],t_p[5],t_p[6]))
                        else:
                            if abs(traj_l[i-1][0] - t_p[0]) >= lattice_dist_step:
                                '''manv_mode, speed, manv, y_dist, x_pos'''
                                trajs_l_states.append((t_p[1],t_p[2],t_p[0],t_p[4],t_p[5],t_p[6]))
                        
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
        conn = sqlite3.connect(rg_constants.get_rg_db_path(self.file_id))
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
                        traj_entry = [(traj_id,float(x[1]),float(x[2]),float(x[3]),float(x[4]),x[6],x[0],x[7],x[8]) for x in trj] 
                        traj_mtdt_entry = [(traj_id,float(trj[0][1]),float(trj[0][2]),float(trj[0][3]),float(trj[0][4]),float(trj[-1][3]),traj_manv,traj_mode,ag_type,init_time,parent_traj_id)]
                        traj_metadata.extend(traj_mtdt_entry)
                        trajs.extend(traj_entry)
                        traj_id += 1
            c.executemany(i_string,trajs)
            c.executemany(i_string_tj_mtdata,traj_metadata)
            print('agent trajectories inserted')           
        
        
        conn.commit()
        conn.close()
        
    
    def build_final_trajectories(self,maneuver_constraints):
        time_interval_axes = [[(0,int(1/self.freq))], [(0,int(2*int(1/self.freq))),(int(1/self.freq),int(1/self.freq))]]
        #time_interval_axes = [[(0,int(2*int(1/self.freq)))]]
        for time_intervals in time_interval_axes:
            state_lattics = self.get_current_states(time_intervals,maneuver_constraints)
            for ts,ts_v in state_lattics.items():
                for ag,ag_v in ts_v.items():
                    for manv,manv_v in ag_v.items():
                        ct,N = 0,len(manv_v)
                        for init_st in manv_v:
                            ct += 1
                            v = init_st[3]
                            parent_traj_arcl = init_st[5]
                            waypt,waypt_vel = self.construct_centerline(init_st[5],ag,v,maneuver_constraints,(init_st[0],init_st[1]))
                            path = maneuver_constraints[ag]['agent_state'].waypoints
                            
                            '''
                            plt.figure()
                            plt.plot([x[0] for x in path],[x[1] for x in path],color='blue',marker='x')
                            plt.plot([x[0] for x in waypt],[x[1] for x in waypt],color='red',marker='o')
                            plt.plot([init_st[0]],[init_st[1]],color='black',marker='s')
                            plt.show()
                            '''
                            assert len(waypt) == len(waypt_vel)
                            act = Actions(maneuver_constraints)
                            generating_manv_l = [x for x in maneuver_constraints[ag]['maneuvers'].keys() if x != manv]
                            for generating_manv in generating_manv_l:
                                step_horizon = self.horizon - ts
                                parent_traj_id = init_st[4]
                                '''
                                if ag == 'agent_2' and ts == 4:
                                    plt.figure()
                                    plt.title('path')
                                    plt.plot([x[0] for x in waypt], [x[1] for x in waypt],marker='x',color='red')
                                    v = waypt
                                    plt.arrow(v[0][0], v[0][1],v[1][0]-v[0][0] , v[1][1]-v[0][1], width=.25,color='green')
                                    v = path
                                    plt.arrow(v[-2][0], v[-2][1],v[-1][0]-v[-2][0] , v[-1][1]-v[-2][1], width=.25,color='black')
                                    plt.plot([x[0] for x in path],[x[1] for x in path],color='blue',marker='o')
                                    plt.axis('equal')
                                    plt.show()
                                '''
                                trajs = act.generate_agent_action(ts, v, waypt, waypt_vel, generating_manv, ag, step_horizon, parent_traj_arcl)
                                if self.initialize_db:
                                    self.insert_trajs_into_db(trajs, ag, ts, parent_traj_id)
                                print('generating',ag,'time',ts,'manv',generating_manv,ct,'/',N)
                                print(' '.join([str(_k)+':'+str(len(_v)) for _k,_v in trajs[generating_manv].items()]))
                
       
    def build_complete_tree(self,maneuver_constraints):
        self.build_initial_reachability_states(maneuver_constraints)
        self.build_final_trajectories(maneuver_constraints)
        
    def __init__(self,freq,initialize_db):
        self.initialize_db = initialize_db
        self.freq = 0.5 if freq is None else freq
        self.horizon = int(3/self.freq)




        
    
class Node:
    
    progress_ctr = 0
    tree_size = 0
    
    def print_Node(self,last_decision_level,results):
        node_result = {'uspe':[],'mspe':[],'qlk':{'ag1':[],'ag2':[]},'ag1_ac':None,'ag1_nac':None,'ag2_ac':None,'ag2_nac':None,'ag1_robust':[],'ag2_robust':[],'ag1_auto_resp':[],'ag2_auto_resp':[]}
        util_residuals = {'uspe':[],'mspe':[],'qlk':{'ag1':[],'ag2':[]},'ag1_ac':None,'ag1_nac':None,'ag2_ac':None,'ag2_nac':None,'ag1_robust':[],'ag2_robust':[],'ag1_auto_resp':[],'ag2_auto_resp':[]}
        if self.level == last_decision_level:
            if hasattr(self, 'emp_path') and self.emp_path:
                '''
                for n in self.children:
                    if hasattr(n, 'emp_path') and n.emp_path:
                        next_emp_node = n
                        break
                '''
                ag1_emp_trajl = self.path_from_root['agent_1'].get_last().length
                ag2_emp_trajl = self.path_from_root['agent_2'].get_last().length
                if hasattr(self, 'on_mspe') and np.any(self.on_mspe):
                    for i in np.arange(self.on_mspe.shape[0]):
                        for j in np.arange(self.on_mspe.shape[1]):
                            if self.on_mspe[i,j]:
                                node_result['mspe'].append((i,j))
                if hasattr(self, 'on_uspe') and np.any(self.on_uspe):
                    for i in np.arange(self.on_uspe.shape[0]):
                        for j in np.arange(self.on_uspe.shape[1]):
                            if self.on_uspe[i,j]:
                                node_result['uspe'].append((i,j))
                ag1_utilsdiff = np.ndarray(shape=self._parent.equilibrium_solutions.shape,dtype=float)
                ag2_utilsdiff = np.ndarray(shape=self._parent.equilibrium_solutions.shape,dtype=float)
                for i in np.arange(self._parent.equilibrium_solutions.shape[0]):
                    for j in np.arange(self._parent.equilibrium_solutions.shape[1]):
                        ag1_eq_acts = [x.veh_eq_acts[0] for x in self._parent.equilibrium_solutions[i,j]]
                        ag2_eq_acts = [x.peds_eq_acts[0] for x in self._parent.equilibrium_solutions[i,j]]
                        this_ag1_utils = [self._parent.util_info['spe'][(ag1_emp_trajl,x)][0][i,j] for x in ag2_eq_acts]
                        this_ag2_utils = [self._parent.util_info['spe'][(x,ag2_emp_trajl)][1][i,j] for x in ag1_eq_acts]
                        ag1_utilsdiff[i,j] = min([_resd for _resd in [x-y for x,y in itertools.product([self._parent.equilibrium_solutions[i,j][_eqidx].veh_br_map[x][0][i,j]['utils'] for _eqidx,x in enumerate(ag2_eq_acts)],this_ag1_utils)] if _resd >=0 ])
                        ag2_utilsdiff[i,j] = min([_resd for _resd in [x-y for x,y in itertools.product([self._parent.equilibrium_solutions[i,j][_eqidx].peds_br_map[x][0][i,j]['utils'] for _eqidx,x in enumerate(ag1_eq_acts)],this_ag2_utils)] if _resd >=0 ])
                        if ag1_utilsdiff[i,j] < 0 or ag2_utilsdiff[i,j] < 0:
                            brk = 1
                util_residuals['uspe'].append((ag1_utilsdiff,ag2_utilsdiff))
                util_residuals['mspe'].append((ag1_utilsdiff,ag2_utilsdiff))
                        
                
                if hasattr(self, 'automata_strategy_info'):
                    node_result['ag1_ac'] = self.automata_strategy_info['agent_1']['ac_auto_gamma'],
                    node_result['ag1_nac'] = self.automata_strategy_info['agent_1']['nac_auto_gamma']
                    node_result['ag2_ac'] = self.automata_strategy_info['agent_2']['ac_auto_gamma']
                    node_result['ag2_nac'] = self.automata_strategy_info['agent_2']['nac_auto_gamma']
                if hasattr(self.parent, 'auto_strategy_response'):
                    ag1_resp = self.parent.auto_strategy_response['agent_1'][0][:,0]
                    ag1_resp_min = self.parent.auto_strategy_response['agent_1'][1][:,0]
                    for i,resp in enumerate(ag1_resp):
                        if  min(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']) <= ag1_emp_trajl <= max(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']):
                            node_result['ag1_auto_resp'].append(i)
                        #util_residuals['ag1_auto_resp'].append((ag1_resp[i]['utils']-,))
                    ag1_emp_utils = None
                    ag1_all_trajls = self.parent.auto_strategy_response['agent_1_all_responses'][:,0,0]['traj_l']
                    _this_trajl_index, = np.where(np.isclose(ag1_all_trajls, ag1_emp_trajl))
                    ag1_emp_utils = self.parent.auto_strategy_response['agent_1_all_responses'][_this_trajl_index,:,0]['utils'] if _this_trajl_index.shape[0] > 0 else None
                    ag1_utilsdiff_autoresp = ag1_resp['utils'] - ag1_emp_utils if ag1_emp_utils is not None else None
                    ag2_resp = self.parent.auto_strategy_response['agent_2'][0][0,:]
                    ag2_resp_min = self.parent.auto_strategy_response['agent_2'][1][0,:]
                    for i,resp in enumerate(ag2_resp):
                        if  min(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']) <= ag2_emp_trajl <= max(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']):
                            node_result['ag2_auto_resp'].append(i)
                    ag2_emp_utils = None
                    ag2_all_trajls = self.parent.auto_strategy_response['agent_2_all_responses'][:,0,0]['traj_l']
                    _this_trajl_index, = np.where(np.isclose(ag2_all_trajls, ag2_emp_trajl))
                    ag2_emp_utils = self.parent.auto_strategy_response['agent_2_all_responses'][_this_trajl_index,0,:]['utils'] if _this_trajl_index.shape[0] > 0 else None
                    ag2_utilsdiff_autoresp = ag2_resp['utils'] - ag2_emp_utils if ag2_emp_utils is not None else None
                    util_residuals['ag1_auto_resp'].append(ag1_utilsdiff_autoresp)
                    util_residuals['ag2_auto_resp'].append(ag2_utilsdiff_autoresp)
                     
                if hasattr(self.parent, 'robust_response'):
                    ag1_resp = self.parent.robust_response['agent_1'][:,0]
                    for i,resp in enumerate(ag1_resp):
                        if  min(ag1_resp[i].veh_eq_acts) <= ag1_emp_trajl <= max(ag1_resp[i].veh_eq_acts):
                            node_result['ag1_robust'].append(i)
                    ag2_resp = self.parent.robust_response['agent_2'][:,0]
                    for i,resp in enumerate(ag2_resp):
                        if  min(ag2_resp[i].peds_eq_acts) <= ag2_emp_trajl <= max(ag2_resp[i].peds_eq_acts):
                            node_result['ag2_robust'].append(i) 
                    if hasattr(self.parent, 'robust_response_type'):
                        if self.parent.robust_response_type['agent_1'] == 'auto':
                            util_residuals['ag1_robust'].append(util_residuals['ag1_auto_resp'][-1])
                        elif self.parent.robust_response_type['agent_1'] == 'spe':
                            util_residuals['ag1_robust'].append(util_residuals['mspe'][-1][0])
                        else:
                            util_residuals['ag1_robust'].append(None)
                        if self.parent.robust_response_type['agent_2'] == 'auto':
                            util_residuals['ag2_robust'].append(util_residuals['ag2_auto_resp'][-1])
                        elif self.parent.robust_response_type['agent_2'] == 'spe':
                            util_residuals['ag2_robust'].append(util_residuals['mspe'][-1][1])   
                        else:
                            util_residuals['ag2_robust'].append(None)
                            
                if hasattr(self.parent, 'ql1_response'):
                    ag1_resp = self.parent.ql1_response['response']['agent_1'][:,0]
                    for i,resp in enumerate(ag1_resp):
                        if resp['traj_l'] == ag1_emp_trajl:
                            node_result['qlk']['ag1'].append((i,1))
                        else:
                            _prob = self.parent.ql1_response['distribution']['agent_1'][ag1_emp_trajl][i]
                            node_result['qlk']['ag1'].append((i,_prob))
                    ag1_emp_utils = None
                    ag1_all_trajls = self.parent.ql1_response['all_responses']['agent_1'][:,0,0]['traj_l']
                    _this_trajl_index, = np.where(np.isclose(ag1_all_trajls, ag1_emp_trajl))
                    ag1_emp_utils = self.parent.ql1_response['all_responses']['agent_1'][_this_trajl_index,:,0]['utils'] if _this_trajl_index.shape[0] > 0 else None
                    ag1_utilsdiff_qlk = ag1_resp['utils'] - ag1_emp_utils if ag1_emp_utils is not None else None      
                    
                    ag2_resp = self.parent.ql1_response['response']['agent_2'][:,0]
                    for i,resp in enumerate(ag2_resp):
                        if resp['traj_l'] == ag2_emp_trajl:
                            node_result['qlk']['ag2'].append((i,1))
                        else:
                            _prob = self.parent.ql1_response['distribution']['agent_2'][ag2_emp_trajl][i]
                            node_result['qlk']['ag2'].append((i,_prob))
                    ag2_emp_utils = None
                    ag2_all_trajls = self.parent.ql1_response['all_responses']['agent_2'][:,0,0]['traj_l']
                    _this_trajl_index, = np.where(np.isclose(ag2_all_trajls, ag2_emp_trajl))
                    ag2_emp_utils = self.parent.auto_strategy_response['agent_2_all_responses'][_this_trajl_index,0,:]['utils'] if _this_trajl_index.shape[0] > 0 else None
                    ag2_utilsdiff_qlk = ag2_resp['utils'] - ag2_emp_utils if ag2_emp_utils is not None else None      
                    util_residuals['qlk']['ag1'].append(ag1_utilsdiff_qlk)
                    util_residuals['qlk']['ag2'].append(ag2_utilsdiff_qlk)
                    
                print_str = [str(self.level)]
                for k,v in node_result.items():
                    print_str.append(k+':'+str(v))
                print(' '.join(print_str))
                if self.level not in results:
                    results[self.level] = []
                results[self.level].append({'node_result':node_result,'node':self, 'util_residuals':util_residuals})
                    
        else:
            if not self.is_leaf: 
                for c in self.children:
                    c.print_Node(last_decision_level,results)
                if not self.is_root:
                    if hasattr(self, 'emp_path') and self.emp_path:
                        ag1_emp_trajl = self.path_from_root['agent_1'].get_last().length
                        ag2_emp_trajl = self.path_from_root['agent_2'].get_last().length
                        if hasattr(self, 'on_mspe') and np.any(self.on_mspe):
                            for i in np.arange(self.on_mspe.shape[0]):
                                for j in np.arange(self.on_mspe.shape[1]):
                                    if self.on_mspe[i,j]:
                                        node_result['mspe'].append((i,j))
                        if hasattr(self, 'on_uspe') and np.any(self.on_uspe):
                            for i in np.arange(self.on_uspe.shape[0]):
                                for j in np.arange(self.on_uspe.shape[1]):
                                    if self.on_uspe[i,j]:
                                        node_result['uspe'].append((i,j))
                    
                        if self.automata_strategy_info is not None:
                            node_result['ag1_ac'] = self.automata_strategy_info['agent_1']['ac_auto_gamma'],
                            node_result['ag1_nac'] = self.automata_strategy_info['agent_1']['nac_auto_gamma']
                            node_result['ag2_ac'] = self.automata_strategy_info['agent_2']['ac_auto_gamma']
                            node_result['ag2_nac'] = self.automata_strategy_info['agent_2']['nac_auto_gamma']
                        if hasattr(self.parent, 'auto_strategy_response'):
                            ag1_resp = self.parent.auto_strategy_response['agent_1'][0][:,0]
                            ag1_resp_min = self.parent.auto_strategy_response['agent_1'][1][:,0]
                            for i,resp in enumerate(ag1_resp):
                                if  min(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']) <= ag1_emp_trajl <= max(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']):
                                    node_result['ag1_auto_resp'].append(i)
                            ag2_resp = self.parent.auto_strategy_response['agent_2'][0][0,:]
                            ag2_resp_min = self.parent.auto_strategy_response['agent_2'][1][0,:]
                            for i,resp in enumerate(ag2_resp):
                                if  min(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']) <= ag2_emp_trajl <= max(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']):
                                    node_result['ag2_auto_resp'].append(i) 
                        if hasattr(self.parent, 'robust_response'):
                            ag1_resp = self.parent.robust_response['agent_1'][:,0]
                            for i,resp in enumerate(ag1_resp):
                                if  min(ag1_resp[i].veh_eq_acts) <= ag1_emp_trajl <= max(ag1_resp[i].veh_eq_acts):
                                    node_result['ag1_robust'].append(i)
                            ag2_resp = self.parent.robust_response['agent_2'][:,0]
                            for i,resp in enumerate(ag2_resp):
                                if  min(ag2_resp[i].peds_eq_acts) <= ag2_emp_trajl <= max(ag2_resp[i].peds_eq_acts):
                                    node_result['ag2_robust'].append(i) 
                        if hasattr(self.parent, 'ql1_response'):
                            ag1_resp = self.parent.ql1_response['response']['agent_1'][:,0]
                            for i,resp in enumerate(ag1_resp):
                                if resp['traj_l'] == ag1_emp_trajl:
                                    node_result['qlk']['ag1'].append((i,1))
                                else:
                                    _prob = self.parent.ql1_response['distribution']['agent_1'][ag1_emp_trajl][i]
                                    node_result['qlk']['ag1'].append((i,_prob))
                                    
                            ag2_resp = self.parent.ql1_response['response']['agent_2'][:,0]
                            for i,resp in enumerate(ag2_resp):
                                if resp['traj_l'] == ag2_emp_trajl:
                                    node_result['qlk']['ag2'].append((i,1))
                                else:
                                    _prob = self.parent.ql1_response['distribution']['agent_2'][ag2_emp_trajl][i]
                                    node_result['qlk']['ag2'].append((i,_prob))
                            
                
                        print_str = [str(self.level)]
                        for k,v in node_result.items():
                            print_str.append(k+':'+str(v))
                        print(' '.join(print_str))  
                        if self.level not in results:
                            results[self.level] = []
                        results[self.level].append({'node_result':node_result,'node':self})
        
    def __init__(self,level, path_from_root,_ext_id,tree_link):
        self.level = level
        self.path_from_root = path_from_root
        self._tree_link = tree_link
        self._ext_id = _ext_id
        self.freq = self._tree_link.freq
    
    def load(self,file_id, traj_cache = None):
        if traj_cache is None:
            traj_cache = dict()
            for x in [(0,0,int(1/self.freq)),(0,int(1/self.freq),int(2*int(1/self.freq))),(int(1/self.freq),int(1/self.freq),int(2*int(1/self.freq)))]:
                traj_cache[x] = TrajectoryCache(init_time=x[0],time_range=(x[1],x[2]),ag_type=None,file_id=file_id)
        
        if not self.is_root:
            for p in self.path_from_root.values():
                p.load(file_id,traj_cache)
        if self.children is not None:
            for n in self.children:
                n.load(file_id,traj_cache)
        Node.progress_ctr += 1
        #print('loaded node',Node.progress_ctr,'/',Node.tree_size)
        '''
        if self.is_leaf:
            for p in self.actions['vehicle']:
                p.load()
            for p in self.actions['pedestrian']:
                p.load()
        ''' 
    def set_actions(self):
        actions = {'agent_1':[],'agent_2':[]}
        ''' get the children nodes and their path from root.
            actions are the last trajectory fragment of that path
        '''
        if self.is_leaf:
            veh_path_from_root = self.path_from_root['agent_1']
            peds_path_from_root = self.path_from_root['agent_2']
            v_tf = veh_path_from_root.get_last()
            p_tf = peds_path_from_root.get_last()
            self.actions = None
            v_tf._next_node = None
            p_tf._next_node = None
                
        else:
            for cidx,c in enumerate(self.children):
                c.set_actions()
                veh_path_from_root = c.path_from_root['agent_1']
                peds_path_from_root = c.path_from_root['agent_2']
                v_tf = copy.copy(veh_path_from_root.get_last())
                p_tf = copy.copy(peds_path_from_root.get_last())
                v_tf._next_node = c
                p_tf._next_node = c
                actions['agent_1'].append(v_tf)
                actions['agent_2'].append(p_tf)
            self.actions = actions
            '''
            assert len(actions['vehicle']) == len(self.children)
            assert len(actions['pedestrian']) == len(self.children)
            for i in np.arange(len(self.children)):
                assert self.children[i] is self.actions['vehicle'][i]._next_node, i
                assert self.children[i] is self.actions['pedestrian'][i]._next_node, i
            '''
                    
    @property
    def is_root(self):
        return True if self.path_from_root is None else False
    
    @property
    def is_leaf(self):
        return True if self.children is None else False
    
    @property
    def children(self):
        return self._children
    
    @property
    def parent(self):
        return self._parent
    
    @children.setter
    def children(self, value):
        self._children = value
        
    @parent.setter
    def parent(self, value):
        self._parent = value
    
    
    def size(self,last_decision_level):
        if self.level == last_decision_level:
            return 1
        else:
            if not self.is_leaf: 
                nc = 0
                for c in self.children:
                    nc += c.size(last_decision_level)
                return nc
    
    def assign_predicted_distgaps(self,model):
        agent_1_trajl = [self.path_from_root['agent_1'].length]
        agent_2_trajl = [self.path_from_root['agent_2'].length]
        
    
class GameTree:
    
    counter = 1
    
    def __init__(self,file_id,freq=None):
        self.file_id = file_id
        self.freq = freq if freq is not None else 0.5
        self.horizon = int(3/self.freq)
        
    
      
        
    
        
    def _process_three_change_tree(self,level_nodes_2s,level_nodes_4s,level_nodes_6s):
        v_tcache_4_6, p_tcache_4_6 = dict(), dict()
        v_tcache_4_6[(int(2*int(1/self.freq)),int(2*int(1/self.freq)),self.horizon)] = TrajectoryCache(init_time=int(2*int(1/self.freq)),time_range=(int(2*int(1/self.freq)),self.horizon),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6[(int(2*int(1/self.freq)),int(2*int(1/self.freq)),self.horizon)] = TrajectoryCache(init_time=int(2*int(1/self.freq)),time_range=(int(2*int(1/self.freq)),self.horizon),ag_type='agent_2',file_id=self.file_id)
        v_tcache_4_6[(int(1/self.freq),int(2*int(1/self.freq)),self.horizon)] = TrajectoryCache(init_time=int(1/self.freq),time_range=(int(2*int(1/self.freq)),self.horizon),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6[(int(1/self.freq),int(2*int(1/self.freq)),self.horizon)] = TrajectoryCache(init_time=int(1/self.freq),time_range=(int(2*int(1/self.freq)),self.horizon),ag_type='agent_2',file_id=self.file_id)
        v_tcache_4_6[(0,int(2*int(1/self.freq)),self.horizon)] = TrajectoryCache(init_time=0,time_range=(int(2*int(1/self.freq)),self.horizon),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6[(0,int(2*int(1/self.freq)),self.horizon)] = TrajectoryCache(init_time=0,time_range=(int(2*int(1/self.freq)),self.horizon),ag_type='agent_2',file_id=self.file_id)
        _ext_id = GameTree.counter
        self.root = Node(0,None,_ext_id,self)
        self.root.parent = None
        GameTree.counter += 1
        
        for s_path,n in level_nodes_2s.items():
            n_children = []
            n.parent = self.root
            for n4s,n4 in level_nodes_4s.items():
                if s_path[0][0] == n4s[0][0] and s_path[1][0] == n4s[1][0]:
                    
                    level_nodes_4s[n4s].children = []
                    level_nodes_4s[n4s].parent = n
                    n_children.append(level_nodes_4s[n4s])
                    if n4s[0][-1] in level_nodes_6s['agent_1'] and n4s[1][-1] in level_nodes_6s['agent_2']:
                        _children_info = list(itertools.product(level_nodes_6s['agent_1'][n4s[0][-1]], level_nodes_6s['agent_2'][n4s[1][-1]]))
                        n6 = []
                        for _c in _children_info:
                            vtf = TrajectoryFragment(time_range=(int(2*int(1/self.freq)),self.horizon),traj_id=_c[0][-2],manv=_c[0][0],manv_mode=_c[0][1],init_time=_c[0][-3],horizon=self.horizon)
                            ptf = TrajectoryFragment(time_range=(int(2*int(1/self.freq)),self.horizon),traj_id=_c[1][-2],manv=_c[1][0],manv_mode=_c[1][1],init_time=_c[1][-3],horizon=self.horizon)
                            vtf.load(self.file_id,v_tcache_4_6)
                            ptf.load(self.file_id,p_tcache_4_6)
                            _newvtf = copy.deepcopy(level_nodes_4s[n4s].path_from_root['agent_1'])
                            _newvtf.is_last = False
                            _newvtf.get_last().next_fragment = vtf
                            _newptf = copy.deepcopy(level_nodes_4s[n4s].path_from_root['agent_2'])
                            _newptf.is_last = False
                            _newptf.get_last().next_fragment = ptf
                            _ext_id = GameTree.counter
                            nd = Node(self.horizon,{'agent_1':_newvtf, 'agent_2':_newptf},_ext_id,self)
                            GameTree.counter += 1
                            nd.parent = level_nodes_4s[n4s]
                            n6.append(nd)
                        for _n in n6:
                            _n.children = None
                        
                        level_nodes_4s[n4s].children += n6
                    if len(level_nodes_4s[n4s].children) == 0:
                        level_nodes_4s[n4s].children = None
            n.children = n_children
            
            
                
        self.root.children = list(level_nodes_2s.values())
        
        
        
    def build_tree(self,maneuver_constraints):
        print('building lattice nodes...')
        self.maneuver_constraints = maneuver_constraints
        start_time = time.time()
        level_nodes_2s = self.build_level_nodes(int(1/self.freq))
        level_nodes_4s = self.build_level_nodes(int(2*int(1/self.freq)))
        level_nodes_6s = self.build_level_nodes(self.horizon)
        if len(level_nodes_2s) == 0 or len(level_nodes_4s) == 0 or len(level_nodes_6s['agent_1'])==0 or len(level_nodes_6s['agent_2'])==0:
            raise UnsupportedLatticeException(l2_size=len(level_nodes_2s), l4_size=len(level_nodes_4s), l6_size=(len(level_nodes_6s['agent_1']),len(level_nodes_6s['agent_2'])))
        self.last_decision_level = int(2*int(1/self.freq))
        self._process_three_change_tree(level_nodes_2s, level_nodes_4s, level_nodes_6s)
        '''
        if len(level_nodes_2s) == 0 and len(level_nodes_4s) != 0:
            self.last_decision_level = 4
            self._process_two_change_tree(level_nodes_4s, 4, level_nodes_6s)
        elif len(level_nodes_2s) != 0 and len(level_nodes_4s) == 0:
            self.last_decision_level = 2
            self._process_two_change_tree(level_nodes_2s, 2, level_nodes_6s)
        elif len(level_nodes_2s) != 0 and len(level_nodes_4s) != 0:
            self.last_decision_level = 4
            self._process_three_change_tree(level_nodes_2s, level_nodes_4s, level_nodes_6s)
        else:
            self.last_decision_level = 0
        '''
        print('N (2s):',len(level_nodes_2s), 'N (4s):',len(level_nodes_4s), 'N (6s):',len(level_nodes_6s['agent_1'])*len(level_nodes_6s['agent_2']))
        Node.tree_size = len(level_nodes_2s) * len(level_nodes_4s)
        print('building lattice nodes....DONE','(%s secs)' % (time.time() - start_time),)
        print('loading tree...')
        start_time = time.time()
        self.root.load(self.file_id)
        print('loading tree....DONE','(%s secs)' % (time.time() - start_time),)
        print('setting actions...')
        start_time = time.time()
        self.root.set_actions()
        print('setting actions....DONE','(%s secs)' % (time.time() - start_time),)
        f=1 
            
    
    def solve(self,eq_class):
        eq_class.solve(node = self.root,last_decision_level=self.last_decision_level)
        if hasattr(eq_class, 'set_oneq_label'):
            eq_class.set_oneq_label(self,'on_mspe')
            eq_class.set_oneq_label(self,'on_uspe')
    
    def print_tree(self):
        if not hasattr(self, 'results'):
            self.results = dict()
        self.root.print_Node(6,self.results)
        
    def animate(self,soln_type):
        analytics_obj = UniWeberAnalytics(constants.CURRENT_FILE_ID)
        done = []
        all_vels = [[],[]]
        if soln_type == 'mspe':
            all_nodes = get_all_level_nodes(node=self.root,node_list=[],tree_level=self.horizon)
            N = len(all_nodes)
            for ct,node in enumerate(all_nodes):
                for i in np.arange(5):
                    for j in np.arange(5):
                        print(ct,N)
                        if ct == 281:
                            f=1
                        if hasattr(node, 'on_mspe') and node.on_mspe[i,j]:
                            parent_node = node.parent
                            if hasattr(parent_node, 'on_mspe') and parent_node.on_mspe[i,j]:
                                grandparent_node = parent_node.parent
                                if hasattr(grandparent_node, 'on_mspe') and grandparent_node.on_mspe[i,j]:
                                    if hasattr(node, 'animation_done'):
                                        continue
                                    all_trajs = [[],[]]
                                    all_trajs[0] = [(x[1],x[2]) for x in node.path_from_root['agent_1'].loaded_traj]
                                    all_trajs[1] = [(x[1],x[2]) for x in node.path_from_root['agent_2'].loaded_traj]
                                    all_vels[0] = [(x[6],x[3]) for x in node.path_from_root['agent_1'].loaded_traj]
                                    all_vels[1] = [(x[6],x[3]) for x in node.path_from_root['agent_2'].loaded_traj]
                                    print('----')
                                    analytics_obj.animate_scene(all_trajs)
                                    done.append((i,j))
                node.animation_done = True
                            
                    #break
        fig, axs = plt.subplots(2)
        fig.suptitle('velocities')
        axs[0].plot(np.arange(len(all_vels[0])), [x[1] for x in all_vels[0]])
        axs[1].plot(np.arange(len(all_vels[1])), [x[1] for x in all_vels[1]])
        plt.show()
        
    
    
    def build_level_nodes(self, level):
        conn = sqlite3.connect(rg_constants.get_rg_db_path(self.file_id))
        c = conn.cursor()
        all_level_nodes = dict()
            
        def _getqstring(ag_type):
            start_time,one_step_t,two_step_t,three_step_t = 0,int(1/self.freq),int(2*int(1/self.freq)),self.horizon
            return "select MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X)*ABS(INIT_POS_X-X)+ABS(INIT_POS_Y-Y)*ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.INIT_TIME,TRAJECTORY_METADATA.TRAJ_ID,TRAJECTORY_METADATA.PARENT_TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME="+str(two_step_t)+" AND TRAJECTORIES.TIME="+str(one_step_t)+" \
                        AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' \
                        UNION \
                        select MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X)*ABS(INIT_POS_X-X)+ABS(INIT_POS_Y-Y)*ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.INIT_TIME,TRAJECTORY_METADATA.TRAJ_ID,TRAJECTORY_METADATA.PARENT_TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME="+str(one_step_t)+" AND TRAJECTORIES.TIME="+str(two_step_t)+" \
                        AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.TRAJ_ID IN (select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.INIT_TIME="+str(two_step_t)+") \
                        UNION \
                        select MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X)*ABS(INIT_POS_X-X)+ABS(INIT_POS_Y-Y)*ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.INIT_TIME,TRAJECTORY_METADATA.TRAJ_ID,TRAJECTORY_METADATA.PARENT_TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME=0 AND TRAJECTORIES.TIME="+str(three_step_t)+" AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.TRAJ_ID IN (select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.INIT_TIME="+str(one_step_t)+")"
        if level == self.horizon:
            c.execute(_getqstring('agent_1'))
            veh_res = c.fetchall()
            c.execute(_getqstring('agent_2'))
            peds_res = c.fetchall()
            #veh_res = np.array([(x[0],x[1],x[2],LA.norm([x[3],x[4]]),LA.norm([x[3],x[4]]),x[5],x[6],x[7],x[8],x[9],x[10]) for x in veh_res])
            veh_lattice_states,latc_tracker_veh = dict(),dict()
            #peds_res = np.array([(x[0],x[1],x[2],LA.norm([x[3],x[4]]),LA.norm([x[3],x[4]]),x[5],x[6],x[7],x[8],x[9],x[10]) for x in peds_res])
            peds_lattice_states,latc_tracker_ped = dict(), dict()
            lattice_dist_step_ag1 = 0.5
            lattice_vel_step_ag1 = 0.3
            lattice_dist_step_ag2 = 5
            lattice_vel_step_ag2 = 1
                
            for idx1,veh_row in enumerate(veh_res):
                parent_traj_id = veh_row[-1] if veh_row[-1] is not None else veh_row[-2]
                if parent_traj_id not in veh_lattice_states:
                        veh_lattice_states[parent_traj_id] = []
                        
                if veh_row[-3] == int(2*int(1/self.freq)):
                    if parent_traj_id not in latc_tracker_veh or ( math.sqrt(veh_row[3])-min(latc_tracker_veh[parent_traj_id], key=lambda x:abs(x-math.sqrt(veh_row[3]))) > lattice_dist_step_ag1 ):
                        if parent_traj_id not in latc_tracker_veh:
                            latc_tracker_veh[parent_traj_id] = []
                        veh_lattice_states[parent_traj_id].append(tuple([x if _i !=3 else math.sqrt(x) for _i,x in enumerate(veh_row)]))
                        latc_tracker_veh[parent_traj_id].append(math.sqrt(veh_row[3]))
                else:
                    veh_lattice_states[parent_traj_id].append(tuple([x if _i !=3 else math.sqrt(x) for _i,x in enumerate(veh_row)]))
                    
            for idx2,peds_row in enumerate(peds_res):
                parent_traj_id = peds_row[-1] if peds_row[-1] is not None else peds_row[-2]
                if parent_traj_id not in peds_lattice_states:
                        peds_lattice_states[parent_traj_id] = []
                if peds_row[-3] == int(2*int(1/self.freq)):
                    if parent_traj_id not in latc_tracker_ped or ( math.sqrt(peds_row[3])-min(latc_tracker_ped[parent_traj_id], key=lambda x:abs(x-math.sqrt(peds_row[3]))) > lattice_dist_step_ag2 ):
                        if parent_traj_id not in latc_tracker_ped:
                            latc_tracker_ped[parent_traj_id] = []
                        
                        
                        peds_lattice_states[parent_traj_id].append(tuple([x if _i !=3 else math.sqrt(x) for _i,x in enumerate(peds_row)]))
                        latc_tracker_ped[parent_traj_id].append(math.sqrt(peds_row[3]))
                else:
                    peds_lattice_states[parent_traj_id].append(tuple([x if _i !=3 else math.sqrt(x) for _i,x in enumerate(peds_row)]))
            
            all_level_nodes['agent_1'] = veh_lattice_states
            all_level_nodes['agent_2'] = peds_lattice_states
        else:
            ''' one way to find the lattice nodes at level l is to just query for trajectories that offshoot (i.e. are the parents) trajectories from level l to the end.
            In other words, trajectories which are the parent trajectories for trajectories that start from level l'''
            
            q_string = "SELECT *  FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.TRAJ_ID IN ( \
                            select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+'agent_1'+"' AND TRAJECTORY_METADATA.INIT_TIME="+str(level)+");"
            c.execute(q_string)
            veh_res = c.fetchall()
            veh_traj_info = {row[0]:(row[6],row[7],row[9]) for row in veh_res}
            
            if level == int(2*int(1/self.freq)):
                ''' get the grandparent fragment info. Only level 4 has grandparent, and that starts from time 0 '''
                q_string = "SELECT *  FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.INIT_TIME=0 and TRAJECTORY_METADATA.AGENT_TYPE='"+'agent_1'+"'"
                c.execute(q_string)
                res = c.fetchall()
                veh_res = veh_res + res
                for row in res:
                    veh_traj_info[row[0]] = (row[6],row[7],row[9])
            
            
            q_string = "SELECT *  FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.TRAJ_ID IN ( \
                            select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+'agent_2'+"' AND TRAJECTORY_METADATA.INIT_TIME="+str(level)+");"
            c.execute(q_string)
            peds_res = c.fetchall()
            peds_traj_info = {row[0]:(row[6],row[7],row[9]) for row in peds_res}
            if level == int(2*int(1/self.freq)):
                ''' get the grandparent fragment info. Only level 4 has grandparent, and that starts from time 0 '''
                q_string = "SELECT *  FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.INIT_TIME=0 and TRAJECTORY_METADATA.AGENT_TYPE='"+'agent_2'+"'"
                c.execute(q_string)
                res = c.fetchall()
                peds_res = peds_res + res
                for row in res:
                    peds_traj_info[row[0]] = (row[6],row[7],row[9])
            
            
            ct,N = 0,len(veh_res)*len(peds_res)
            for idx1,veh_row in enumerate(veh_res):
                if veh_row[9] == 0:
                    vtf1 = TrajectoryFragment(time_range = (0,int(1/self.freq)) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2],horizon=self.horizon)
                    if level == int(2*int(1/self.freq)):
                        vtf2 = TrajectoryFragment(time_range = (int(1/self.freq),int(2*int(1/self.freq))) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2],horizon=self.horizon)
                        vtf2.is_last = True
                        vtf1.next_fragment = vtf2
                    else:
                        vtf1.is_last = True
                else:
                    ''' get the grandparent fragment '''
                    vtf1 = TrajectoryFragment(time_range = (0,int(1/self.freq)) ,traj_id = veh_row[10] ,manv = veh_traj_info[veh_row[10]][0],manv_mode = veh_traj_info[veh_row[10]][1],init_time = veh_traj_info[veh_row[10]][2],horizon=self.horizon)
                    if level == int(2*int(1/self.freq)):
                        vtf2 = TrajectoryFragment(time_range = (int(1/self.freq),int(2*int(1/self.freq))) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2],horizon=self.horizon)
                        vtf2.is_last = True
                        vtf1.next_fragment = vtf2
                    else:
                        vtf1.is_last = True
                for idx2,peds_row in enumerate(peds_res):
                    ct += 1
                    #print('level',level,ct,'/',N)
                    if peds_row[9] == 0:
                        ptf1 = TrajectoryFragment(time_range = (0,int(1/self.freq)) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2],horizon=self.horizon)
                        if level == int(2*int(1/self.freq)):
                            ptf2 = TrajectoryFragment(time_range = (int(1/self.freq),int(2*int(1/self.freq))) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2],horizon=self.horizon)
                            ptf2.is_last = True
                            ptf1.next_fragment = ptf2
                        else:
                            ptf1.is_last = True
                    else:
                        ptf1 = TrajectoryFragment(time_range = (0,int(1/self.freq)) ,traj_id = peds_row[10] ,manv = peds_traj_info[peds_row[10]][0],manv_mode = peds_traj_info[peds_row[10]][1],init_time = peds_traj_info[peds_row[10]][2],horizon=self.horizon)
                        if level == int(2*int(1/self.freq)):
                            ptf2 = TrajectoryFragment(time_range = (int(1/self.freq),int(2*int(1/self.freq))) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2],horizon=self.horizon)
                            ptf2.is_last = True
                            ptf1.next_fragment = ptf2
                        else:
                            ptf1.is_last = True
                    _ext_id = GameTree.counter
                    nd = Node(level,{'agent_1':vtf1, 'agent_2':ptf1}, _ext_id,self)
                    GameTree.counter += 1
                    #nd.load()
                    if level == int(2*int(1/self.freq)):
                        s_key = ((vtf1.traj_id,vtf2.traj_id),(ptf1.traj_id,ptf2.traj_id))
                        print(s_key,(vtf1.manv,vtf2.manv),(ptf1.manv,ptf2.manv))
                    else:
                        s_key = ((vtf1.traj_id,),(ptf1.traj_id,))
                        print(s_key,(vtf1.manv),(ptf1.manv))
                    all_level_nodes[s_key] = nd
                
        return all_level_nodes
    
def run_one_scenario(dbfile_id,agent1_id,agent2_id,start_ts,initialize_db,freq):
    scene_def = ScenarioDef(agent_1_id=agent1_id,agent_2_id=agent2_id,file_id=dbfile_id,initialize_db=initialize_db,start_ts=start_ts,freq=freq)
    maneuver_constraints = scene_def.setup_trajectory_constraints()
    tree_builder = TreeBuilder(freq,initialize_db)
    tree_builder.build_complete_tree(maneuver_constraints)
    
    file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
    gt = GameTree(file_id,freq)
    gt.build_tree(maneuver_constraints)
    type(gt.root).progress_ctr = 0
    type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
    m = MinDistanceGapModel(file_id,freq)
    m.build_model()   
    context = RunContext()
    context.gt_obj = gt
    manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'track_speed'}}
    context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True,'maneuver_constraints':maneuver_constraints})
    drassign_obj = AssignDistRanges()
    drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
    start_time = time.time()
    eq_obj = SatisficingEquilibria(context)
    gt.solve(eq_obj)
    print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(RobustResponse(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(Ql1Model(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    
    gt.scene_def = scene_def
    assign_emp_nodes(gt,gt.scene_def)
    gt.print_tree()
    #plot_velocity_profiles(gt,scene_def,freq)
    gt.animate('mspe')
    f=1


def animate_one_scenario(gt_file_id):
    gt = all_utils.utils.pickle_load(os.path.join(rg_constants.TREE_FILES,gt_file_id+'.gt'))
    gt.animate('mspe')
                


def run_all_scenarios():
    freq = 0.5
    initialize_db = True
    initialize_files = False
    rerun_failed_files = False
    failed_files = []
    scene_type = sys.argv[2]
    rg_constants.SCENE_TYPE = ('REAL',None)
    #inp_file_ids = [769,770,771,775,776]
    with open(rg_constants.FAILED_FILES_PATH,newline='\n') as csv_file:
        sc_reader = csv.reader(csv_file, delimiter=',')
        for row in sc_reader:
            dbfile_id = row[0]
            agent1_id = int(row[3])
            agent2_id = int(row[4])
            start_ts = float(row[5])
            failed_files.append((dbfile_id,agent1_id,agent2_id,start_ts))
            
    with open(rg_constants.SCENE_OUT_PATH,newline='\n') as csv_file:
        sc_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in sc_reader:
            inp_file_ids = ast.literal_eval(sys.argv[1])
            dbfile_id = row[0]
            row_sc_type = row[1]
            if int(dbfile_id) not in inp_file_ids:
                continue
            if scene_type is not None and scene_type != row_sc_type:
                print('row',row,'not the scene type',sys.argv[2],'...continuing')
                continue
            if not initialize_files:
                if os.path.isfile(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt')):
                    print('row',row,'processed...continuing')
                    continue
            
            
            agent1_id = int(row[3])
            agent2_id = int(row[4])
            start_ts = float(row[5])
            if not rerun_failed_files:
                if (dbfile_id,agent1_id,agent2_id,start_ts) in failed_files:
                    print('row',row,'had failed...continuing')
                    continue
            
            print('processing',dbfile_id,line_count+1,agent1_id,agent2_id)
            try:
                scene_def = ScenarioDef(agent_1_id=agent1_id,agent_2_id=agent2_id,file_id=dbfile_id,initialize_db=initialize_db,start_ts=start_ts,freq=freq)
                if scene_def.time_crossed:
                    continue
                maneuver_constraints = scene_def.setup_trajectory_constraints()
                if initialize_db:
                    tree_builder = TreeBuilder(freq,initialize_db)
                    tree_builder.build_complete_tree(maneuver_constraints)
                
                file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
                gt = GameTree(file_id,freq)
                gt.build_tree(maneuver_constraints)
                type(gt.root).progress_ctr = 0
                type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
                m = MinDistanceGapModel(file_id,freq)
                m.build_model()   
                context = RunContext()
                manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'track_speed'}}
                context.gt_obj = gt
                context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True,'maneuver_constraints':maneuver_constraints})
                drassign_obj = AssignDistRanges()
                drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
                start_time = time.time()
                eq_obj = SatisficingEquilibria(context)
                gt.solve(eq_obj)
                print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
                start_time = time.time()
                gt.solve(RobustResponse(context))
                print('solving robust. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
                start_time = time.time()
                gt.solve(Ql1Model(context))
                print('solving qlk. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    
                gt.scene_def = scene_def
                gt.maneuver_constraints = None
                pickle_dump_to_dir(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt'), gt)
                #pickle_dump_to_dir(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.scenedef'), scene_def)
                
            except Exception as e:
                    # Get current system exception
                ex_type, ex_value, ex_traceback = sys.exc_info()
            
                # Extract unformatter stack traces as tuples
                trace_back = traceback.extract_tb(ex_traceback)
            
                # Format stacktrace
                stack_trace = list()
            
                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                log.warn('caught and recorded exception in file')
                with open(rg_constants.FAILED_FILES_PATH, mode='a') as failed_file:
                    fail_writer = csv.writer(failed_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    msg = row + [str(ex_type.__name__),str(ex_value),str(stack_trace)]
                    fail_writer.writerow(msg)
                '''
                if not isinstance(e, UnsupportedLatticeException) and not isinstance(e, UnsupportedAgentObservationException) :
                    raise
                '''
            line_count += 1
            
            


    
def assign_emp_nodes(gt,scene_def):
    gt.root.emp_path = True
    l2_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=int(1/gt.freq))
    _path = []
    if not (len(scene_def.agent1_emp_traj)==0 or len(scene_def.agent2_emp_traj)==0):
        emp_2l = get_nearest_node(l2_nodes, scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[0])
        for n2l in l2_nodes:
            '''assign emp path value '''
            if n2l._ext_id == emp_2l[0]:
                n2l.emp_path = True
                _path.append(n2l)
                if len(scene_def.agent1_emp_traj) > 1 and n2l.children is not None:
                    emp_4l = get_nearest_node(n2l.children, scene_def.agent1_emp_traj[1]-scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[1]-scene_def.agent2_emp_traj[0])
                    for n4l in n2l.children:
                        if n4l._ext_id == emp_4l[0]:
                            n4l.emp_path = True
                            _path.append(n4l)
                            if len(scene_def.agent1_emp_traj) > 2 and n4l.children is not None:
                                emp_6l = get_nearest_node(n4l.children, scene_def.agent1_emp_traj[2]-scene_def.agent1_emp_traj[1], scene_def.agent2_emp_traj[2]-scene_def.agent2_emp_traj[1])
                                for n6l in n4l.children:
                                    if n6l._ext_id == emp_6l[0]:
                                        n6l.emp_path = True
                                        _path.append(n6l)
                                    else:
                                        n6l.emp_path = False
                        else:
                            n4l.emp_path = False
                f=1
            else:
                n2l.emp_path = False
            f=1
    print('empirical path assigned',str([(x.path_from_root['agent_1'].get_last().length,x.path_from_root['agent_2'].get_last().length) for x in _path]))
        

def results_all_scenarios():
    regenerate = True
    with open(rg_constants.SCENE_OUT_PATH,newline='\n') as csv_file:
        sc_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in sc_reader:
            if os.path.isfile(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt')):
                if not regenerate:
                    if os.path.isfile(os.path.join(rg_constants.RESULTS_FILES,'_'.join(row).replace('.',',')+'.results')):
                        print('row',row,'processed...continuing')
                        continue
                print('row',row,'processing..')
                dbfile_id = row[0]
                agent1_id = int(row[3])
                agent2_id = int(row[4])
                start_ts = float(row[5])
                #scene_def = ScenarioDef(agent_1_id=agent1_id,agent_2_id=agent2_id,file_id=dbfile_id,initialize_db=False,start_ts=start_ts)
                try:
                    gt = all_utils.utils.pickle_load(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt'))
                except EOFError:
                    print('row',row,'failed..continuing')
                    continue
                    
                assign_emp_nodes(gt,gt.scene_def)
                gt.print_tree()
                print('---------')
                pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'_'.join(row).replace('.',',')+'.results'), gt.results)
                line_count += 1
            

def plot_all_results():
    
    hit_ct = {'uspe':0,'mspe':0,'auto_resp':0,'ac':0,'nac':0,'robust':0,'no_exp.':0,'qlk':0}  
    range_var = {'auto_resp':[],'ac':[],'nac':[],'robust':[],'uspe':[],'mspe':[],'qlk':[]}
    pooling_map = {'auto_resp':OrderedDict(),'ac':OrderedDict(),'nac':OrderedDict(),'robust':OrderedDict(),'uspe':OrderedDict(),'mspe':OrderedDict(),'qlk':OrderedDict()}
    disagreement_map = {k:{'ag1':[],'ag2':[]} for k in itertools.product(pooling_map.keys(), pooling_map.keys())}
    line_count = 0
    resultfiles = [f for f in listdir(rg_constants.RESULTS_FILES) if isfile(join(rg_constants.RESULTS_FILES, f))]
    residual_freq_ct = {'ag1_auto_resp':OrderedDict(), 'ag1_robust_resp':OrderedDict(), 'ag1_spe':OrderedDict(), 'qlk': OrderedDict()}
    scene_type = 'rt'
    for resfile_name in resultfiles:
        this_scene_type = resfile_name.split('_')[1]
        if scene_type != this_scene_type:
            print('row',this_scene_type,'not the scene type',scene_type,'...continuing')
            continue
        line_count += 1
        print(line_count)
        #if line_count >= 30:
        #    break
        res_info = all_utils.utils.pickle_load(os.path.join(rg_constants.RESULTS_FILES,resfile_name))
        for l,rl in res_info.items():
            if l!= 6:
                continue
            for res in rl:
                node_res = res['node_result']
                util_residuals = res['util_residuals']
                no_expl = True
                if (node_res['ag1_ac'] is not False or node_res['ag1_nac'] is not False) and (node_res['ag2_ac'] is not False or node_res['ag2_nac'] is not False):
                    if node_res['ag1_ac'] is not False and  len(node_res['ag1_ac']) == 1:
                        node_res['ag1_ac'] = node_res['ag1_ac'][0]
                    if node_res['ag2_ac'] is not False and  len(node_res['ag2_ac']) == 1:
                        node_res['ag2_ac'] = node_res['ag2_ac'][0]
                    if node_res['ag1_nac'] is not False and len(node_res['ag1_nac']) == 1:
                        node_res['ag1_nac'] = node_res['ag1_nac'][0]
                    if node_res['ag2_nac'] is not False and len(node_res['ag2_nac']) == 1:
                        node_res['ag2_nac'] = node_res['ag2_nac'][0]
                        
                    if node_res['ag1_ac'] is not False:
                        hit_ct['ac'] += .5
                        range_var['ac'].append(len(np.arange(min(node_res['ag1_ac']), max(node_res['ag1_ac'])+.5,.5)))
                        type_list = np.arange(round(min(node_res['ag1_ac']),1), round(max(node_res['ag1_ac']),1)+.5,.5).tolist()
                        type_list.sort()
                        if tuple(type_list) not in pooling_map['ac']:
                            pooling_map['ac'][tuple(type_list)] = 1
                        else:
                            pooling_map['ac'][tuple(type_list)] += 1
                    else:
                        hit_ct['nac'] += .5
                        range_var['nac'].append(len(np.arange(min(node_res['ag1_nac']), max(node_res['ag1_nac'])+.5,.5)))
                        type_list = np.arange(round(min(node_res['ag1_nac']),1), round(max(node_res['ag1_nac']),1)+.5,.5).tolist()
                        type_list.sort()
                        if tuple(type_list) not in pooling_map['nac']:
                            pooling_map['nac'][tuple(type_list)] = 1
                        else:
                            pooling_map['nac'][tuple(type_list)] += 1
                    if node_res['ag2_ac'] is not False:
                        hit_ct['ac'] += .5
                        range_var['ac'].append(len(np.arange(min(node_res['ag2_ac']), max(node_res['ag2_ac'])+.5,.5)))
                        type_list = np.arange(round(min(node_res['ag2_ac']),1), round(max(node_res['ag2_ac']),1)+.5,.5).tolist()
                        type_list.sort()
                        if tuple(type_list) not in pooling_map['ac']:
                            pooling_map['ac'][tuple(type_list)] = 1
                        else:
                            pooling_map['ac'][tuple(type_list)] += 1
                    else:
                        hit_ct['nac'] += .5
                        range_var['nac'].append(len(np.arange(min(node_res['ag2_nac']), max(node_res['ag2_nac'])+.5,.5)))
                        type_list = np.arange(round(min(node_res['ag2_nac']),1), round(max(node_res['ag2_nac']),1)+.5,.5).tolist()
                        type_list.sort()
                        if tuple(type_list) not in pooling_map['nac']:
                            pooling_map['nac'][tuple(type_list)] = 1
                        else:
                            pooling_map['nac'][tuple(type_list)] += 1
                if len(node_res['ag1_auto_resp']) > 0:
                    hit_ct['auto_resp'] += 0.5
                    range_var['auto_resp'].append(len(node_res['ag1_auto_resp']))
                    type_list = node_res['ag1_auto_resp']
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['auto_resp']:
                        pooling_map['auto_resp'][tuple(type_list)] = 1
                    else:
                        pooling_map['auto_resp'][tuple(type_list)] += 1
                    
                if len(node_res['ag2_auto_resp']) > 0:
                    hit_ct['auto_resp'] += 0.5
                    range_var['auto_resp'].append(len(node_res['ag2_auto_resp']))
                    type_list = node_res['ag2_auto_resp']
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['auto_resp']:
                        pooling_map['auto_resp'][tuple(type_list)] = 1
                    else:
                        pooling_map['auto_resp'][tuple(type_list)] += 1
                    
                if len(node_res['ag1_robust']) > 0:
                    hit_ct['robust'] += 0.5
                    range_var['robust'].append(len(node_res['ag1_robust']))
                    type_list = node_res['ag1_robust']
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['robust']:
                        pooling_map['robust'][tuple(type_list)] = 1
                    else:
                        pooling_map['robust'][tuple(type_list)] += 1
                    
                    
                if len(node_res['ag2_robust']) > 0:
                    hit_ct['robust'] += 0.5
                    range_var['robust'].append(len(node_res['ag2_robust']))
                    type_list = node_res['ag2_robust']
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['robust']:
                        pooling_map['robust'][tuple(type_list)] = 1
                    else:
                        pooling_map['robust'][tuple(type_list)] += 1
                    
                if len(node_res['qlk']['ag1']) >0 and len(node_res['qlk']['ag2']) >0:
                    _ag1_br = [1 if x[1]==1 else 0 for x in node_res['qlk']['ag1']]
                    type_list = [x[0] for x in node_res['qlk']['ag1'] if x[1] == 1]
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if len(type_list) > 0:
                        if tuple(type_list) not in pooling_map['qlk']:
                            pooling_map['qlk'][tuple(type_list)] = 1
                        else:
                            pooling_map['qlk'][tuple(type_list)] += 1
                    type_list = [x[0] for x in node_res['qlk']['ag2'] if x[1] == 1]
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if len(type_list) > 0:
                        if tuple(type_list) not in pooling_map['qlk']:
                            pooling_map['qlk'][tuple(type_list)] = 1
                        else:
                            pooling_map['qlk'][tuple(type_list)] += 1
                        
                    _ag2_br = [1 if x[1]==1 else 0 for x in node_res['qlk']['ag2']]
                    if max(_ag1_br) == 1:
                        hit_ct['qlk'] += 0.5
                    if max(_ag2_br) == 1:
                        hit_ct['qlk'] += 0.5
                    #hit_ct['qlk'] += max(_ag1_br + [x[1] for x in node_res['qlk']['ag2']])
                    #range_var['qlk'].append(min(len(_ag1_br),len(node_res['qlk']['ag2'])))
                    range_var['qlk'].append(_ag1_br.count(1) + _ag2_br.count(1))
                    #pooling_map['qlk'] += [x for x in _ag1_br if x in node_res['qlk']['ag2']]
                    
                if len(node_res['mspe']) > 0:
                    hit_ct['mspe'] += 1
                    #print('mspe')
                    #print(list(set([x[0] for x in node_res['mspe']])))
                    #print(list(set([x[1] for x in node_res['mspe']])))
                    range_var['mspe'].append(len(list(set([x[0] for x in node_res['mspe']]))))
                    range_var['mspe'].append(len(list(set([x[1] for x in node_res['mspe']]))))
                    type_list = list(set([x[0] for x in node_res['mspe']]))
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['mspe']:
                        pooling_map['mspe'][tuple(type_list)] = 1
                    else:
                        pooling_map['mspe'][tuple(type_list)] += 1
                    type_list = list(set([x[1] for x in node_res['mspe']]))
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['mspe']:
                        pooling_map['mspe'][tuple(type_list)] = 1
                    else:
                        pooling_map['mspe'][tuple(type_list)] += 1
                    
                if len(node_res['uspe']) > 0:
                    hit_ct['uspe'] += 1
                    range_var['uspe'].append(len(list(set([x[0] for x in node_res['uspe']]))))
                    range_var['uspe'].append(len(list(set([x[1] for x in node_res['uspe']]))))
                    type_list = list(set([x[0] for x in node_res['uspe']]))
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['uspe']:
                        pooling_map['uspe'][tuple(type_list)] = 1
                    else:
                        pooling_map['uspe'][tuple(type_list)] += 1
                    type_list = list(set([x[1] for x in node_res['uspe']]))
                    type_list = rg_utils.to_type(type_list)
                    type_list.sort()
                    if tuple(type_list) not in pooling_map['uspe']:
                        pooling_map['uspe'][tuple(type_list)] = 1
                    else:
                        pooling_map['uspe'][tuple(type_list)] += 1
                    
                    #print('uspe')
                    #print(list(set([x[0] for x in node_res['uspe']])))
                    #print(list(set([x[1] for x in node_res['uspe']])))
                
                '''    
                if node_res['mspe'] is not False:
                    hit_ct['mspe'] += 1
                if node_res['uspe'] is not False:
                    hit_ct['uspe'] += 1
                '''
                if (node_res['ag1_ac'] is not False or node_res['ag1_nac'] is not False or len(node_res['ag1_auto_resp']) > 0 or len(node_res['ag1_robust']) > 0) and  \
                    (node_res['ag2_ac'] is not False or node_res['ag2_nac'] is not False or len(node_res['ag2_auto_resp']) > 0 or len(node_res['ag2_robust']) > 0):
                        no_expl = False
                if len(node_res['uspe']) > 0:
                    no_expl = False
                if len(node_res['mspe']) > 0:
                    no_expl = False
                
                if no_expl:
                    hit_ct['no_exp'] += 1
                    
                ''' residual distributions '''
                _e = util_residuals['ag1_auto_resp'][0][0,2]
                _e = round(_e, 2)
                if _e not in residual_freq_ct['ag1_auto_resp']:
                    residual_freq_ct['ag1_auto_resp'][_e] = 1
                else:
                    residual_freq_ct['ag1_auto_resp'][_e] += 1
                
                _e = util_residuals['ag1_robust'][0][0,2]
                _e = round(_e, 2)
                if _e not in residual_freq_ct['ag1_robust_resp']:
                    residual_freq_ct['ag1_robust_resp'][_e] = 1
                else:
                    residual_freq_ct['ag1_robust_resp'][_e] += 1
                    
                _e = util_residuals['mspe'][0][0][2,2]
                _e = round(_e, 2)
                if _e not in residual_freq_ct['ag1_spe']:
                    residual_freq_ct['ag1_spe'][_e] = 1
                else:
                    residual_freq_ct['ag1_spe'][_e] += 1
                    
                _e = util_residuals['qlk']['ag1'][0][0,2]
                _e = round(_e, 2)
                if _e not in residual_freq_ct['qlk']:
                    residual_freq_ct['qlk'][_e] = 1
                else:
                    residual_freq_ct['qlk'][_e] += 1
                
                
    plt.figure()
    plt.bar(np.arange(len(hit_ct)), list(hit_ct.values()), align='center', alpha=0.5)
    plt.xticks(np.arange(len(hit_ct)), list(hit_ct.keys()))
    for k,v in pooling_map.items():
        fig, axs = plt.subplots(2)
        #fig.suptitle('pooling map-'+k,y=1.12)
        ax = fig.gca()
        disp_scene_type = 'right_turn' if scene_type == 'rt' else 'left_turn'
        ax.set_title(k+'-'+disp_scene_type, pad=20)
        axs[0].bar(np.arange(len(v)), list(v.values()), align='center', alpha=0.5)
        axs[0].set_xticks(np.arange(len(v.keys())))
        axs[0].set_xticklabels(list(v.keys()), rotation=45)
        #axs[0].title.set_text(str(list(zip(v.keys(),v.values()))))
        x_key_arr = np.zeros(shape=(5,len(v)))
        key_arr_ord = []
        for _k1_idx,_k1 in enumerate(v.keys()):
            for tol_idx,tol in enumerate(np.arange(-1,1.5,0.5).tolist()):
                if tol in _k1:
                    #print(tol_idx,_k1_idx)
                    x_key_arr[tol_idx,_k1_idx] = tol+2
            key_arr_ord.append(_k1)
        rg_utils.plot_heatmap(plt, fig, axs[1], x_key_arr)
        #axs[1].title.set_text(str(key_arr_ord))
        fig.tight_layout()
    
    '''
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(list(range_var.values()))
    ax.set_yticklabels(list(range_var.keys()))
    '''
    
    '''
    for k,v in pooling_map.items():
        if k == 'auto_resp' or k == 'uspe' or k == 'mspe' or k == 'robust' or k=='qlk':
            v = [-1 + (x*0.5) for x in v]
        print(k,np.mean(v),np.std(v),np.min(v),np.max(v))
        
        plt.figure()
        plt.title(k)
        plt.hist(v,orientation='horizontal')
        plt.show()
         
        _x.append(np.mean(v))
        _sd.append(np.std(v))
    '''
    '''
    plt.figure()
    plt.title('threshold values')
    plt.errorbar(list(pooling_map.keys()), _x, err_range,linestyle='None', marker='^')
    '''
    '''
    for k,v in residual_freq_ct.items():
        plt.figure()
        plt.title('residuals - '+k)
        plt.scatter([x for x in v.keys()],[x for x in v.values()])
    '''
    print('for scene type',scene_type)
    for k,v in hit_ct.items():
        print(k,':',v)
    plt.show()
                  
def plot_velocity_profiles(gt,scene_def,freq):
    ag1_vel_profiles,ag2_vel_profiles = {'1':[],'2':[],'3':[]},{'1':[],'2':[],'3':[]}
    ag1_emp_profiles,ag2_emp_profiles = {'1':[],'2':[],'3':[]},{'1':[],'2':[],'3':[]}
    l2_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=int(1/gt.freq))
    if not (len(scene_def.agent1_emp_traj)==0 or len(scene_def.agent2_emp_traj)==0):
        emp_2l = get_nearest_node(l2_nodes, scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[0])
        for n2l in l2_nodes:
            '''assign emp path value '''
            if n2l._ext_id == emp_2l[0]:
                n2l.emp_path = True
                
                if len(scene_def.agent1_emp_traj) > 1 and n2l.children is not None:
                    emp_4l = get_nearest_node(n2l.children, scene_def.agent1_emp_traj[1]-scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[1]-scene_def.agent2_emp_traj[0])
                    for n4l in n2l.children:
                        if n4l._ext_id == emp_4l[0]:
                            n4l.emp_path = True
                            
                            if len(scene_def.agent1_emp_traj) > 2 and n4l.children is not None:
                                emp_6l = get_nearest_node(n4l.children, scene_def.agent1_emp_traj[2]-scene_def.agent1_emp_traj[1], scene_def.agent2_emp_traj[2]-scene_def.agent2_emp_traj[1])
                                for n6l in n4l.children:
                                    if n6l._ext_id == emp_6l[0]:
                                        n6l.emp_path = True
                                        ag1_emp_profiles['3'].append([x[3] for x in n6l.path_from_root['agent_1'].loaded_traj])
                                        ag2_emp_profiles['3'].append([x[3] for x in n6l.path_from_root['agent_2'].loaded_traj])
                                    else:
                                        n6l.emp_path = False
                        else:
                            n4l.emp_path = False
                f=1
            else:
                n2l.emp_path = False
            f=1
    
    if not (len(scene_def.agent1_emp_traj)==0 or len(scene_def.agent2_emp_traj)==0):
        emp_2l = get_nearest_node(l2_nodes, scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[0])
        for n2l in l2_nodes:
            '''assign emp path value '''
            #ag1_vel_profiles['1'].append([x[3] for x in n2l.path_from_root['agent_1'].loaded_traj_frag])
            #ag2_vel_profiles['1'].append([x[3] for x in n2l.path_from_root['agent_2'].loaded_traj_frag])
            if n2l._ext_id == emp_2l[0]:
                n2l.emp_path = True
            else:
                n2l.emp_path = False
            if len(scene_def.agent1_emp_traj) > 1 and n2l.children is not None:
                    emp_4l = get_nearest_node(n2l.children, scene_def.agent1_emp_traj[1]-scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[1]-scene_def.agent2_emp_traj[0])
                    for n4l in n2l.children:
                        if n4l._ext_id == emp_4l[0]:
                            n4l.emp_path = True
                        else:
                            n4l.emp_path = False
                        #ag1_vel_profiles['2'].append([x[3] for x in n4l.path_from_root['agent_1'].get_last().loaded_traj_frag])
                        #ag2_vel_profiles['2'].append([x[3] for x in n4l.path_from_root['agent_2'].get_last().loaded_traj_frag])
                        if len(scene_def.agent1_emp_traj) > 2 and n4l.children is not None:
                            emp_6l = get_nearest_node(n4l.children, scene_def.agent1_emp_traj[2]-scene_def.agent1_emp_traj[1], scene_def.agent2_emp_traj[2]-scene_def.agent2_emp_traj[1])
                            for n6l in n4l.children:
                                if n6l._ext_id == emp_6l[0]:
                                    n6l.emp_path = True
                                    
                                else:
                                    n6l.emp_path = False
                                ag1_vel_profiles['3'].append([x[3] for x in n6l.path_from_root['agent_1'].loaded_traj])
                                ag2_vel_profiles['3'].append([x[3] for x in n6l.path_from_root['agent_2'].loaded_traj])
    plt.figure()
    for k,v in ag1_vel_profiles.items():
        _l = int(k)
        for vp in v:
            plt.plot(np.linspace(start=0, stop=int(_l*int(1/freq)), num=len(vp)).tolist(),vp,color=lighten_color('red',0.5))
    for k,v in ag1_emp_profiles.items():
        _l = int(k)
        for vp in v:
            plt.plot(np.linspace(start=0, stop=int(_l*int(1/freq)), num=len(vp)).tolist(),vp,color='black')
    plt.title('ag-1')
    plt.figure()
    for k,v in ag2_vel_profiles.items():
        _l = int(k)
        for vp in v:
            plt.plot(np.linspace(start=0, stop=int(_l*int(1/freq)), num=len(vp)).tolist(),vp,color=lighten_color('blue',0.5))
    for k,v in ag2_emp_profiles.items():
        _l = int(k)
        for vp in v:
            plt.plot(np.linspace(start=0, stop=int(_l*int(1/freq)), num=len(vp)).tolist(),vp,color='black')
    
    plt.title('ag-2')
    plt.show()
    f=1
    
def main():
    run_all_scenarios()
    
if __name__ == '__main__':
    #rg_constants.SCENE_TYPE = ('synthetic','test')
    #rg_constants.CURRENT_RG_FILE_ID = '769_44_49_34,1341'
    #run_one_scenario(dbfile_id='769', agent1_id=44, agent2_id=49, start_ts=24.1341, initialize_db=True, freq=0.5)
    #animate_one_scenario('769_rt_ws_8_23_3,338667')
    plot_all_results()
    #run_all_scenarios()
    #results_all_scenarios()
    f=1
    