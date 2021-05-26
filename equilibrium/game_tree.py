'''
Created on Apr 14, 2021

@author: Atrisha
'''


import numpy as np
import sqlite3
import itertools
from planners.planning_objects import VehicleState, PedestrianState
import time
from equilibrium.equilibria_calculation import SatisficingEquilibria, RobustResponse
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
from code_utils.utils import get_all_level_nodes, get_nearest_node
import os
import traceback
import sys
from equilibrium.range_estimation import MinDistanceGapModel
from equilibrium.gametree_objects import TrajectoryCache, TrajectoryFragment
from code_utils.code_util_objects import RunContext
import all_utils
import copy
from networkx.algorithms.cuts import node_expansion

log = constants.common_logger
from equilibrium.automata_strategies import *

show_plots = False

class UnsupportedLatticeException(Exception):
    def __init__(self,l2_size,l4_size,l6_size):
        message = 'l2:'+str(l2_size)+', l4:'+str(l4_size)+', l6:'+str(l6_size)            
        super().__init__(message)
        

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
                node.automata_strategy_info = None
    
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
            if node.level == last_decision_level and node.level > 2:
                obs_trajl.append(obs_trajl[-1]+node.path_from_root[ag].next_fragment.length)
                obs_manvs.append(node.path_from_root[ag].next_fragment.manv)
                obs_manvs_mode.append(node.path_from_root[ag].next_fragment.manv_mode)
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
    
    def generate_agent_action(self,init_time,init_veh_vel,waypoint,waypoint_vels,manv,ag,horizon):
        trajs = dict()
        if manv in ['wait']:
            manv_constr = WaitTrajectoryConstraints(init_vel=init_veh_vel,waypoints=waypoint,stop_horizon_dist_sampling_range=(10,100),stop_horizon_time_sampling_range=(1,10))
        else:
            manv_constr = ProceedTrajectoryConstraints(waypoints=waypoint, waypoint_vel_sampling_range=waypoint_vels)
        manv_constr.set_limit_constraints()
        agent_motion = VehicleTrajectoryPlanner(traj_constr_obj=manv_constr, maneuver=manv, mode=None, horizon=horizon)
        agent_motion.generate_trajectory(True)
        trajs[manv] = agent_motion.all_trajectories
        return trajs
                
        
    
    def generate_actions(self,init_time,agent1_init_vel,agent2_init_vel,horizon, insert_into_db = False):
        
        horizon = 6
        agent1_trajs,agent2_trajs = dict(), dict()
        maneuver_constraints = self.maneuver_constraints
        for manv,manv_constr in maneuver_constraints['agent_1']['maneuvers'].items():
            manv_constr.set_limit_constraints()
            agent1_motion = VehicleTrajectoryPlanner(traj_constr_obj=manv_constr,maneuver= manv, mode=None, horizon=horizon)
            agent1_motion.generate_trajectory(True)
            agent1_trajs[manv] = agent1_motion.all_trajectories
        
        for manv,manv_constr in maneuver_constraints['agent_2']['maneuvers'].items():
            manv_constr.set_limit_constraints()
            agent2_motion = VehicleTrajectoryPlanner(traj_constr_obj=manv_constr,maneuver= manv, mode=None, horizon=horizon)
            agent2_motion.generate_trajectory(True)
            agent2_trajs[manv] = agent2_motion.all_trajectories
        
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        
        if insert_into_db:
            parent_traj_id = None
            conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+file_id+'.db')
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
    
    def construct_centerline(self,dist,ag,v0,maneuver_constraints):
        path = maneuver_constraints[ag]['agent_state'].waypoints
        waypoint_velocity = maneuver_constraints[ag]['maneuvers']['turn'].waypoint_vel_sampling_range if ag == 'agent_1' else maneuver_constraints[ag]['maneuvers']['track_speed'].waypoint_vel_sampling_range
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
        if len(new_path) != len(new_waypt_vel):
            f=1
        return new_path,new_waypt_vel


    def build_initial_reachability_states(self, maneuver_constraints):
        init_time = 0
        self.file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        agent1_init_vel = maneuver_constraints['agent_1']['agent_state'].velocity
        agent2_init_vel = maneuver_constraints['agent_2']['agent_state'].velocity
        time_horizon = 6
        acts = Actions(maneuver_constraints)
        acts.generate_actions(init_time,agent1_init_vel,agent2_init_vel,time_horizon,True)
        #acts.insert_interaction_data()
        
    def get_current_states(self,time_intervals,maneuver_constraints):
        state_lattice = dict()
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+self.file_id+'.db')
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
                            if abs(traj_l[i-1][0] - t_p[0]) >= lattice_dist_step:
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
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+self.file_id+'.db')
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
        
    
    def build_final_trajectories(self,maneuver_constraints):
        time_interval_axes = [[(0,2)], [(0,4),(2,2)]]
        #time_interval_axes = [[(0,4)]]
        for time_intervals in time_interval_axes:
            state_lattics = self.get_current_states(time_intervals,maneuver_constraints)
            for ts,ts_v in state_lattics.items():
                for ag,ag_v in ts_v.items():
                    for manv,manv_v in ag_v.items():
                        ct,N = 0,len(manv_v)
                        for init_st in manv_v:
                            ct += 1
                            v = init_st[3]
                            waypt,waypt_vel = self.construct_centerline(init_st[2],ag,v,maneuver_constraints)
                            assert len(waypt) == len(waypt_vel)
                            act = Actions(maneuver_constraints)
                            generating_manv_l = [x for x in maneuver_constraints[ag]['maneuvers'].keys() if x != manv]
                            for generating_manv in generating_manv_l:
                                horizon = 6 - ts
                                parent_traj_id = init_st[4]
                                if ag == 'agent_2' and (ts == 4 or ts ==2) and generating_manv == 'wait':
                                    brk = 1
                                trajs = act.generate_agent_action(ts, v, waypt, waypt_vel, generating_manv, ag, horizon)
                                self.insert_trajs_into_db(trajs, ag, ts, parent_traj_id)
                                print('generating',ag,'time',ts,'manv',generating_manv,ct,'/',N)
                                print(' '.join([str(_k)+':'+str(len(_v)) for _k,_v in trajs[generating_manv].items()]))
                
       
    def build_complete_tree(self,maneuver_constraints):
        self.build_initial_reachability_states(maneuver_constraints)
        self.build_final_trajectories(maneuver_constraints)




        
    
class Node:
    
    progress_ctr = 0
    tree_size = 0
    
    def print_Node(self,last_decision_level):
        node_result = {'mspe':False,'ag1_ac':None,'ag1_nac':None,'ag2_ac':None,'ag2_nac':None,'ag1_robust':[],'ag2_robust':[],'ag1_auto_resp':[],'ag2_auto_resp':[]}
        if self.level == last_decision_level:
            if hasattr(self, 'emp_path') and self.emp_path:
                next_emp_node = None
                for n in self.children:
                    if hasattr(n, 'emp_path') and n.emp_path:
                        next_emp_node = n
                        break
                ag1_emp_trajl = next_emp_node.path_from_root['agent_1'].get_last().length
                ag2_emp_trajl = next_emp_node.path_from_root['agent_2'].get_last().length
                if hasattr(next_emp_node, 'on_mspe') and np.any(next_emp_node.onmspe):
                    node_result['mspe'] = True
                else:
                    node_result['mspe'] = False
                if hasattr(self, 'automata_strategy_info'):
                    node_result['ag1_ac'] = self.automata_strategy_info['agent_1']['ac_auto_gamma'],
                    node_result['ag1_nac'] = self.automata_strategy_info['agent_1']['nac_auto_gamma']
                    node_result['ag2_ac'] = self.automata_strategy_info['agent_2']['ac_auto_gamma']
                    node_result['ag2_nac'] = self.automata_strategy_info['agent_2']['nac_auto_gamma']
                if hasattr(self, 'auto_strategy_response'):
                    ag1_resp = self.auto_strategy_response['agent_1'][0][:,0]
                    ag1_resp_min = self.auto_strategy_response['agent_1'][1][:,0]
                    for i,resp in enumerate(ag1_resp):
                        if  min(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']) <= ag1_emp_trajl <= max(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']):
                            node_result['ag1_auto_resp'].append(i)
                    ag2_resp = self.auto_strategy_response['agent_2'][0][0,:]
                    ag2_resp_min = self.auto_strategy_response['agent_2'][1][0,:]
                    for i,resp in enumerate(ag2_resp):
                        if  min(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']) <= ag2_emp_trajl <= max(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']):
                            node_result['ag2_auto_resp'].append(i) 
                print_str = [str(self.level)]
                for k,v in node_result.items():
                    print_str.append(k+':'+str(v))
                print(' '.join(print_str))
                    
        else:
            if not self.is_leaf: 
                for c in self.children:
                    c.print_Node(last_decision_level)
                if hasattr(self, 'emp_path') and self.emp_path:
                    next_emp_node = None
                    for n in self.children:
                        if hasattr(n, 'emp_path') and n.emp_path:
                            next_emp_node = n
                            break
                    ag1_emp_trajl = next_emp_node.path_from_root['agent_1'].get_last().length
                    ag2_emp_trajl = next_emp_node.path_from_root['agent_2'].get_last().length
                    if hasattr(next_emp_node, 'on_mspe') and np.any(next_emp_node.on_mspe):
                        node_result['mspe'] = True
                    else:
                        node_result['mspe'] = False
                    if self.automata_strategy_info is not None:
                        node_result['ag1_ac'] = self.automata_strategy_info['agent_1']['ac_auto_gamma'],
                        node_result['ag1_nac'] = self.automata_strategy_info['agent_1']['nac_auto_gamma']
                        node_result['ag2_ac'] = self.automata_strategy_info['agent_2']['ac_auto_gamma']
                        node_result['ag2_nac'] = self.automata_strategy_info['agent_2']['nac_auto_gamma']
                    if hasattr(self, 'auto_strategy_response'):
                        ag1_resp = self.auto_strategy_response['agent_1'][0][:,0]
                        ag1_resp_min = self.auto_strategy_response['agent_1'][1][:,0]
                        for i,resp in enumerate(ag1_resp):
                            if  min(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']) <= ag1_emp_trajl <= max(ag1_resp[i]['traj_l'],ag1_resp_min[i]['traj_l']):
                                node_result['ag1_auto_resp'].append(i)
                        ag2_resp = self.auto_strategy_response['agent_2'][0][0,:]
                        ag2_resp_min = self.auto_strategy_response['agent_2'][1][0,:]
                        for i,resp in enumerate(ag2_resp):
                            if  min(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']) <= ag2_emp_trajl <= max(ag2_resp[i]['traj_l'],ag2_resp_min[i]['traj_l']):
                                node_result['ag2_auto_resp'].append(i) 
                
                    print_str = [str(self.level)]
                    for k,v in node_result.items():
                        print_str.append(k+':'+str(v))
                    print(' '.join(print_str))    
        '''        
        if self.level == last_decision_level:
            if all(x==[x.manv for x in self.actions['agent_1']][0] for x in [x.manv for x in self.actions['agent_1']]):
                ag1_alleq = True
            else:
                ag1_alleq = False
            if all(x==[x.manv for x in self.actions['agent_2']][0] for x in [x.manv for x in self.actions['agent_2']]):
                ag2_alleq = True
            else:
                ag2_alleq = False
            try:
                print(self._ext_id,ag1_alleq,ag2_alleq,self.level, self.auto_strategy_response['agent_1'][0][2,2]['traj_l'] if 'agent_1' in self.auto_strategy_response else 'None', self.equilibrium_solutions[2,2][0].veh_eq_acts[0] if self.equilibrium_solutions[2,2] is not None else 'None', self.robust_response['agent_1'][2][0].veh_eq_acts[0] if self.robust_response is not None and 'agent_1' in self.robust_response else 'None',
                  self.auto_strategy_response['agent_2'][0][2,2]['traj_l'] if 'agent_2' in self.auto_strategy_response else 'None', self.equilibrium_solutions[2,2][0].peds_eq_acts[0] if self.equilibrium_solutions[2,2] is not None else 'None', self.robust_response['agent_2'][2][0].peds_eq_acts[0] if self.robust_response is not None and 'agent_2' in self.robust_response else 'None')
            except AttributeError:
                f=1
                raise
        else:
            if not self.is_leaf: 
                for c in self.children:
                    c.print_Node(last_decision_level)
                if all(x==[x.manv for x in self.actions['agent_1']][0] for x in [x.manv for x in self.actions['agent_1']]):
                    ag1_alleq = True
                else:
                    ag1_alleq = False
                if all(x==[x.manv for x in self.actions['agent_2']][0] for x in [x.manv for x in self.actions['agent_2']]):
                    ag2_alleq = True
                else:
                    ag2_alleq = False
            
                print(self._ext_id,ag1_alleq,ag2_alleq,self.level, self.auto_strategy_response['agent_1'][0][2,2]['traj_l'] if 'agent_1' in self.auto_strategy_response else 'None', self.equilibrium_solutions[2,2][0].veh_eq_acts[0] if self.equilibrium_solutions[2,2] is not None and self.equilibrium_solutions[2,2][0] is not None else 'None', self.robust_response['agent_1'][2][0].veh_eq_acts[0] if self.robust_response is not None and 'agent_1' in self.robust_response else 'None',
                  self.auto_strategy_response['agent_2'][0][2,2]['traj_l'] if 'agent_2' in self.auto_strategy_response else 'None', self.equilibrium_solutions[2,2][0].peds_eq_acts[0] if self.equilibrium_solutions[2,2] is not None else 'None', self.robust_response['agent_2'][2][0].peds_eq_acts[0] if self.robust_response is not None and 'agent_2' in self.robust_response else 'None')
        '''
    def __init__(self,level, path_from_root,_ext_id):
        self.level = level
        self.path_from_root = path_from_root
        self._ext_id = _ext_id
    
    def load(self,file_id, traj_cache = None):
        if traj_cache is None:
            traj_cache = dict()
            for x in [(0,0,2),(0,2,4),(2,2,4)]:
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
    
    @children.setter
    def children(self, value):
        self._children = value
    
    
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
    
    def __init__(self,file_id):
        self.file_id = file_id
        
    
    def _process_zero_change_tree(self,level_nodes_6s):
        v_tcache_4_6 = TrajectoryCache(init_time=0,time_range=(0,6),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6 = TrajectoryCache(init_time=0,time_range=(0,6),ag_type='agent_2',file_id=self.file_id)
        _ext_id = GameTree.counter
        self.root = Node(0,None,_ext_id)
        GameTree.counter += 1
        self.root.children = []
        
        
        
        
    def _process_two_change_tree(self,level_nodes_xs,x,level_nodes_6s):
        v_tcache_4_6 = TrajectoryCache(init_time=x,time_range=(x,6),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6 = TrajectoryCache(init_time=x,time_range=(x,6),ag_type='agent_2',file_id=self.file_id)
        _ext_id = GameTree.counter
        self.root = Node(0,None,_ext_id)
        GameTree.counter += 1
        
        for n4s,n4 in level_nodes_xs.items():
            level_nodes_xs[n4s].children = []
            if n4s[0][-1] in level_nodes_6s['agent_1'] and n4s[1][-1] in level_nodes_6s['agent_2']:
                _children_info = list(itertools.product(level_nodes_6s['agent_1'][n4s[0][-1]], level_nodes_6s['agent_2'][n4s[1][-1]]))
                n6 = []
                for _c in _children_info:
                    vtf = TrajectoryFragment(time_range=(x,6),traj_id=_c[0][-2],manv=_c[0][0],manv_mode=_c[0][1],init_time=_c[0][-3])
                    ptf = TrajectoryFragment(time_range=(x,6),traj_id=_c[1][-2],manv=_c[1][0],manv_mode=_c[1][1],init_time=_c[1][-3])
                    vtf.load(self.file_id,v_tcache_4_6.traj_cache)
                    ptf.load(self.file_id,p_tcache_4_6.traj_cache)
                    _ext_id = GameTree.counter
                    nd = Node(6,{'agent_1':vtf, 'agent_2':ptf},_ext_id)
                    GameTree.counter += 1
                    n6.append(nd)
                for _n in n6:
                    _n.children = None
                level_nodes_xs[n4s].children += n6
            if len(level_nodes_xs[n4s].children) == 0:
                level_nodes_xs[n4s].children = None
        self.root.children = list(level_nodes_xs.values())
        
    def _process_three_change_tree(self,level_nodes_2s,level_nodes_4s,level_nodes_6s):
        v_tcache_4_6, p_tcache_4_6 = dict(), dict()
        v_tcache_4_6[(4,4,6)] = TrajectoryCache(init_time=4,time_range=(4,6),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6[(4,4,6)] = TrajectoryCache(init_time=4,time_range=(4,6),ag_type='agent_2',file_id=self.file_id)
        v_tcache_4_6[(2,4,6)] = TrajectoryCache(init_time=2,time_range=(4,6),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6[(2,4,6)] = TrajectoryCache(init_time=2,time_range=(4,6),ag_type='agent_2',file_id=self.file_id)
        v_tcache_4_6[(0,4,6)] = TrajectoryCache(init_time=0,time_range=(4,6),ag_type='agent_1',file_id=self.file_id)
        p_tcache_4_6[(0,4,6)] = TrajectoryCache(init_time=0,time_range=(4,6),ag_type='agent_2',file_id=self.file_id)
        _ext_id = GameTree.counter
        self.root = Node(0,None,_ext_id)
        GameTree.counter += 1
        
        for s_path,n in level_nodes_2s.items():
            n_children = []
            for n4s,n4 in level_nodes_4s.items():
                if s_path[0][0] == n4s[0][0] and s_path[1][0] == n4s[1][0]:
                    level_nodes_4s[n4s].children = []
                    n_children.append(level_nodes_4s[n4s])
                    if n4s[0][-1] in level_nodes_6s['agent_1'] and n4s[1][-1] in level_nodes_6s['agent_2']:
                        _children_info = list(itertools.product(level_nodes_6s['agent_1'][n4s[0][-1]], level_nodes_6s['agent_2'][n4s[1][-1]]))
                        n6 = []
                        for _c in _children_info:
                            vtf = TrajectoryFragment(time_range=(4,6),traj_id=_c[0][-2],manv=_c[0][0],manv_mode=_c[0][1],init_time=_c[0][-3])
                            ptf = TrajectoryFragment(time_range=(4,6),traj_id=_c[1][-2],manv=_c[1][0],manv_mode=_c[1][1],init_time=_c[1][-3])
                            vtf.load(self.file_id,v_tcache_4_6)
                            ptf.load(self.file_id,p_tcache_4_6)
                            _newvtf = copy.deepcopy(level_nodes_4s[n4s].path_from_root['agent_1'])
                            _newvtf.is_last = False
                            _newvtf.get_last().next_fragment = vtf
                            _newptf = copy.deepcopy(level_nodes_4s[n4s].path_from_root['agent_2'])
                            _newptf.is_last = False
                            _newptf.get_last().next_fragment = ptf
                            _ext_id = GameTree.counter
                            nd = Node(6,{'agent_1':_newvtf, 'agent_2':_newptf},_ext_id)
                            GameTree.counter += 1
                            n6.append(nd)
                        for _n in n6:
                            _n.children = None
                        level_nodes_4s[n4s].children += n6
                    if len(level_nodes_4s[n4s].children) == 0:
                        level_nodes_4s[n4s].children = None
            n.children = n_children
            
            
                
        self.root.children = list(level_nodes_2s.values())
        
        
        
    def build_tree(self):
        print('building lattice nodes...')
        start_time = time.time()
        level_nodes_2s = self.build_level_nodes(2)
        level_nodes_4s = self.build_level_nodes(4)
        level_nodes_6s = self.build_level_nodes(6)
        if len(level_nodes_2s) == 0 or len(level_nodes_4s) == 0 or len(level_nodes_6s['agent_1'])==0 or len(level_nodes_6s['agent_2'])==0:
            raise UnsupportedLatticeException(l2_size=len(level_nodes_2s), l4_size=len(level_nodes_4s), l6_size=(len(level_nodes_6s['agent_1']),len(level_nodes_6s['agent_2'])))
        self.last_decision_level = 4
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
        eq_class.set_oneq_label(self)
    
    def print_tree(self):
        self.root.print_Node(6)
    
    def build_level_nodes(self, level):
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+self.file_id+'.db')
        c = conn.cursor()
        all_level_nodes = dict()
            
        def _getqstring(ag_type):
            return "select MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X)*ABS(INIT_POS_X-X)+ABS(INIT_POS_Y-Y)*ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.INIT_TIME,TRAJECTORY_METADATA.TRAJ_ID,TRAJECTORY_METADATA.PARENT_TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME=4 AND TRAJECTORIES.TIME=2 \
                        AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' \
                        UNION \
                        select MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X)*ABS(INIT_POS_X-X)+ABS(INIT_POS_Y-Y)*ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.INIT_TIME,TRAJECTORY_METADATA.TRAJ_ID,TRAJECTORY_METADATA.PARENT_TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME=2 AND TRAJECTORIES.TIME=4 \
                        AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.TRAJ_ID IN (select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.INIT_TIME=4) \
                        UNION \
                        select MANEUVER, MANEUVER_MODE, SPEED, ABS(INIT_POS_X-X)*ABS(INIT_POS_X-X)+ABS(INIT_POS_Y-Y)*ABS(INIT_POS_Y-Y),X,Y,ANGLE,TRAJECTORY_METADATA.INIT_TIME,TRAJECTORY_METADATA.TRAJ_ID,TRAJECTORY_METADATA.PARENT_TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME=0 AND TRAJECTORIES.TIME=6 AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.TRAJ_ID IN (select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"' AND TRAJECTORY_METADATA.INIT_TIME=2)"
        if level == 6:
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
                        
                if veh_row[-3] == 4:
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
                if peds_row[-3] == 4:
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
            
            if level == 4:
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
            if level == 4:
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
                    vtf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2])
                    if level == 4:
                        vtf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2])
                        vtf2.is_last = True
                        vtf1.next_fragment = vtf2
                    else:
                        vtf1.is_last = True
                else:
                    ''' get the grandparent fragment '''
                    vtf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = veh_row[10] ,manv = veh_traj_info[veh_row[10]][0],manv_mode = veh_traj_info[veh_row[10]][1],init_time = veh_traj_info[veh_row[10]][2])
                    if level == 4:
                        vtf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2])
                        vtf2.is_last = True
                        vtf1.next_fragment = vtf2
                    else:
                        vtf1.is_last = True
                for idx2,peds_row in enumerate(peds_res):
                    ct += 1
                    #print('level',level,ct,'/',N)
                    if peds_row[9] == 0:
                        ptf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2])
                        if level == 4:
                            ptf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2])
                            ptf2.is_last = True
                            ptf1.next_fragment = ptf2
                        else:
                            ptf1.is_last = True
                    else:
                        ptf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = peds_row[10] ,manv = peds_traj_info[peds_row[10]][0],manv_mode = peds_traj_info[peds_row[10]][1],init_time = peds_traj_info[peds_row[10]][2])
                        if level == 4:
                            ptf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2])
                            ptf2.is_last = True
                            ptf1.next_fragment = ptf2
                        else:
                            ptf1.is_last = True
                    _ext_id = GameTree.counter
                    nd = Node(level,{'agent_1':vtf1, 'agent_2':ptf1}, _ext_id)
                    GameTree.counter += 1
                    #nd.load()
                    if level == 4:
                        s_key = ((vtf1.traj_id,vtf2.traj_id),(ptf1.traj_id,ptf2.traj_id))
                        print(s_key,(vtf1.manv,vtf2.manv),(ptf1.manv,ptf2.manv))
                    else:
                        s_key = ((vtf1.traj_id,),(ptf1.traj_id,))
                        print(s_key,(vtf1.manv),(ptf1.manv))
                    all_level_nodes[s_key] = nd
                
        return all_level_nodes

def run_all_scenarios():
    initialize_db = True
    initialize_files = False
    with open(rg_constants.SCENE_OUT_PATH,newline='\n') as csv_file:
        sc_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in sc_reader:
            if not initialize_files:
                if os.path.isfile(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt')):
                    print('row',row,'processed...continuing')
                    continue
            dbfile_id = row[0]
            agent1_id = int(row[3])
            agent2_id = int(row[4])
            start_ts = float(row[5])
            
            print('processing',dbfile_id,line_count+1,agent1_id,agent2_id)
            try:
                scene_def = ScenarioDef(agent_1_id=agent1_id,agent_2_id=agent2_id,file_id=dbfile_id,initialize_db=initialize_db,start_ts=start_ts)
                if scene_def.time_crossed:
                    continue
                maneuver_constraints = scene_def.setup_trajectory_constraints()
                if initialize_db:
                    tree_builder = TreeBuilder()
                    tree_builder.build_complete_tree(maneuver_constraints)
                
                file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
                gt = GameTree(file_id)
                gt.build_tree()
                type(gt.root).progress_ctr = 0
                type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
                m = MinDistanceGapModel(file_id)
                m.build_model()   
                context = RunContext()
                manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'track_speed'}}
                context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True})
                drassign_obj = AssignDistRanges()
                drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
                start_time = time.time()
                eq_obj = SatisficingEquilibria(context)
                gt.solve(eq_obj)
                print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
                start_time = time.time()
                gt.solve(RobustResponse(context))
                print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
                gt.scene_def = scene_def
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
                if not isinstance(e, UnsupportedLatticeException):
                    raise
                '''
            line_count += 1
            if line_count > 2:
                break
            


    
def assign_emp_nodes(gt,scene_def):
    gt.root.emp_path = True
    l2_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=2)
    emp_2l = get_nearest_node(l2_nodes, scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[0])
    for n2l in l2_nodes:
        '''assign emp path value '''
        if n2l._ext_id == emp_2l[0]:
            n2l.emp_path = True
            emp_4l = get_nearest_node(n2l.children, scene_def.agent1_emp_traj[1]-scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[1]-scene_def.agent2_emp_traj[0])
            for n4l in n2l.children:
                if n4l._ext_id == emp_4l[0]:
                    n4l.emp_path = True
                    emp_6l = get_nearest_node(n4l.children, scene_def.agent1_emp_traj[2]-scene_def.agent1_emp_traj[1], scene_def.agent2_emp_traj[2]-scene_def.agent2_emp_traj[1])
                    for n6l in n4l.children:
                        if n6l._ext_id == emp_6l[0]:
                            n6l.emp_path = True
                        else:
                            n6l.emp_path = False
                else:
                    n4l.emp_path = False
            f=1
        else:
            n2l.emp_path = False
        f=1
        
        

def results_all_scenarios():
    with open(rg_constants.SCENE_OUT_PATH,newline='\n') as csv_file:
        sc_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in sc_reader:
            if os.path.isfile(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt')):
                dbfile_id = row[0]
                agent1_id = int(row[3])
                agent2_id = int(row[4])
                start_ts = float(row[5])
                scene_def = ScenarioDef(agent_1_id=agent1_id,agent_2_id=agent2_id,file_id=dbfile_id,initialize_db=False,start_ts=start_ts)
                gt = all_utils.utils.pickle_load(os.path.join(rg_constants.TREE_FILES,'_'.join(row).replace('.',',')+'.gt'))
                gt.scene_def = scene_def
                assign_emp_nodes(gt,scene_def)
                gt.print_tree()
                if line_count == 0:
                    break
                line_count += 1
            print('processing',dbfile_id,line_count+1,agent1_id,agent2_id)
            

if __name__ == '__main__':
    run_all_scenarios()