'''
Created on Apr 14, 2021

@author: Atrisha
'''


import numpy as np
import sqlite3
import itertools
from planners.planning_objects import VehicleState, PedestrianState, TrajectoryFragment
import time
from equilibrium.equilibria_calculation import SatisficingEquilibria
import copy
from maps.map_info import NYCMapInfo
from mpl_toolkits.mplot3d import Axes3D
from planners.trajectory_planner import VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner
from equilibrium.utilities import Utilities
from numpy import linalg as LA
import math



show_plots = False


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

veh_maneuvers = ['turn','wait']
ped_maneuver = ['walk','wait']
modes = ['normal','aggressive']

        
class Actions:
    
    def generate_agent_action(self,init_time,init_veh_vel,waypoint,waypoint_vels,manv,ag,horizon):
        trajs = dict()
        if ag == 'vehicle':
            motion = VehicleTrajectoryPlanner(waypoint,waypoint_vels,manv,None,horizon)
            motion.generate_trajectory(True)
            trajs[manv] = motion.all_trajectories
        else:
            motion = PedestrianTrajectoryPlanner(waypoint,waypoint_vels,manv,None,horizon)
            motion.generate_trajectory(True)
            trajs[manv] = motion.all_trajectories
        return trajs
                
        
    
    def generate_actions(self,init_time,init_veh_vel,init_ped_vel,horizon, insert_into_db = False):
        
        
        veh_trajs,ped_trajs = dict(), dict()
        veh_waypoint = NYCMapInfo.veh_centerline
        veh_waypoint_velocity = [(init_veh_vel,),(None,),(1,init_veh_vel),(None,),(init_veh_vel,10)]
        ped_waypoint = NYCMapInfo.ped_centerline
        ped_waypoint_velocity = [(init_ped_vel,),(init_ped_vel,1.8),(init_ped_vel,1.8)]
        
        for veh_m in veh_maneuvers:
            veh_motion = VehicleTrajectoryPlanner(veh_waypoint,veh_waypoint_velocity,veh_m,None,horizon)
            veh_motion.generate_trajectory(True)
            veh_trajs[veh_m] = veh_motion.all_trajectories
        
    
        for ped_m in ped_maneuver:
            ped_motion = PedestrianTrajectoryPlanner(ped_waypoint,ped_waypoint_velocity,ped_m,None,horizon)
            ped_motion.generate_trajectory(True)
            ped_trajs[ped_m] = ped_motion.all_trajectories
        
        if insert_into_db:
            parent_traj_id = None
            conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
            c = conn.cursor()
            i_string = 'INSERT INTO TRAJECTORIES VALUES (?,?,?,?,?,?,?,?,?)'
            i_string_tj_mtdata = 'INSERT INTO TRAJECTORY_METADATA VALUES (?,?,?,?,?,?,?,?,?,?,?)'
            traj_id = 1
            for ag_type_idx,traj_det_dict in enumerate([veh_trajs,ped_trajs]):
                ag_type = 'vehicle' if ag_type_idx == 0 else 'pedestrian'
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
            f=1  
        return veh_trajs, ped_trajs
    
    def insert_interaction_data(self):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='vehicle'"
        c.execute(q_string)
        veh_trajids = c.fetchall()
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='pedestrian'"
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
        #acts.insert_interaction_data()
        
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
        time_interval_axes = [[(0,2)], [(0,4),(2,2)]]
        #time_interval_axes = [[(0,4)]]
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



class TrajectoryCache:
    
    def __init__(self,init_time,time_range,ag_type):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES WHERE TRAJECTORIES.TRACK_ID IN ( \
                    select TRAJECTORY_METADATA.TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME="+str(init_time)+" AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"') \
                    AND TRAJECTORIES.TIME BETWEEN "+str(time_range[0]-init_time)+" AND "+str(time_range[1]-init_time)
        c.execute(q_string)
        res = c.fetchall()
        self.traj_cache = dict()
        for row in res:
            if row[0] not in self.traj_cache:
                self.traj_cache[row[0]] = dict()
                self.traj_cache[row[0]][time_range] = []
            self.traj_cache[row[0]][time_range].append(row)
        

        
    
class Node:
    
    def __init__(self,level, path_from_root,_ext_id):
        self.level = level
        self.path_from_root = path_from_root
        self._ext_id = _ext_id
    
    def load(self):
        if not self.is_root:
            for p in self.path_from_root.values():
                p.load()
        if self.children is not None:
            for n in self.children:
                n.load()
        '''
        if self.is_leaf:
            for p in self.actions['vehicle']:
                p.load()
            for p in self.actions['pedestrian']:
                p.load()
        ''' 
    def set_actions(self):
        actions = {'vehicle':[],'pedestrian':[]}
        ''' get the children nodes and their path from root.
            actions are the last trajectory fragment of that path
        '''
        if self.is_leaf:
            veh_path_from_root = self.path_from_root['vehicle']
            peds_path_from_root = self.path_from_root['pedestrian']
            v_tf = veh_path_from_root.get_last()
            p_tf = peds_path_from_root.get_last()
            self.actions = None
            v_tf._next_node = None
            p_tf._next_node = None
                
        else:
            for cidx,c in enumerate(self.children):
                c.set_actions()
                veh_path_from_root = c.path_from_root['vehicle']
                peds_path_from_root = c.path_from_root['pedestrian']
                v_tf = copy.copy(veh_path_from_root.get_last())
                p_tf = copy.copy(peds_path_from_root.get_last())
                v_tf._next_node = c
                p_tf._next_node = c
                actions['vehicle'].append(v_tf)
                actions['pedestrian'].append(p_tf)
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
        
class GameTree:
    
    counter = 1
    
    def build_tree(self):
        print('building lattice nodes...')
        start_time = time.time()
        level_nodes_2s = self.build_level_nodes(2)
        level_nodes_4s = self.build_level_nodes(4)
        level_nodes_6s = self.build_level_nodes(6)
        v_tcache_4_6 = TrajectoryCache(init_time=4,time_range=(4,6),ag_type='vehicle')
        p_tcache_4_6 = TrajectoryCache(init_time=4,time_range=(4,6),ag_type='pedestrian')
        _ext_id = GameTree.counter
        self.root = Node(0,None,_ext_id)
        GameTree.counter += 1
        
        for s_path,n in level_nodes_2s.items():
            n_children = []
            for n4s,n4 in level_nodes_4s.items():
                
                
                if s_path[0][0] == n4s[0][0] and s_path[1][0] == n4s[1][0]:
                    level_nodes_4s[n4s].children = []
                    n_children.append(level_nodes_4s[n4s])
                    if n4s[0][-1] in level_nodes_6s['vehicle'] and n4s[1][-1] in level_nodes_6s['pedestrian']:
                        _children_info = list(itertools.product(level_nodes_6s['vehicle'][n4s[0][-1]], level_nodes_6s['pedestrian'][n4s[1][-1]]))
                        n6 = []
                        for _c in _children_info:
                            vtf = TrajectoryFragment(time_range=(4,6),traj_id=_c[0][-2],manv=_c[0][0],manv_mode=_c[0][1],init_time=_c[0][-3])
                            ptf = TrajectoryFragment(time_range=(4,6),traj_id=_c[1][-2],manv=_c[1][0],manv_mode=_c[1][1],init_time=_c[1][-3])
                            vtf.load(v_tcache_4_6.traj_cache)
                            ptf.load(p_tcache_4_6.traj_cache)
                            _ext_id = GameTree.counter
                            nd = Node(6,{'vehicle':vtf, 'pedestrian':ptf},_ext_id)
                            GameTree.counter += 1
                            n6.append(nd)
                        for _n in n6:
                            _n.children = None
                        level_nodes_4s[n4s].children += n6
                    if len(level_nodes_4s[n4s].children) == 0:
                        level_nodes_4s[n4s].children = None
            n.children = n_children
        self.root.children = list(level_nodes_2s.values())
        print('building lattice nodes....DONE','(%s secs)' % (time.time() - start_time),)
        print('loading tree...')
        start_time = time.time()
        self.root.load()
        print('loading tree....DONE','(%s secs)' % (time.time() - start_time),)
        print('setting actions...')
        start_time = time.time()
        self.root.set_actions()
        print('setting actions....DONE','(%s secs)' % (time.time() - start_time),)
        f=1 
            
    
    def solve(self,eq_class):
        
        eq_class.solve(node = self.root)
        
    
    def build_level_nodes(self, level):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
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
            c.execute(_getqstring('vehicle'))
            veh_res = c.fetchall()
            c.execute(_getqstring('pedestrian'))
            peds_res = c.fetchall()
            #veh_res = np.array([(x[0],x[1],x[2],LA.norm([x[3],x[4]]),LA.norm([x[3],x[4]]),x[5],x[6],x[7],x[8],x[9],x[10]) for x in veh_res])
            veh_lattice_states,latc_tracker_veh = dict(),dict()
            #peds_res = np.array([(x[0],x[1],x[2],LA.norm([x[3],x[4]]),LA.norm([x[3],x[4]]),x[5],x[6],x[7],x[8],x[9],x[10]) for x in peds_res])
            peds_lattice_states,latc_tracker_ped = dict(), dict()
            for idx1,veh_row in enumerate(veh_res):
                parent_traj_id = veh_row[-1] if veh_row[-1] is not None else veh_row[-2]
                if parent_traj_id not in veh_lattice_states:
                        veh_lattice_states[parent_traj_id] = []
                        
                if veh_row[-3] == 4:
                    if parent_traj_id not in latc_tracker_veh or ( math.sqrt(veh_row[3])-min(latc_tracker_veh[parent_traj_id], key=lambda x:abs(x-math.sqrt(veh_row[3]))) > 0.5 ):
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
                    if parent_traj_id not in latc_tracker_ped or ( math.sqrt(peds_row[3])-min(latc_tracker_ped[parent_traj_id], key=lambda x:abs(x-math.sqrt(peds_row[3]))) > 0.1 ):
                        if parent_traj_id not in latc_tracker_ped:
                            latc_tracker_ped[parent_traj_id] = []
                        
                        
                        peds_lattice_states[parent_traj_id].append(tuple([x if _i !=3 else math.sqrt(x) for _i,x in enumerate(peds_row)]))
                        latc_tracker_ped[parent_traj_id].append(math.sqrt(peds_row[3]))
                else:
                    peds_lattice_states[parent_traj_id].append(tuple([x if _i !=3 else math.sqrt(x) for _i,x in enumerate(peds_row)]))
            
            all_level_nodes['vehicle'] = veh_lattice_states
            all_level_nodes['pedestrian'] = peds_lattice_states
        else:
            q_string = "SELECT *  FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.TRAJ_ID IN ( \
                            select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+'vehicle'+"' AND TRAJECTORY_METADATA.INIT_TIME="+str(level)+");"
            c.execute(q_string)
            veh_res = c.fetchall()
            veh_traj_info = {row[0]:(row[6],row[7],row[9]) for row in veh_res}
            
            q_string = "SELECT *  FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.TRAJ_ID IN ( \
                            select distinct PARENT_TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='"+'pedestrian'+"' AND TRAJECTORY_METADATA.INIT_TIME="+str(level)+");"
            c.execute(q_string)
            peds_res = c.fetchall()
            peds_traj_info = {row[0]:(row[6],row[7],row[9]) for row in peds_res}
            ct,N = 0,len(veh_res)*len(peds_res)
            for idx1,veh_row in enumerate(veh_res):
                if veh_row[9] == 0:
                    vtf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2])
                    if level == 4:
                        vtf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2])
                        vtf2.is_last = True
                        vtf1.set_next_fragment(vtf2)
                    else:
                        vtf1.is_last = True
                else:
                    if veh_row[10] not in veh_traj_info:
                        continue
                    vtf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = veh_row[10] ,manv = veh_traj_info[veh_row[10]][0],manv_mode = veh_traj_info[veh_row[10]][1],init_time = veh_traj_info[veh_row[10]][2])
                    if level == 4:
                        vtf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = veh_row[0] ,manv = veh_traj_info[veh_row[0]][0],manv_mode = veh_traj_info[veh_row[0]][1],init_time = veh_traj_info[veh_row[0]][2])
                        vtf2.is_last = True
                        vtf1.set_next_fragment(vtf2)
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
                            ptf1.set_next_fragment(ptf2)
                        else:
                            ptf1.is_last = True
                    else:
                        if peds_row[10] not in peds_traj_info:
                            continue
                        ptf1 = TrajectoryFragment(time_range = (0,2) ,traj_id = peds_row[10] ,manv = peds_traj_info[peds_row[10]][0],manv_mode = peds_traj_info[peds_row[10]][1],init_time = peds_traj_info[peds_row[10]][2])
                        if level == 4:
                            ptf2 = TrajectoryFragment(time_range = (2,4) ,traj_id = peds_row[0] ,manv = peds_traj_info[peds_row[0]][0],manv_mode = peds_traj_info[peds_row[0]][1],init_time = peds_traj_info[peds_row[0]][2])
                            ptf2.is_last = True
                            ptf1.set_next_fragment(ptf2)
                        else:
                            ptf1.is_last = True
                    _ext_id = GameTree.counter
                    nd = Node(level,{'vehicle':vtf1, 'pedestrian':ptf1}, _ext_id)
                    GameTree.counter += 1
                    #nd.load()
                    if level == 4:
                        s_key = ((vtf1.traj_id,vtf2.traj_id),(ptf1.traj_id,ptf2.traj_id))
                    else:
                        s_key = ((vtf1.traj_id,),(ptf1.traj_id,))
                    all_level_nodes[s_key] = nd
                
        return all_level_nodes
    
tree_builder = TreeBuilder()
tree_builder.build_complete_tree()
gt = GameTree()
gt.build_tree()
start_time = time.time()
gt.solve(SatisficingEquilibria())
print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
           
                
                
        
        