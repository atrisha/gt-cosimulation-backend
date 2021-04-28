'''
Created on Apr 14, 2021

@author: Atrisha
'''


import numpy as np
import sqlite3
import itertools
from planners.planning_objects import VehicleState, PedestrianState, TrajectoryFragment
import time
import math
from equilibrium.equilibria_calculation import SatisficingEquilibria
from numpy import linalg as LA
import copy


show_plots = False

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
gt = GameTree()
gt.build_tree()
gt.solve(SatisficingEquilibria())
f=1
           
                
                
        
        