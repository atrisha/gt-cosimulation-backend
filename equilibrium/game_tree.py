'''
Created on Apr 14, 2021

@author: Atrisha
'''


import numpy as np
import sqlite3
import itertools
from planners.planning_objects import VehicleState, PedestrianState
import time

show_plots = False

class TrajectoryFragment:
    
    def __init__(self,time_range,traj_id,manv,manv_mode,init_time):
        self.time_range = time_range
        self.init_time = init_time
        self.traj_id = traj_id
        self.manv = manv
        self.manv_mode = manv_mode
        self._is_last = False
    
    def set_next_fragment(self, traj_fragment):
        self.next_fragment = traj_fragment
    
    @property
    def is_last(self):
        return self._is_last or self.time_range[1] == 6
    
    @is_last.setter
    def is_last(self, value):
        self._is_last = value
        
    def load(self):
        end_time = self.time_range[1] if self.time_range[1] == 6 else self.time_range[1]-self.init_time-0.01
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES WHERE TRAJECTORIES.TRACK_ID="+str(self.traj_id)+" AND TRAJECTORIES.TIME BETWEEN "+str(int(self.time_range[0]-self.init_time))+" AND "+str(end_time)+" ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        loaded_traj_frag = res
        loaded_traj = []
        self.loaded_traj_frag = loaded_traj_frag
        if not self.is_last:
            loaded_traj = loaded_traj_frag + self.next_fragment.load()
            self.loaded_traj = loaded_traj
            return self.loaded_traj
        else:
            self.loaded_traj = loaded_traj_frag
            return loaded_traj_frag
        
class Node:
    
    def __init__(self,level, path_from_root):
        self.level = level
        self.path_from_root = path_from_root
    
    def load(self):
        if not self.is_root:
            for p in self.path_from_root.values():
                p.load()
        if self.children is not None:
            for n in self.children:
                n.load()
        
    
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
    
    def build_tree(self):
        level_nodes_2s = self.build_level_nodes(2)
        level_nodes_4s = self.build_level_nodes(4)
        self.root = Node(0,None)
        print('building tree')
        start_time = time.time()
        for s_path,n in level_nodes_2s.items():
            n_children = []
            for n4s in level_nodes_4s.keys():
                if s_path[0][0] == n4s[0][0] and s_path[1][0] == n4s[1][0]:
                    level_nodes_4s[n4s].children = None
                    n_children.append(level_nodes_4s[n4s])
            n.children = n_children
        self.root.children = list(level_nodes_2s.values())
        self.root.load()
        print('building tree....DONE','(%s secs)' % (time.time() - start_time),)
            
            
        
    
    def build_level_nodes(self, level):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        all_level_nodes = dict()
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
                nd = Node(level,{'vehicle':vtf1, 'pedestrian':ptf1})
                #nd.load()
                if level == 4:
                    s_key = ((vtf1.traj_id,vtf2.traj_id),(ptf1.traj_id,ptf2.traj_id))
                else:
                    s_key = ((vtf1.traj_id,),(ptf1.traj_id,))
                all_level_nodes[s_key] = nd
                
        return all_level_nodes
gt = GameTree()
gt.build_tree()
           
                
                
        
        