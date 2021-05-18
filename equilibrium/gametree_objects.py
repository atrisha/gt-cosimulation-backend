'''
Created on May 17, 2021

@author: Atrisha
'''

import sqlite3
from shapely.geometry import Polygon, LineString

class TrajectoryCache:
    
    def __init__(self,init_time,time_range,ag_type,file_id):
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+file_id+'.db')
        c = conn.cursor()
        if ag_type is not None:
            q_string = "select * from TRAJECTORIES WHERE TRAJECTORIES.TRACK_ID IN ( \
                    select TRAJECTORY_METADATA.TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME="+str(init_time)+" AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"') \
                    AND TRAJECTORIES.TIME BETWEEN "+str(time_range[0]-init_time)+" AND "+str(time_range[1]-init_time)
        else:
            q_string = "select * from TRAJECTORIES WHERE TRAJECTORIES.TRACK_ID IN ( \
                    select TRAJECTORY_METADATA.TRAJ_ID from TRAJECTORIES INNER JOIN TRAJECTORY_METADATA ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID WHERE TRAJECTORY_METADATA.INIT_TIME="+str(init_time)+") \
                    AND TRAJECTORIES.TIME BETWEEN "+str(time_range[0]-init_time)+" AND "+str(time_range[1]-init_time)
        c.execute(q_string)
        res = c.fetchall()
        self.traj_cache = dict()
        for row in res:
            if row[0] not in self.traj_cache:
                self.traj_cache[row[0]] = dict()
                self.traj_cache[row[0]][time_range] = []
            self.traj_cache[row[0]][time_range].append(row)
        



class TrajectoryFragment:
    
    def __init__(self,time_range,traj_id,manv,manv_mode,init_time):
        self.time_range = time_range
        self.init_time = init_time
        self.traj_id = traj_id
        self.manv = manv
        self.manv_mode = manv_mode
        self._is_last = False
        self.loaded = False
    
    def set_next_fragment(self, traj_fragment):
        self.next_fragment = traj_fragment
    
    @property 
    def length(self):
        traj_path = LineString([(x[1],x[2]) for x in self.loaded_traj_frag])
        #math.hypot(self.loaded_traj_frag[0][1] - self.loaded_traj_frag[-1][1], self.loaded_traj_frag[0][2] - self.loaded_traj_frag[-1][2])
        return traj_path.length 
        
    
    @property
    def is_last(self):
        return self._is_last or self.time_range[1] == 6
    
    @is_last.setter
    def is_last(self, value):
        self._is_last = value
    
    @property
    def loaded(self):
        return self._loaded
    
    @loaded.setter
    def loaded(self, value):
        self._loaded = value
        
    def load(self, file_id,t_cache):
        if not self.loaded:
            end_time = self.time_range[1] if self.time_range[1] == 6 else self.time_range[1]-self.init_time-0.01
            if t_cache is not None and t_cache[(self.init_time,self.time_range[0],self.time_range[1])] is not None and self.traj_id in t_cache[(self.init_time,self.time_range[0],self.time_range[1])].traj_cache:
                res = t_cache[(self.init_time,self.time_range[0],self.time_range[1])].traj_cache[self.traj_id][self.time_range]
            else:
                conn = sqlite3.connect('D:\\repeated_games_data\\'+file_id+'.db')
                c = conn.cursor()
                q_string = "select * from TRAJECTORIES WHERE TRAJECTORIES.TRACK_ID="+str(self.traj_id)+" AND TRAJECTORIES.TIME BETWEEN "+str(int(self.time_range[0]-self.init_time))+" AND "+str(end_time)+" ORDER BY TIME"
                c.execute(q_string)
                res = c.fetchall()
            loaded_traj_frag = res
            loaded_traj = []
            self.loaded_traj_frag = loaded_traj_frag
            self.loaded = True
        else:
            loaded_traj_frag = self.loaded_traj_frag
        if not self.is_last:
            loaded_traj = loaded_traj_frag + self.next_fragment.load(file_id, t_cache)
            self.loaded_traj = loaded_traj
            return self.loaded_traj
        else:
            self.loaded_traj = loaded_traj_frag
            return self.loaded_traj
        
            
    
    def get_last(self):
        _tf = self
        while not _tf.is_last:
            _tf = _tf.next_fragment
        return _tf

