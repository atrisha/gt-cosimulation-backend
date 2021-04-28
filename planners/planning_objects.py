'''
Created on Apr 14, 2021

@author: Atrisha
'''
import sqlite3
from shapely.geometry import LineString

class AgentState:
    
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
    @property
    def x(self):
        return self.x
    
    @property
    def y(self):
        return self.y
    
    @property
    def velocity(self):
        return self.velocity
    
    @x.setter
    def x(self,x):
        self.x = x
    
    @y.setter
    def y(self,y):
        self.y = y
    
    @velocity.setter
    def velocity(self, vel):
        self.velocity = vel
        
class VehicleState(AgentState):
    
    def __init__(self,**kwargs):
        AgentState.__init__(kwargs)


class PedestrianState(AgentState):
    
    def __init__(self,**kwargs):
        AgentState.__init__(kwargs)
        
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
        
    def load(self,t_cache = None):
        if not self.loaded:
            end_time = self.time_range[1] if self.time_range[1] == 6 else self.time_range[1]-self.init_time-0.01
            if t_cache is not None and self.traj_id in t_cache:
                res = t_cache[self.traj_id][self.time_range]
            else:
                conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
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
            loaded_traj = loaded_traj_frag + self.next_fragment.load()
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

