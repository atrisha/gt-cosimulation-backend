'''
Created on Apr 14, 2021

@author: Atrisha
'''
import sqlite3
from shapely.geometry import LineString

class AgentState:
    
    def __init__(self,attrib_map):
        for k,v in attrib_map.items():
            setattr(self, k, v)
    '''   
    @property
    def x(self):
        return self.x
    
    @property
    def y(self):
        return self.y
    
    @property
    def velocity(self):
        return self.velocity
    
    @property
    def waypoints(self):
        return self.waypoints
    '''
        
class VehicleState(AgentState):
    
    def __init__(self,attrib_map):
        super().__init__(attrib_map)


class PedestrianState(AgentState):
    
    def __init__(self,attrib_map):
        super().__init__(attrib_map)
        
