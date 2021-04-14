'''
Created on Apr 14, 2021

@author: Atrisha
'''


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
