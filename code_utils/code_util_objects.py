'''
Created on May 18, 2021

@author: Atrisha
'''


class RunContext:
    
    def set_attrib(self,attrib_map):
        for k,v in attrib_map.items():
            setattr(self, k, v)