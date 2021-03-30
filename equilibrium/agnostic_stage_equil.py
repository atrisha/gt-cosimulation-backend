'''
Created on Mar 22, 2021

@author: Atrisha
'''


class TrajectoryAgnosticEqStage():
    
    def __init__(self, input_struc):
        
        self.input_struc = input_struc
        
    
    '''
        This takes in a key value pair of (m,t_idx) : u_i for a give -i action.
        Generates the best response set modulo manuver.
        Also generates the utility interval for the best response set.
    '''
    def calc_br_correspondence(self,util_dict):
        f=1
        
        