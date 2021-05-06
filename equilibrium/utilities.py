'''
Created on Apr 15, 2021

@author: Atrisha
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.special





class Utilities:
    
    def __init__(self):
        self._util_info = None
        
    @property
    def util_info(self):
        return self._util_info
    
    @util_info.setter
    def util_info(self, util_info):
        self._util_info = util_info
    
    def exp_dist_payoffs(self,dist_arr,params=None):
        params = (5.5,1,.25)
        if not isinstance(params, np.ndarray):
            if dist_arr < 7:
                c2 = scipy.special.erf((dist_arr - params[0]) / (params[1] * 2))
                x = np.exp(np.log(c2+2)/7)
                u = x**dist_arr -2
            else:
                u = scipy.special.erf((dist_arr - params[0]) / (params[1] * 2))
        else:
            u = scipy.special.erf((dist_arr - params[:,:,0]) / (params[:,:,1] * 2))
        if u < -1 or u > 1:
            brk = 1
        return u
    
    def generate_safety_utils(self,ped_trajectory,veh_trajectory):
        dist_gap = min([math.hypot(x[1]-y[1], x[2]-y[2]) for x,y in zip(ped_trajectory,veh_trajectory)])
        safe_utils = self.exp_dist_payoffs(dist_gap)
        return safe_utils
    
    def calc_dist_gap(self,veh_traj,ped_traj,xy_indexes=None):
        if xy_indexes is None:
            xy_indexes = (1,2)
        dist_gap = min([math.hypot(x[xy_indexes[0]]-y[xy_indexes[0]], x[xy_indexes[1]]-y[xy_indexes[1]]) for x,y in zip(veh_traj,ped_traj)])
        return dist_gap
    
    def calc_safe_payoff(self,dist_gap):
        safe_utils = self.exp_dist_payoffs(dist_gap)
        return safe_utils
    
    def calc_traj_length(self,traj):
        
        len = sum([math.hypot(x[0][0]- x[1][0],x[0][1]- x[1][1]) for x in zip(traj[:-1],traj[1:])])
        return len
        
    def progress_payoff(self,m,i):
        if m == 'wait':
            return -1 if i == 1 else -0.75
        else:
            return 1
        
    def progress_payoff_dist(self,dist,ag_type):
        prog_util = min(dist / 100, 1)
        return prog_util
        
    def combine_utils(self,prog_util,safe_util,thresh):
        if isinstance(thresh, np.ndarray):
            prog_util_matrix = np.full(shape=thresh.shape, fill_value=prog_util)
            safe_util_matrix = np.full(shape=thresh.shape, fill_value=safe_util)
            comb_util_matrix = np.where(safe_util_matrix < thresh, safe_util_matrix, prog_util_matrix)
            return comb_util_matrix
        else:
            if safe_util < thresh:
                return safe_util
            else:
                return prog_util
        