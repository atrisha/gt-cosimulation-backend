'''
Created on Apr 15, 2021

@author: Atrisha
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.special
import matplotlib.pyplot as plt
from code_utils import utils
from code_utils.utils import lighten_color
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import itertools
from matplotlib.ticker import MaxNLocator



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
        params = (10.5,3,.25)
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
    
    def gen_dist_utils(self,dist,parm=7):
        #parm = 7
        u = scipy.special.erf(1.75*(dist - parm) / (2 * math.sqrt(3)))
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
        prog_util = min(dist / 60, 1) if ag_type == 'agent_2' else min(dist / 17, 1) 
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
    
    def plot_only_safety(self):
        plt.figure()
        plt.title('safety util')
        plt.xlabel('distance gap (meters)')
        plt.ylabel('utility')
        X = [x for x in np.arange(0,20,.25)]
        Y = [self.gen_dist_utils(x,9) for x in np.arange(0,20,.25)]
        plt.plot(X,Y)
        plt.show()
                
    
    def plot(self):
        plt.figure()
        plt.title('safety util')
        plt.xlabel('distance gap (meters)')
        plt.ylabel('utility')
        X = [x for x in np.arange(0,20,.25)]
        Y = [self.calc_safe_payoff(x) for x in np.arange(0,20,.25)]
        plt.plot(X,Y)
        
        plt.figure()
        plt.title('progress util')
        plt.xlabel('trajectory length (meters)')
        plt.ylabel('utility')
        X = [x for x in np.arange(0,120,.25)]
        Y = [self.progress_payoff_dist(x,1) for x in np.arange(0,120,.25)]
        plt.plot(X,Y)
        plt.show()
        '''
        plt.figure()
        plt.title('combined util')
        plt.xlabel('trajectory length (meters)')
        plt.ylabel('utility')
        gamma_list = enumerate(np.linspace(start=-1, stop=1, num=5))
        for idx,g in gamma_list:
            X = [x for x in np.arange(0,100,.25)]
            Y = [self.combine_utils(self.progress_payoff_dist(x,1), self.calc_safe_payoff(x), g) for x in np.arange(0,100,.25)]
            f = (idx/(len(gamma_list)-1))*.75
            plt.plot(X,Y,c=lighten_color('red',f))
        '''
        Xs = np.linspace(start=0, stop=20, num=50)
        Ys = np.linspace(start=0, stop=120, num=50)
        xyz1, xyz2, xyz3 = [], [] , []
        for z in itertools.product(Xs,Ys):
            xyz1.append((z[0],z[1],self.combine_utils(self.progress_payoff_dist(z[1], 1), self.calc_safe_payoff(z[0]), 0)))
            #xyz2.append((z[0],z[1],self.combine_utils(self.progress_payoff_dist(z[1], 1), self.calc_safe_payoff(z[0]), 0)))
            #xyz3.append((z[0],z[1],self.combine_utils(self.progress_payoff_dist(z[1], 1), self.calc_safe_payoff(z[0]), 0.5)))
        Xs1 = [x[0] for x in xyz1]
        Ys1 = [x[1] for x in xyz1]
        Zs1 = [x[2] for x in xyz1]
        
        Xs2 = [x[0] for x in xyz2]
        Ys2 = [x[1] for x in xyz2]
        Zs2 = [x[2] for x in xyz2]
        
        Xs3 = [x[0] for x in xyz3]
        Ys3 = [x[1] for x in xyz3]
        Zs3 = [x[2] for x in xyz3]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_trisurf(Xs1, Ys1, Zs1, cmap=cm.YlGn, linewidth=0)
        #ax.plot_trisurf(Xs2, Ys2, Zs2, cmap=cm.Blues, linewidth=0)
        #ax.plot_trisurf(Xs3, Ys3, Zs3, cmap=cm.Purples, linewidth=0)
        #fig.colorbar(surf)
        ax.set_ylabel('trajectory length (meters)')
        ax.set_xlabel('distance gap (meters)')
        ax.set_zlabel('utility')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        
        fig.tight_layout()
        
        plt.show() # or:



       
if __name__ == '__main__':
    u = Utilities()
    u.plot_only_safety()
        