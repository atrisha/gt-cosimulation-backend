'''
Created on Mar 22, 2021

@author: Atrisha
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sqlite3
from shapely.geometry import Polygon, LineString
from equilibrium.utilities import Utilities
from code_utils.utils import lighten_color

class SatisficingEquilibriaCompleteGame():
    
    def __init__(self, input_struc):
        
        self.input_struc = input_struc
        
    
    '''
        This takes in a key value pair of (m,t_idx) : u_i for a give -i action.
        Generates the best response set modulo manuver.
        Also generates the utility interval for the best response set.
    '''
    def calc_equilibria(self,util_dict):
        u = Utilities()
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        ''' pedestrian's best response to vehicle trajectory choice '''
        q_string = "SELECT A.TRAJ_1_ID, A.TRAJ_2_ID, B.MANEUVER, B.MANEUVER_MODE, C.MANEUVER, C.MANEUVER_MODE, A.DISTANCE_GAP, A.TRAJ_2_LENGTH, A.TRAJ_1_LENGTH, A.TRAJ_2_LENGTH \
                    FROM TRAJ_INTERACTIONS AS A INNER JOIN TRAJECTORY_METADATA AS B on A.TRAJ_1_ID = B.TRAJ_ID INNER JOIN TRAJECTORY_METADATA AS C on A.TRAJ_2_ID = C.TRAJ_ID \
                    "
        c.execute(q_string)
        res = c.fetchall()
        interac_dict = dict()
        for row in res:
            #ttxp = 
            if row[8] not in interac_dict:
                interac_dict[row[8]] = []
            interac_dict[row[8]].append(row)
        X_lb_ub = []
        ct,N = 0,len(interac_dict)
        for p_fv,int_v in interac_dict.items():
            
            ct += 1
            print('pedestrian responding',ct,'/',N)
            resp_vect = [(x[2],x[9],u.combine_utils(u.progress_payoff_dist(x[7], 1), u.calc_safe_payoff(x[6]), 0.5)) for x in int_v]
            resp_vect.sort(key=lambda tup: tup[-1], reverse=True)
            thresh_tup = (resp_vect[0][0],resp_vect[1][2], resp_vect[0][2])
            brk_idx = 0
            for ridx in np.arange(1,len(resp_vect)):
                if resp_vect[ridx][0] != thresh_tup[0] and resp_vect[ridx][2] < resp_vect[0][2]:
                    brk_idx = max(ridx-1,0)
                    break
                else:
                    thresh_tup = (resp_vect[ridx][0], resp_vect[ridx][1], resp_vect[ridx][2])
            X_lb_ub.append((p_fv,resp_vect[0][1],resp_vect[brk_idx][1]))
            
        X_lb_ub.sort(key=lambda tup: tup[0])
        plt.figure()
        plt.plot([x[1] for x in X_lb_ub],[x[0] for x in X_lb_ub],c='red')
        plt.plot([x[2] for x in X_lb_ub],[x[0] for x in X_lb_ub],c=lighten_color('red', .5),label = 'pedestrian best response')
        if [x[1] for x in X_lb_ub] == [x[2] for x in X_lb_ub]:
            print('bounds are equal')
            p1 = (LineString(list(zip([x[1] for x in X_lb_ub],[x[0] for x in X_lb_ub]))) , )
        else:
            print('bounds are unequal')
            p1 = (LineString(list(zip([x[1] for x in X_lb_ub],[x[0] for x in X_lb_ub]))) , LineString(list(zip([x[2] for x in X_lb_ub],[x[0] for x in X_lb_ub]))))
        
        ''' vehicle best reponse to pedestrian trajectory choice'''
        
        q_string = "SELECT A.TRAJ_1_ID, A.TRAJ_2_ID, B.MANEUVER, B.MANEUVER_MODE, C.MANEUVER, C.MANEUVER_MODE, A.DISTANCE_GAP, A.TRAJ_1_LENGTH, A.TRAJ_1_LENGTH, A.TRAJ_2_LENGTH \
                    FROM TRAJ_INTERACTIONS AS A INNER JOIN TRAJECTORY_METADATA AS B on A.TRAJ_1_ID = B.TRAJ_ID INNER JOIN TRAJECTORY_METADATA AS C on A.TRAJ_2_ID = C.TRAJ_ID \
                    "
        c.execute(q_string)
        res = c.fetchall()
        interac_dict = dict()
        for row in res:
            if row[9] not in interac_dict:
                interac_dict[row[9]] = []
            interac_dict[row[9]].append(row)
        X_lb_ub = []
        ct,N = 0,len(interac_dict)
        for p_fv,int_v in interac_dict.items():
            ct += 1
            print('vehicle responding',ct,'/',N)
            
            resp_vect = [(x[2],x[8],u.combine_utils(u.progress_payoff_dist(x[7], 0), u.calc_safe_payoff(x[6]), 0.1)) for x in int_v]
            resp_vect.sort(key=lambda tup: tup[-1], reverse=True)
            thresh_tup = (resp_vect[0][0],resp_vect[1][2], resp_vect[0][2])
            brk_idx = 0
            for ridx in np.arange(1,len(resp_vect)):
                if resp_vect[ridx][0] != thresh_tup[0] and resp_vect[ridx][2] < resp_vect[0][2]:
                    if p_fv > 1.4:
                        brk = 1
                    brk_idx = max(ridx-1,0)
                    break
                else:
                    thresh_tup = (resp_vect[ridx][0], resp_vect[ridx][1], resp_vect[ridx][2])
            X_lb_ub.append((p_fv,resp_vect[0][1],resp_vect[brk_idx][1]))
            
        X_lb_ub.sort(key=lambda tup: tup[0])
        #plt.figure()
        plt.plot([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub],c='blue')
        plt.plot([x[0] for x in X_lb_ub],[x[2] for x in X_lb_ub],c=lighten_color('blue', .5),label = 'vehicle best response')
        if [x[1] for x in X_lb_ub] == [x[2] for x in X_lb_ub]:
            print('bounds are equal')
            p2 = (LineString(list(zip([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub]))) , )
        else:
            print('bounds are unequal')
            p2 = (LineString(list(zip([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub]))), LineString(list(zip([x[0] for x in X_lb_ub],[x[2] for x in X_lb_ub]))))
            
        if p1[0].intersects(p2[0]):
            print('Equilibrium exists')
            eq_pt_ubub = p1[0].intersection(p2[0])
            lX,lY = p1[0].xy
            print(eq_pt_ubub)
        else:
            print('Equilibrium doesn\'t exists')
             
        #eq_reg = p1.intersection(p2)
        #print(eq_reg)
        '''
        fig = plt.figure()
        ax = fig.add_subplot(121)
        
        for ob in eq_reg:
            x, y = ob.xy
            if len(x) == 1:
                ax.plot(x, y, 'o', color='BLUE', zorder=2)
            else:
                ax.plot(x, y, color='BLUE', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        '''
        plt.show()
        
        