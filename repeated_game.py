'''
Created on Mar 9, 2021

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
import matplotlib.patches as patches
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sqlite3
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import math
import scipy.integrate
import copy
from collections import namedtuple 
from scipy.optimize import minimize, Bounds
from planners.trajectory_planner import TrajectoryPlanner, VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner
import random
import itertools
import scipy.special
from ext.convex_hull import ConvexHull
import simulation_sketch
from shapely.geometry import Polygon, LineString
from equilibrium.utilities import Utilities

Point = namedtuple('Point', 'x y')

veh_maneuvers = ['turn','wait']
ped_maneuver = ['walk','wait']
modes = ['normal','aggressive']

show_points = False



                    

cols = ['r','g','b','c']

def plot_action_space(fig_list,ax_list,outcome_map,util):
    
    fig,fig2,fig3 = fig_list[0],fig_list[1],fig_list[2]
    ax,ax2,ax3 = ax_list[0],ax_list[1],ax_list[2]
    fig = plt.figure()
    ax = Axes3D(fig)
    for windx,thresh in enumerate([[.5,.5],[.5,.2]]):
        idx = 0
        X,Y,Z = [],[],[]
        for k,v in outcome_map.items():
            combined_payoff_list = []
            print(k,cols[idx])
            safe_payoffs,prog_payoffs,combined_payoffs,all_pts_veh,all_pts_ped = [],[],[],[],[]
            
            for i in np.arange(2):
                #print('safety payoff',k,i,[util.exp_dist_payoffs(x) for x in v])
                ag_type = 'veh' if i==0 else 'ped'
                #combined_payoff_list = [(util.exp_dist_payoffs(x[2])+util.progress_payoff_dist(x[3+i],ag_type))/2 for x in v]
                combined_payoff_list = [util.combine_utils(x[2],util.progress_payoff_dist(x[3+i],ag_type),thresh[i]) for x in v]
                #_safe_payoff_list = [util.exp_dist_payoffs(x[2]) for x in v]
                    
                _safe_payoff_list = [x[2] for x in v]
                _prog_list = [util.progress_payoff_dist(x[3+i],ag_type) for x in v]
                if i == 0:
                    ax.scatter([x[0] for x in v], [x[1] for x in v], combined_payoff_list,color = lighten_color(cols[idx]),label=k, marker = '.')
                else:
                    ax.scatter([x[0] for x in v], [x[1] for x in v], combined_payoff_list,color = cols[idx],label=k, marker = '.')    
            
            idx += 1
    
        
    ax.set_xlabel('vehicle')
    ax.set_ylabel('pedestrian')
    ax.set_title('combined utils')
    


def plot_util_space(fig_list,ax_list,outcome_map,util):
    
    fig,fig2,fig3 = fig_list[0],fig_list[1],fig_list[2]
    ax,ax2,ax3 = ax_list[0],ax_list[1],ax_list[2]
    for windx,thresh in enumerate([[.5,.5],[.5,.2]]):
        idx = 0
        for k,v in outcome_map.items():
            
            print(k,cols[idx])
            safe_payoffs,prog_payoffs,combined_payoffs,all_pts_veh,all_pts_ped = [],[],[],[],[]
            for i in np.arange(2):
                #print('safety payoff',k,i,[util.exp_dist_payoffs(x) for x in v])
                ag_type = 'veh' if i==0 else 'ped'
                #_combined_payoff_list = [(util.exp_dist_payoffs(x[2])+util.progress_payoff_dist(x[3+i],ag_type))/2 for x in v]
                _combined_payoff_list = [util.combine_utils(x[2],util.progress_payoff_dist(x[3+i],ag_type),thresh[i]) for x in v]
                #_safe_payoff_list = [util.exp_dist_payoffs(x[2]) for x in v]
                _safe_payoff_list = [x[2] for x in v]
                _prog_list = [util.progress_payoff_dist(x[3+i],ag_type) for x in v]
                if show_points:
                    if ag_type == 'veh':
                        all_pts_veh = list(_combined_payoff_list)
                    else:
                        all_pts_ped = list(_combined_payoff_list)
                p_ranges_combined = [min(_combined_payoff_list),max(_combined_payoff_list)]
                combined_payoffs.append(p_ranges_combined)
                
                p_ranges_safe = [min(_safe_payoff_list),max(_safe_payoff_list)]
                safe_payoffs.append(p_ranges_safe)
                
                p_ranges_prog = [min(_prog_list),max(_prog_list)]
                prog_payoffs.append(p_ranges_prog)
                
            p_nodes_combined = list(itertools.product(combined_payoffs[0],combined_payoffs[1]))
            p_nodes_combined = [p_nodes_combined[0],p_nodes_combined[1],p_nodes_combined[3],p_nodes_combined[2]]
            
            p_nodes_safe = list(itertools.product(safe_payoffs[0],safe_payoffs[1]))
            p_nodes_safe = [p_nodes_safe[0],p_nodes_safe[1],p_nodes_safe[3],p_nodes_safe[2]]
            
            p_nodes_prog = list(itertools.product(prog_payoffs[0],prog_payoffs[1]))
            p_nodes_prog = [p_nodes_prog[0],p_nodes_prog[1],p_nodes_prog[3],p_nodes_prog[2]]
            
            
            if windx == 0:
                p = Polygon(p_nodes_combined, edgecolor = cols[idx], fill = False,label=k,ls='--')
            else:
                p = Polygon(p_nodes_combined, edgecolor = cols[idx], fill = False,label=k)
            ax.add_patch(p)
            
            if windx == 0:
                p1 = Polygon(p_nodes_safe, edgecolor = cols[idx], fill = False,label=k,ls='--')
            else:
                p1 = Polygon(p_nodes_safe, edgecolor = cols[idx], fill = False,label=k)
            ax2.add_patch(p1)
            
            print(p_nodes_safe)
            if windx == 0:
                p2 = Polygon(p_nodes_prog, edgecolor = cols[idx], fill = False,label=k,ls='--')
            else:
                p2 = Polygon(p_nodes_prog, edgecolor = cols[idx], fill = False,label=k)
            ax3.add_patch(p2)
            
            if show_points:
                ax.scatter(all_pts_veh,all_pts_ped,marker='.',c=cols[idx], label=k)
            
            if windx == 0:
                ax.legend()
                ax2.legend()
                ax3.legend()
        
            idx += 1
    
        
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_xlabel('vehicle')
    ax.set_ylabel('pedestrian')
    ax.set_title('combined utils')
    
    ax2.set_xlim([-1.2,1.2])
    ax2.set_ylim([-1.2,1.2])
    ax2.set_xlabel('vehicle')
    ax2.set_ylabel('pedestrian')
    ax2.set_title('safety utils')
    
    ax3.set_xlim([-1.2,1.2])
    ax3.set_ylim([-1.2,1.2])
    ax3.set_xlabel('vehicle')
    ax3.set_ylabel('pedestrian')
    ax3.set_title('progress utils')




def analyse_br():
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
    
def analyze_max_min_resp():
    u = Utilities()
    conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
    c = conn.cursor()
    ''' pedestrian's maxmin response '''
    q_string = "SELECT A.TRAJ_1_ID, A.TRAJ_2_ID, B.MANEUVER, B.MANEUVER_MODE, C.MANEUVER, C.MANEUVER_MODE, A.DISTANCE_GAP, A.TRAJ_1_LENGTH, A.TRAJ_2_LENGTH \
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
        '''
        ped_manv = x[4]
        veh_traj_length = x[7]
        ped_traj_length = x[8]
        '''
        resp_vect = [(x[4],x[7],u.combine_utils(u.progress_payoff_dist(x[8], 1), u.calc_safe_payoff(x[6]), 0.5)) for x in int_v]
        resp_vect.sort(key=lambda tup: tup[-1])
        X_lb_ub.append((p_fv,resp_vect[0][1],resp_vect[0][2]))
        
    X_lb_ub.sort(key=lambda tup: tup[0])
    plt.figure()
    plt.plot([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub])
    plt.title('worst case veh traj choice for ped\'s traj choice')
    plt.figure()
    plt.plot([x[0] for x in X_lb_ub],[x[2] for x in X_lb_ub])
    plt.title('min util for traj choice (pedestrian)')
    
    
    ''' vehicle maxmin to pedestrian trajectory choice'''
    
    q_string = "SELECT A.TRAJ_1_ID, A.TRAJ_2_ID, B.MANEUVER, B.MANEUVER_MODE, C.MANEUVER, C.MANEUVER_MODE, A.DISTANCE_GAP, A.TRAJ_1_LENGTH, A.TRAJ_2_LENGTH \
                FROM TRAJ_INTERACTIONS AS A INNER JOIN TRAJECTORY_METADATA AS B on A.TRAJ_1_ID = B.TRAJ_ID INNER JOIN TRAJECTORY_METADATA AS C on A.TRAJ_2_ID = C.TRAJ_ID \
                "
    c.execute(q_string)
    res = c.fetchall()
    interac_dict = dict()
    for row in res:
        if row[7] not in interac_dict:
            interac_dict[row[7]] = []
        interac_dict[row[7]].append(row)
    X_lb_ub = []
    ct,N = 0,len(interac_dict)
    for p_fv,int_v in interac_dict.items():
        ct += 1
        print('vehicle responding',ct,'/',N)
        '''
        veh_manv = x[2]
        veh_traj_length = x[7]
        ped_traj_length = x[8]
        '''
        resp_vect = [(x[2],x[8],u.combine_utils(u.progress_payoff_dist(x[7], 0), u.calc_safe_payoff(x[6]), 0.1)) for x in int_v]
        resp_vect.sort(key=lambda tup: tup[-1])
        X_lb_ub.append((p_fv,resp_vect[0][1],resp_vect[0][2]))
        
    X_lb_ub.sort(key=lambda tup: tup[0])
    plt.figure()
    plt.plot([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub])
    plt.title('worst case pedest traj choice for veh\'s traj choice')
    plt.figure()
    plt.plot([x[0] for x in X_lb_ub],[x[2] for x in X_lb_ub])
    plt.title('min util for traj choice (vehicle)')
    plt.show()
    
def analyse_only_util_br():
    u = Utilities()
    conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
    c = conn.cursor()
    ''' pedestrian's best response to vehicle trajectory choice '''
    q_string = "SELECT A.TRAJ_1_ID, A.TRAJ_2_ID, B.MANEUVER, B.MANEUVER_MODE, C.MANEUVER, C.MANEUVER_MODE, A.DISTANCE_GAP, A.TRAJ_1_LENGTH, A.TRAJ_2_LENGTH, B.FINAL_VEL, C.FINAL_VEL \
                FROM TRAJ_INTERACTIONS AS A \
                INNER JOIN TRAJECTORY_METADATA AS B on A.TRAJ_1_ID = B.TRAJ_ID \
                INNER JOIN TRAJECTORY_METADATA AS C on A.TRAJ_2_ID = C.TRAJ_ID"
    c.execute(q_string)
    res = c.fetchall()
    interac_dict = dict()
    for row in res:
        veh_util = round(u.combine_utils(u.progress_payoff_dist(row[7], 0), u.calc_safe_payoff(row[6]), 0.5),2)
        if veh_util not in interac_dict:
            interac_dict[veh_util] = []
        interac_dict[veh_util].append(row)
    X_lb_ub = []
    for p_fv,int_v in interac_dict.items():
        resp_vect = [(x[2],u.combine_utils(u.progress_payoff_dist(x[8], 1), u.calc_safe_payoff(x[6]), 0.1)) for x in int_v]
        resp_vect.sort(key=lambda tup: tup[-1], reverse=True)
        thresh_tup = (resp_vect[0][0],resp_vect[0][1])
        brk_idx = 0
        for ridx in np.arange(1,len(resp_vect)):
            if resp_vect[ridx][0] != thresh_tup[0] and resp_vect[ridx][1] < thresh_tup[1]:
                brk_idx = ridx
                break
            else:
                thresh_tup = (resp_vect[ridx][0], resp_vect[ridx][1])
        X_lb_ub.append((p_fv,resp_vect[0][1],resp_vect[brk_idx][1]))
    X_lb_ub.sort(key=lambda tup: tup[0])
    plt.figure()
    plt.title('pedestrian BR')
    plt.plot([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub],c='green')
    plt.plot([x[0] for x in X_lb_ub],[x[2] for x in X_lb_ub],c='blue')
    
    
    c.execute(q_string)
    res = c.fetchall()
    interac_dict = dict()
    for row in res:
        ped_util = round(u.combine_utils(u.progress_payoff_dist(row[8], 1), u.calc_safe_payoff(row[6]), 0.1),)
        if ped_util not in interac_dict:
            interac_dict[ped_util] = []
        interac_dict[ped_util].append(row)
    X_lb_ub = []
    for p_fv,int_v in interac_dict.items():
        resp_vect = [(x[2],u.combine_utils(u.progress_payoff_dist(x[7], 0), u.calc_safe_payoff(x[6]), 0.5)) for x in int_v]
        resp_vect.sort(key=lambda tup: tup[-1], reverse=True)
        thresh_tup = (resp_vect[0][0],resp_vect[0][1])
        brk_idx = 0
        for ridx in np.arange(1,len(resp_vect)):
            if resp_vect[ridx][0] != thresh_tup[0] and resp_vect[ridx][1] < thresh_tup[1]:
                brk_idx = ridx
                break
            else:
                thresh_tup = (resp_vect[ridx][0], resp_vect[ridx][1])
        X_lb_ub.append((p_fv,resp_vect[0][1],resp_vect[brk_idx][1]))
    X_lb_ub.sort(key=lambda tup: tup[0])
    plt.figure()
    plt.title('vehicle BR')
    plt.plot([x[0] for x in X_lb_ub],[x[1] for x in X_lb_ub],c='green')
    plt.plot([x[0] for x in X_lb_ub],[x[2] for x in X_lb_ub],c='blue')
    
    
    plt.show()
        
        
                
                
        
    
        

if __name__ == '__main__':    
    '''
    acts = Actions()
    acts.generate_actions()

    util = acts.utils
    
    fig,ax = plt.subplots()
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    
    
    #plot_util_space([fig,fig2,fig3], [ax,ax2,ax3], acts.dist_gap_map)
    plot_action_space([fig,fig2,fig3], [ax,ax2,ax3], acts.dist_gap_map)   
    plt.show()
    '''
    
    '''
    acts = Actions()
    acts.generate_actions(0,5,1.38,6,True)
    acts.insert_interaction_data()
    '''
    
    analyse_br()
    '''
    init_veh_vel = 5
    init_ped_vel = 1.38
    veh_trajs,ped_trajs = dict(), dict()
    veh_waypoint = NYCMapInfo.veh_centerline
    veh_waypoint_velocity = [(init_veh_vel,),(None,),(1,init_veh_vel),(None,),(init_veh_vel,10)]
    ped_waypoint = NYCMapInfo.ped_centerline
    ped_waypoint_velocity = [(init_ped_vel,),(init_ped_vel,1.8),(init_ped_vel,1.8)]
    
    for veh_m in veh_maneuvers:
        veh_motion = VehicleTrajectoryPlanner(veh_waypoint,veh_waypoint_velocity,veh_m,None)
        veh_motion.generate_trajectory(True)
        veh_trajs[veh_m] = veh_motion.all_trajectories
    

    for ped_m in ped_maneuver:
        ped_motion = PedestrianTrajectoryPlanner(ped_waypoint,ped_waypoint_velocity,ped_m,None)
        ped_motion.generate_trajectory(True)
        ped_trajs[ped_m] = ped_motion.all_trajectories
    '''
    
    
    