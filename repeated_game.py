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
from maps.map_info import NYCMapInfo
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

Point = namedtuple('Point', 'x y')

veh_maneuvers = ['turn','wait']
ped_maneuver = ['walk','wait']
modes = ['normal','aggressive']

show_points = False



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
        if ag_type == 'veh':
            prog_util = min(dist / 20, 1)
        else:
            prog_util =  min(dist / 12, 1)
        #print(ag_type,dist,prog_util)
        return prog_util
        
    def combine_utils(self,prog_util,safe_util,thresh):
        if safe_util < thresh:
            return safe_util
        else:
            return prog_util
        
        
class Actions:
    
    def generate_agent_action(self,init_time,init_veh_vel,waypoint,waypoint_vels,manv,ag,horizon):
        trajs = dict()
        if ag == 'vehicle':
            motion = VehicleTrajectoryPlanner(waypoint,waypoint_vels,manv,None,horizon)
            motion.generate_trajectory(True)
            trajs[manv] = motion.all_trajectories
        else:
            motion = PedestrianTrajectoryPlanner(waypoint,waypoint_vels,manv,None,horizon)
            motion.generate_trajectory(True)
            trajs[manv] = motion.all_trajectories
        return trajs
                
        
    
    def generate_actions(self,init_time,init_veh_vel,init_ped_vel,horizon, insert_into_db = False):
        
        
        veh_trajs,ped_trajs = dict(), dict()
        veh_waypoint = NYCMapInfo.veh_centerline
        veh_waypoint_velocity = [(init_veh_vel,),(None,),(1,init_veh_vel),(None,),(init_veh_vel,10)]
        ped_waypoint = NYCMapInfo.ped_centerline
        ped_waypoint_velocity = [(init_ped_vel,),(init_ped_vel,1.8),(init_ped_vel,1.8)]
        
        for veh_m in veh_maneuvers:
            veh_motion = VehicleTrajectoryPlanner(veh_waypoint,veh_waypoint_velocity,veh_m,None,horizon)
            veh_motion.generate_trajectory(True)
            veh_trajs[veh_m] = veh_motion.all_trajectories
        
    
        for ped_m in ped_maneuver:
            ped_motion = PedestrianTrajectoryPlanner(ped_waypoint,ped_waypoint_velocity,ped_m,None,horizon)
            ped_motion.generate_trajectory(True)
            ped_trajs[ped_m] = ped_motion.all_trajectories
        
        if insert_into_db:
            parent_traj_id = None
            conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
            c = conn.cursor()
            i_string = 'INSERT INTO TRAJECTORIES VALUES (?,?,?,?,?,?,?,?,?)'
            i_string_tj_mtdata = 'INSERT INTO TRAJECTORY_METADATA VALUES (?,?,?,?,?,?,?,?,?,?,?)'
            traj_id = 1
            for ag_type_idx,traj_det_dict in enumerate([veh_trajs,ped_trajs]):
                ag_type = 'vehicle' if ag_type_idx == 0 else 'pedestrian'
                trajs, traj_metadata = [],[]
                for traj_manv,tm_v in traj_det_dict.items():
                    for traj_mode,tmd_v in tm_v.items():
                        for trj in tmd_v:
                            traj_entry = [(traj_id,float(x[1]),float(x[2]),float(x[3]),float(x[4]),x[6],x[0],x[7],None) for x in trj] 
                            traj_mtdt_entry = [(traj_id,float(trj[0][1]),float(trj[0][2]),float(trj[0][3]),float(trj[0][4]),float(trj[-1][3]),traj_manv,traj_mode,ag_type,init_time,parent_traj_id)]
                            traj_metadata.extend(traj_mtdt_entry)
                            trajs.extend(traj_entry)
                            traj_id += 1
                c.executemany(i_string,trajs)
                c.executemany(i_string_tj_mtdata,traj_metadata)
                print('agent trajectories inserted')           
            
            
            conn.commit()
            conn.close()        
            f=1  
        return veh_trajs, ped_trajs
    
    def insert_interaction_data(self):
        conn = sqlite3.connect('D:\\repeated_games_data\\right_turn_data.db')
        c = conn.cursor()
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='vehicle'"
        c.execute(q_string)
        veh_trajids = c.fetchall()
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID FROM TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.AGENT_TYPE='pedestrian'"
        c.execute(q_string)
        ped_trajids = c.fetchall()
        all_trajs = dict()
        q_string = "SELECT * FROM TRAJECTORIES ORDER BY TRACK_ID,TIME"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[0] not in all_trajs:
                all_trajs[row[0]] = []
            all_trajs[row[0]].append(row[1:7])
        utils = Utilities()
        interac_id = 1
        interac_data = []
        ct,N = 0, len(veh_trajids)*len(ped_trajids)
        for vt_id,pt_id in itertools.product(veh_trajids,ped_trajids):
            ct += 1
            
            vt = all_trajs[vt_id[0]]
            pt = all_trajs[pt_id[0]]
            d_g = utils.calc_dist_gap(vt, pt)
            vt_l = utils.calc_traj_length(vt)
            pt_l = utils.calc_traj_length(pt)
            interac_entry = [(interac_id,vt_id[0],pt_id[0],d_g,None,vt_l,pt_l)]
            interac_data.extend(interac_entry)
            print('added',ct,'/',N,interac_entry)
            interac_id += 1
        i_string = 'INSERT INTO TRAJ_INTERACTIONS VALUES (?,?,?,?,?,?,?)'
        c.executemany(i_string,interac_data)
        conn.commit()
        conn.close()    
        
    
    def calc_dist_gaps(self):
        init_veh_vel,init_ped_vel = 5, 1.38
        veh_trajs, ped_trajs = self.generate_actions(init_veh_vel,init_ped_vel)
        utils = Utilities()
        util_info = dict()
        dist_gap_map = dict()
        vtidx,ptidx = 0,0
        for veh_m in veh_maneuvers:
            for ped_m in ped_maneuver:
                #Xp,Yp,Xv,Yv = [],[],[],[]
                for m in itertools.product(modes,modes):
                    if m[0] not in veh_trajs[veh_m] or m[1] not in ped_trajs[ped_m]:
                        continue  
                    v_tjs = veh_trajs[veh_m][m[0]]
                    p_tjs = ped_trajs[ped_m][m[1]]
                    
                    for vt in v_tjs:
                        vtidx += 1
                        for pt in p_tjs:
                            ptidx += 1
                            if (veh_m,ped_m) not in dist_gap_map:
                                dist_gap_map[(veh_m,ped_m)] = []
                            d_g = utils.generate_safety_utils(pt, vt)
                            
                            if veh_m == 'turn' and ped_m == 'walk' and d_g > 5:
                                #simulation_sketch.simulate_trajectory(pt, vt,veh_m,ped_m)
                                #plt.plot([x[0] for x in vt],[x[3] for x in vt])
                                #plt.show()
                                pass
                            dist_gap_map[(veh_m,ped_m)].append((vtidx,ptidx,d_g,math.hypot(vt[0][1]-vt[-1][1], vt[0][2]-vt[-1][2]),math.hypot(pt[0][1]-pt[-1][1], pt[0][2]-pt[-1][2])))
                            '''
                            Xv.append(utils.generate_safety_utils(pt,vt))
                            Yv.append(utils.progress_payoff_dist(math.hypot(vt[0][1]-vt[-1][1], vt[0][2]-vt[-1][2]), 'veh'))
                            Xp.append(utils.generate_safety_utils(pt,vt))
                            Yp.append(utils.progress_payoff_dist(math.hypot(pt[0][1]-pt[-1][1], pt[0][2]-pt[-1][2]), 'ped'))
                            '''
                '''
                plt.figure()
                plt.plot(Xv,Yv,'.')
                plt.title('veh '+str(veh_m)+str(ped_m))
                plt.figure()
                plt.plot(Xp,Yp,'.')
                plt.title('ped '+str(veh_m)+str(ped_m))
                plt.show()
                '''     
        for k,v in dist_gap_map.items():
            u_range = [min([x[2] for x in v]),max([x[2] for x in v])]
            print(k," : ",str(u_range))
            util_info[k] = u_range
        utils.util_info = util_info
        self.utils = utils
        self.dist_gap_map = dist_gap_map
        
        '''           
        for ped_traj in ped_motion.all_trajectories:
            for veh_traj in veh_motion.all_trajectories:
                dist_gap = utils.generate_safety_utils(ped_traj, veh_traj)
                if (veh_m,ped_m) not in util_info:
                    util_info[(veh_m,ped_m)] = []
                util_info[(veh_m,ped_m)].append(dist_gap)
        print('utils ranges (dist gaps) ------------- ')
        for k,v in util_info.items():
            u_range = [min(v),max(v)]
            print(k," : ",str(u_range))   
        '''
    
    '''
    utils = Utilities()    
    util_info = {('turn', 'walk')  :  [1.453167769746284, 2.430105932669038],
                ('turn', 'wait')  :  [4.383906901195861, 5.968458852783996],
                ('wait', 'walk')  :  [2.87153787003464, 9.258464652266861],
                ('wait', 'wait')  :  [6.546044963774584, 15.797224181634268]}
    utils.util_info = util_info
    '''               



                    

cols = ['r','g','b','c']
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

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
    
    
    