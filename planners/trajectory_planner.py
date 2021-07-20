'''
Created on Mar 5, 2021

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.animation as animation
import math
import scipy.integrate
from maps.map_info import NYCMapInfo
import copy
from scipy.optimize import minimize, Bounds
import random
import itertools
from collections import defaultdict
import warnings
import constants
import all_utils.utils


class TrajectoryConstraints:
    
    
    ''' from https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list'''
    def list_duplicates(self,seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return ((key,locs) for key,locs in tally.items() 
                                if len(locs)>1)

     
    
    def _remove_duplicate(self,path):
        dup_indxs = []
        for dup in sorted(self.list_duplicates([x[0] for x in path])):
            dup_indxs += dup[1][1:]
        _newpath = [x for idx,x in enumerate(path) if idx not in dup_indxs]
        return _newpath,dup_indxs
    
    
    def __init__(self, init_vel,waypoints):
        unq_waypoints,dup_indxs = self._remove_duplicate(waypoints)
        self.waypoints = unq_waypoints
        self.init_vel = init_vel
        self.dup_indxs = dup_indxs
        
        
    def set_limit_constraints(self,max_lat_acc_lims=5.6,max_vel_lims=22,max_acc_lims=5,max_jerk_lims=2):
        self.max_lat_acc_lims = max_lat_acc_lims
        self.max_vel_lims = max_vel_lims
        self.max_acc_lims = max_acc_lims
        self.max_jerk_lims = max_jerk_lims
        
class WaitTrajectoryConstraints(TrajectoryConstraints):
    ''' 
    stop_horizon_dist_sampling_range = (minimum distance in meters where the vehicle can stop, maximum distance in meters where the vehicle can stop)
    stop_horizon_time_sampling_range = (minimum time in seconds when the vehicle can stop, maximum time in seconds when the vehicle can stop)
    '''
    def __init__(self,init_vel,waypoints,stop_horizon_dist_sampling_range,stop_horizon_time_sampling_range):
        TrajectoryConstraints.__init__(self, init_vel,waypoints)
        self.stop_horizon_dist_sampling_range = stop_horizon_dist_sampling_range
        self.stop_horizon_time_sampling_range = stop_horizon_time_sampling_range

class ProceedTrajectoryConstraints(TrajectoryConstraints):
    '''
    waypoint_vel_sampling_range = [(minimum velocity*,),...,(minimum velocity,maximum velocity),..,(minimum velocity*,maximum velocity*)]
    * : mandatory
    optional values can be passed as (None,)
    should be sample length as the waypoints
    '''
    def __init__(self,waypoints,waypoint_vel_sampling_range):
        try:
            assert len(waypoints) == len(waypoint_vel_sampling_range) , "Lengths should be same"
        except AssertionError:
            f=1
            raise
        init_vel = waypoint_vel_sampling_range[0][0]
        TrajectoryConstraints.__init__(self,init_vel, waypoints)
        waypoint_vel_sampling_range = [x for idx,x in enumerate(waypoint_vel_sampling_range) if idx not in self.dup_indxs]
        self.waypoint_vel_sampling_range = waypoint_vel_sampling_range
        #if len(waypoint_vel_sampling_range) > 2 and all_equal(waypoint_vel_sampling_range[1:-1]) and waypoint_vel_sampling_range[1] == (None,):
        #    self.waypoint_vel_sampling_range = waypoint_vel_sampling_range[0:1] + [(None,) if i != len(np.arange(1,len(self.waypoints)-1))//2 else (np.mean([waypoint_vel_sampling_range[0][0],waypoint_vel_sampling_range[-1][0]]),np.mean([waypoint_vel_sampling_range[0][0],waypoint_vel_sampling_range[-1][1]])) for i in np.arange(1,len(self.waypoints)-1)] + waypoint_vel_sampling_range[-1:]
        assert len(self.waypoints) == len(self.waypoint_vel_sampling_range) , "Lengths should be same, "+str(len(self.waypoints))+","+str(len(self.waypoint_vel_sampling_range))


def find_maxima_minima(max,cs_v,thresh,guess=3):
    
    #lc1 = LinearConstraint(np.ones((1,1)),0,thresh)
    lc1 = Bounds(0,thresh)
    if not max:
        res = minimize(cs_v, x0=[guess], method='TNC', bounds=lc1)
    else:
        f = lambda x : -cs_v(x)
        res = minimize(f, x0=[guess], method='TNC', bounds=lc1)
    
    return res.x

def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

class TriangulationCurve:
    
    def __init__(self,v0,stop_horizon,N):
        self.v0 = v0
        self.b = stop_horizon
        self.a = 0
        curves = []
        if N == 1:
            self.c = stop_horizon/2
            curves = [copy.deepcopy(self)]
        else:
            _cs = np.linspace(0, stop_horizon, N)
            for _c in _cs:
                self.c = _c
                _curve = copy.deepcopy(self)
                curves.append(_curve)
        self._curves = curves
        
    def curves(self):
        return self._curves
    
    def __call__(self,_x):
        x = self.b - _x
        if x <= 0:
            return 0
        elif self.a < x <= self.c:
            return self.v0 * ((((x-self.a)**2) / ((self.b-self.a)*(self.c-self.a))))
        elif self.c < x <= self.b:
            return self.v0 * (1 - (((self.b-x)**2) / ((self.b-self.a)*(self.b-self.c))))
        else:
            return self.v0
        
    def derivative(self,deg):
        
        def _d(_x):
            x = self.b - _x
            if x < 0:
                return 0
            elif self.a <= x <= self.c:
                return self.v0 * ((2*(x-self.a))/((self.b-self.a)*(self.c-self.a)))
            elif self.c < x <= self.b:
                return self.v0 * ((2*(self.b-x))/((self.b-self.a)*(self.b-self.c)))
            else:
                return 0
            
        def _dd(_x):
            x = self.b - _x
            if x < 0:
                return 0
            elif self.a <= x <= self.c:
                return self.v0 * (2/((self.b-self.a)*(self.c-self.a)))
            elif self.c < x < self.b:
                return self.v0 * ((-2)/((self.b-self.a)*(self.b-self.c)))
            else:
                return 0
        
        if deg == 1:
            f = lambda x : _d(x)
            return f
        elif deg == 2:
            f = lambda x : _dd(x)
            return f
        else:
            raise Exception('Only second order derivative is implemented')
'''
plt.figure()    
tcs = TriangulationCurve(4,6,50)
X = np.linspace(0,6,1000)
for tc in tcs.curves():
    Y = [tc(x) for x in X]
    plt.plot(X,Y)
tcs = TriangulationCurve(4,5,50)
X = np.linspace(0,6,1000)
for tc in tcs.curves():
    Y = [tc(x) for x in X]
    plt.plot(X,Y)

plt.figure()

for tc in tcs.curves():
    tcd = tc.derivative(1)
    Y = [tcd(x) for x in X]
    plt.plot(X,Y)
plt.figure()
for tc in tcs.curves():
    Y = [tc(x) for x in X]
    accY = [0]+[(x[1]-x[0])*(1000/6) for x in zip(Y[:-1],Y[1:])]
    plt.plot(X,accY)
plt.show()
'''
    
WAIT_MANEUVERS = ['wait']        


class TrajectoryPlanner:
    
    show_plots = False
    print_console = False
    
    
    def __init__(self,traj_constr_obj, maneuver, mode, horizon):
        '''
        if maneuver in WAIT_MANEUVERS and isinstance(traj_constr_obj, ProceedTrajectoryConstraints):
            raise("Wait maneuvers should be passed WaitTrajectoryConstraints object")
        if maneuver not in WAIT_MANEUVERS and isinstance(traj_constr_obj, WaitTrajectoryConstraints):
            raise("Proceed maneuvers should be passed ProceedTrajectoryConstraints object")
        '''
        self.traj_constr_obj = traj_constr_obj
        self.v0 = traj_constr_obj.init_vel
        self.vel_pts = None
        self.centerline = traj_constr_obj.waypoints 
        self.maneuver = maneuver
        self.mode = mode
        self.horizon = horizon
        if hasattr(traj_constr_obj, 'parent_trajectory_arcl'):
            self.parent_trajectory_arcl = traj_constr_obj.parent_trajectory_arcl
        else:
            self.parent_trajectory_arcl = 0
    
    def build_velocity_lattice(self,vel_pts_range):
        v_ts = []
        vel_step = 0.3
        lattice_N = 10
        for rng in vel_pts_range:
            if len(rng) == 1:
                vtx = list(rng)
            else:
                if rng[0] is not None and rng[1] is not None:
                    vtx = np.linspace(rng[0], rng[1], lattice_N).tolist()
                elif rng[0] is None and rng[1] is not None:
                    vtx = np.linspace(0,rng[1],10).tolist()
                elif rng[0] is not None and rng[1] is None:
                    vtx = np.linspace(rng[0],rng[0]+(lattice_N*vel_step),lattice_N).tolist()
                else:
                    vtx = [None]
            v_ts.append(vtx)
        vel_profiles = list(itertools.product(*v_ts))
        f_dx = self.cs_x.derivative(1)
        f_dy = self.cs_y.derivative(1)
        f = lambda x : math.hypot(f_dx(x),f_dy(x))
        self.arcl = scipy.integrate.quad(f,0,1)[0]
        s_pts = [self.arcl*x for x in self.indx]
        self.all_velocity_profiles = []
        for vp in vel_profiles:
            _v = []
            for i,v in enumerate(vp):
                if v is not None:
                    _v.append(v)
                else:
                    nxt_valid_indx = next(i+idx for idx,item in enumerate(vp[i:]) if item is not None)
                    prev_valid_indx = next(i-idx for idx,item in enumerate(reversed(vp[:i+1])) if item is not None)
                    #prev_valid_prop = math.hypot(self.centerline[0][0]-self.centerline[prev_valid_indx][0], self.centerline[0][1]-self.centerline[prev_valid_indx][1]) / (math.hypot(self.centerline[prev_valid_indx][0]-self.centerline[nxt_valid_indx][0], self.centerline[prev_valid_indx][1]-self.centerline[nxt_valid_indx][1]) + math.hypot(self.centerline[0][0]-self.centerline[prev_valid_indx][0], self.centerline[0][1]-self.centerline[prev_valid_indx][1])) 
                    prev_valid_prop = (s_pts[i]-s_pts[prev_valid_indx])/(s_pts[nxt_valid_indx]-s_pts[prev_valid_indx])
                    intpl_v = prev_valid_prop*vp[prev_valid_indx] + (1-prev_valid_prop)*vp[nxt_valid_indx]
                    _v.append(max(0.1,intpl_v))
            self.all_velocity_profiles.append(_v)
        
    
    def generate_path(self):
        ''' 
        generate path with an index [0,1] that will 
        be later scaled to the arc length
        '''
        wp_l,wp_r = all_utils.utils.add_parallel(self.centerline, 1, 1)
        all_waypoints = [self.centerline,wp_l,wp_r]
        if len(self.centerline) > 4:
            _midpt_idx = int(len(self.centerline)//2)
            all_waypoints.append(self.centerline[:_midpt_idx]+wp_r[_midpt_idx:])
            all_waypoints.append(self.centerline[:_midpt_idx]+wp_l[_midpt_idx:])
            all_waypoints.append(wp_r[:_midpt_idx]+wp_l[_midpt_idx:])
            all_waypoints.append(wp_l[:_midpt_idx]+wp_r[_midpt_idx:])
        res_map = []    
        for wp_idx,wp in enumerate(all_waypoints):
            seg_l = [0]+ [math.hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]) for x in zip(wp[1:],wp[:-1])]
            seg_l = [sum(seg_l[:i+1]) for i,x in enumerate(seg_l)]
            indx = [seg_l[i]/seg_l[-1] for i in np.arange(len(wp))]
            xspl_order,yspl_order = 2,2
            if self.print_console:
                print(indx)
            if hasattr(self.traj_constr_obj, 'path_degree'):
                k = self.traj_constr_obj.path_degree
            else:
                k = 2
            if len(indx) > 2 and not (min([x[0] for x in wp]) == max([x[0] for x in wp])):
                try:
                    self.cs_x = UnivariateSpline(indx,[x[0] for x in wp],k=k)
                except ValueError:
                    print(indx,[x[0] for x in wp])
                    #plt.plot(indx,[x[0] for x in wp])
                    #plt.show()
                    raise
            else:
                #self.cs_x = interp1d(indx,[x[0] for x in wp])
                self.cs_x = UnivariateSpline(indx,[x[0] for x in wp],k=1)
                xspl_order = 1
            
            if len(indx) > 2 and not (min([x[1] for x in wp]) == max([x[1] for x in wp])):
                try:
                    _x = indx
                    _y = [x[1] for x in wp]
                    self.cs_y = UnivariateSpline(_x,_y,k=k)
                except:
                    raise
            else:
                #self.cs_y = interp1d(indx,[x[1] for x in wp])
                self.cs_y = UnivariateSpline(indx,[x[1] for x in wp],k=1)
                yspl_order = 1
            err_x = self.cs_x(0) - self.centerline[0][0]
            x_corrected = lambda x : self.cs_x(x) - err_x
            err_y = self.cs_y(0) - self.centerline[0][1]
            y_corrected = lambda x : self.cs_y(x) - err_y
            '''
            plt.plot([x_corrected(x) for x in np.linspace(indx[0],indx[-1],100)],[y_corrected(x) for x in np.linspace(indx[0],indx[-1],100)])
            if wp_idx == 0:
                plt.plot([x[0] for x in wp],[x[1] for x in wp],'kx')
            else:
                plt.plot([x[0] for x in wp],[x[1] for x in wp],'x')
            plt.show()        
            '''
            residuals = []
            for _i,i in enumerate(indx):
                _res = math.hypot(x_corrected(i)-self.centerline[_i][0], y_corrected(i)-self.centerline[_i][1])
                residuals.append(_res)
            #_max_res = max(residuals[:int(len(self.centerline)/2)])
            _max_res = max(residuals)
            res_map.append((_max_res,self.cs_x,self.cs_y))
            
        res_map.sort(key=lambda tup: tup[0])
        self.cs_x_list, self.cs_y_list = [x[1] for x in res_map], [x[2] for x in res_map]
        self.cs_x, self.cs_y = res_map[0][1],res_map[0][2]
        err_x = self.cs_x(0) - self.centerline[0][0]
        x_corrected = lambda x : self.cs_x(x) - err_x
        err_y = self.cs_y(0) - self.centerline[0][1]
        y_corrected = lambda x : self.cs_y(x) - err_y
        #print(res_map[0][0])
        #if res_map[0][0] > constants.CAR_WIDTH/2:
        #    warnings.warn(message = "Generated path "+str(res_map[0][0])+"m away. Tolerance was set to "+str(constants.CAR_WIDTH/2)+"m", category = UserWarning)
        self.path = [(x,self.cs_x(x),self.cs_y(x)) for x in indx]
        xdd = self.cs_x.derivative(2) if xspl_order == 2 else None
        ydd = self.cs_y.derivative(2) if yspl_order == 2 else None
        xd = self.cs_x.derivative(1)
        yd = self.cs_y.derivative(1)
        plot_indx_x = np.linspace(indx[0],indx[-1],100)
        f_dx = self.cs_x.derivative(1)
        f_dy = self.cs_y.derivative(1)
        f = lambda x : math.hypot(f_dx(x),f_dy(x))
        self.arcl = scipy.integrate.quad(f,0,1)[0]
        if xdd is not None and ydd is not None:
            self.curvature = lambda x: abs(((xd(x)/yd(x))*(ydd(x)/xd(x)**2) - (yd(x)/xd(x))*(xdd(x)/yd(x)**2))) / np.power((xd(x)/yd(x))** 2 + (yd(x)/xd(x))** 2, 3 / 2)
        else:
            self.curvature = lambda x: 0
        if False:
            plt.figure()
            plt.title('path')
            plt.plot([x_corrected(x) for x in plot_indx_x],[y_corrected(x) for x in plot_indx_x])
            v = list(zip([x_corrected(x) for x in plot_indx_x],[y_corrected(x) for x in plot_indx_x]))
            plt.arrow(v[-2][0], v[-2][1],v[-1][0]-v[-2][0] , v[-1][1]-v[-2][1], width=.5)
            plt.axis('equal')
            plt.plot([x[0] for x in self.centerline],[x[1] for x in self.centerline],'x')
            
            plt.figure()
            plt.title('curvature')
            plt.plot(plot_indx_x,[self.curvature(x) for x in plot_indx_x])
            
            plt.show()
        self.indx = indx
        return self.path
    
    def generate_trajectory(self,all=None):
        horizon = self.horizon
        self.generate_path()
        if self.v0 == 0 and isinstance(self.traj_constr_obj, WaitTrajectoryConstraints):
            yaw = math.atan2(self.cs_y.derivative()(0), self.cs_x.derivative()(0))
            traj = []
            all_trajs = {'aggressive':[],'normal':[]}
            for tx in np.arange(0,self.horizon+0.1,.1):
                traj.append((tx,self.traj_constr_obj.waypoints[0][0],self.traj_constr_obj.waypoints[0][1],0,0,0,0,yaw,self.parent_trajectory_arcl))
            all_trajs['aggressive'].append(list(traj))
            all_trajs['normal'].append(list(traj))
            self.all_trajectories = all_trajs
            return all_trajs
        if isinstance(self.traj_constr_obj, ProceedTrajectoryConstraints):
            self.build_velocity_lattice(self.traj_constr_obj.waypoint_vel_sampling_range)
        if isinstance(self.traj_constr_obj, ProceedTrajectoryConstraints):
            self.generate_proceed_velocity_profiles()
        else:
            self.generate_wait_velocity_profiles()
        
        
        all_trajs = dict()
        if self.show_plots:
            plt.figure()
        for m,v_profiles in self.velocity_profiles.items():
            for vp_idx,v in enumerate(v_profiles):
                if all is not None and not all and vp_idx != int(len(v_profiles)//2):
                    continue  
                traj = []
                self.cs_v = v['func']
                
                if isinstance(self.cs_v, TriangulationCurve):
                    ord = 3
                else:
                    ord = len(self.cs_v._eval_args)-1
                if ord > 1:
                    self.cs_a = self.cs_v.derivative(1)
                else:
                    self.cs_a = lambda x : 0
                if ord > 2:
                    self.cs_j = self.cs_v.derivative(2)
                else:
                    self.cs_j = lambda x : 0
            
                stopped_traj = False
                time_st = np.arange(0,horizon+.1,.1)
                if 'target vels' in v:
                    err = self.cs_v(0) - v['target vels'][0]
                    v_corrected = lambda x : abs(self.cs_v(x) - err)
                else:
                    v_corrected = self.cs_v
                
                if hasattr(self.traj_constr_obj,'lateral_path_sampling') and self.traj_constr_obj.lateral_path_sampling:
                    p_idx = np.random.randint(low=0, high=len(self.cs_x_list))
                    self.cs_x = self.cs_x_list[p_idx]
                    self.cs_y = self.cs_y_list[p_idx]
                
                f_dx = self.cs_x.derivative(1)
                f_dy = self.cs_y.derivative(1)
                f = lambda x : math.hypot(f_dx(x),f_dy(x))
                self.arcl = scipy.integrate.quad(f,0,1)[0]
                
                err_x = self.cs_x(0) - self.centerline[0][0]
                x_corrected = lambda x : self.cs_x(x) - err_x
                err_y = self.cs_y(0) - self.centerline[0][1]
                y_corrected = lambda x : self.cs_y(x) - err_y
                self.x_corrected = x_corrected
                self.y_corrected = y_corrected
        
                for t in time_st:
                    if t <= horizon:
                        #s = self.t_s_map[t]
                        #s = self.cs_v(t) * t + 0.5 * abs(self.cs_a(t)) * t**2
                        s = scipy.integrate.quad(v_corrected,0,t)[0]
                        if isinstance(self.traj_constr_obj, WaitTrajectoryConstraints) and v_corrected(t)==0:
                            stopped_traj = True
                        if not stopped_traj:
                            #math.atan2(self.cs_y(s/self.arcl)-traj[-1][2], self.cs_x(s/self.arcl)-traj[-1][1])
                            yaw = math.atan2(self.cs_y.derivative()(s/self.arcl), self.cs_x.derivative()(s/self.arcl))
                            if s < 0:
                                plt.plot(time_st,[v_corrected(x) for x in time_st])
                                plt.plot(time_st,[self.cs_v(x) for x in time_st])
                                plt.show()
                                f=1
                            traj.append((t,x_corrected(s/self.arcl),y_corrected(s/self.arcl),v_corrected(t),self.cs_a(t),self.cs_j(t),(v_corrected(t)**2)*self.curvature(s/self.arcl),yaw,s+self.parent_trajectory_arcl))
                        else:
                            traj.append((t,traj[-1][1],traj[-1][2],0,0,0,0,traj[-1][7],self.parent_trajectory_arcl))
                    else:
                        break
                #max_vel,max_acc,max_jerk,max_lat_acc = max([x[3] for x in traj]),max([x[4] for x in traj]),max([x[5] for x in traj]),max([x[6] for x in traj])
                if self.print_console:
                    print('added trajectory',self.maneuver,self.mode,vp_idx,'max_acc',v['max_acc'],'max_vel',v['max_vel'],'length:',math.hypot(traj[-1][1]-traj[0][1], traj[-1][2]-traj[0][2]))
                if m not in all_trajs:
                    all_trajs[m] = []
                all_trajs[m].append(traj)
                if self.show_plots:
                    plt.plot([x[0] for x in traj], [x[3] for x in traj])
        self.all_trajectories = all_trajs
        if self.show_plots:
            plt.title("all velocity profles")
            plt.show()
                
    def generate_extended_trajectory(self,ag_traj_frag,manv_map):
        manv = ag_traj_frag.get_last().manv
        term_v = ag_traj_frag.get_last().loaded_traj_frag[-1][3]
        past_traj_l = ag_traj_frag.get_last().total_length
        horizon = self.horizon
        
        ext_velocity_profiles = {manv:[]}
        
        if manv_map[manv] == 'wait':
            for stop_horizon in [1,2,3]:
                tcs = TriangulationCurve(term_v,stop_horizon,1)
                all_v_profiles = [(stop_horizon,x) for x in tcs.curves()]
                for st_h,v in all_v_profiles:
                    ext_velocity_profiles[manv].append({'func':v})
        else:
            f = lambda x : term_v
            ext_velocity_profiles[manv].append({'func':f})
        all_trajs = []
        for m,v_profiles in ext_velocity_profiles.items():
            for vp_idx,v in enumerate(v_profiles):
                traj = []
                ext_cs_v = v['func']
                stopped_traj = False
                time_st = np.arange(0,horizon+.1,.1)
                x_corrected = self.x_corrected if hasattr(self, 'x_corrected') else self.cs_x
                y_corrected = self.y_corrected if hasattr(self, 'y_corrected') else self.cs_y
                for t in time_st:
                    if t <= horizon:
                        #s = self.t_s_map[t]
                        #s = self.cs_v(t) * t + 0.5 * abs(self.cs_a(t)) * t**2
                        s = scipy.integrate.quad(ext_cs_v,0,t)[0]
                        if manv_map[manv] in ['wait'] and ext_cs_v(t)==0:
                            stopped_traj = True
                        if not stopped_traj:
                            #math.atan2(self.cs_y(s/self.arcl)-traj[-1][2], self.cs_x(s/self.arcl)-traj[-1][1])
                            traj.append((t,x_corrected((past_traj_l + s)/self.arcl),y_corrected((past_traj_l + s)/self.arcl)))
                        else:
                            if len(traj) > 0:
                                traj.append((t,traj[-1][1],traj[-1][2]))
                            else:
                                traj.append((t, ag_traj_frag.get_last().loaded_traj_frag[-1][1], ag_traj_frag.get_last().loaded_traj_frag[-1][2]))
                    else:
                        break
                all_trajs.append(traj)
        return all_trajs
                
    
        
class VehicleTrajectoryPlanner(TrajectoryPlanner):
    
    def get_next_vel(self,iter):
        knts = np.linspace(self.v0,4,10)
        vel_pts_targ = self.vel_pts
        return [vel_pts_targ[0]]+[knts[iter]]*(len(vel_pts_targ)-2)+[vel_pts_targ[-1]]
    
    def check_bounds(self,max_acc,max_lat_acc,max_vel,max_jerk):
        if self.mode == 'aggressive':
            if 1.5 < max_acc < 3.6 or 0.9 < max_jerk < 2 or 4 < max_lat_acc < 5.6:
                return True 
        else:
            if max_acc < 1.5 and max_jerk < 0.9 and max_lat_acc < 4:
                return True 
            
    def assign_mode(self,max_acc,max_lat_acc,max_vel,max_jerk):
        if max_acc > self.traj_constr_obj.max_acc_lims or max_jerk > self.traj_constr_obj.max_jerk_lims or max_lat_acc > self.traj_constr_obj.max_lat_acc_lims or max_vel > self.traj_constr_obj.max_vel_lims:
            return 'infeasible'
        elif 2 < max_acc < 3.6 or 0.9 < max_jerk < 2 or 4 < max_lat_acc < 5.6:
            return 'aggressive' 
        else:
            return 'normal'
        
        
    def generate_proceed_velocity_profiles(self):
        horizon = self.horizon
        indx = self.indx
        ''' 
        calculate the arc length of the generated path 
        '''
        
        '''
        scale an axis with respect to the arc length
        '''
        s_pts = [self.arcl*x for x in indx]
        
        '''
        initial and target velocity points.
        same length as the index points
        '''
        v0 = self.v0
        
        self.velocity_profiles = dict()
        
        if self.show_plots:
            plt.figure()
            plt.title("all vehicle proceed velocity profiles")
            
        
        for iter,vp in enumerate(self.all_velocity_profiles):
            
            this_vel_targets = list(vp)
            #print('-------iter',iter,this_vel_targets)
            
            
            ''' time scaling i.e. mapping time to arc length'''
            t_max = horizon
            time_st = np.arange(0,t_max+.1,.1)
            time_pts = [t_max* (x/s_pts[-1]) for x in s_pts]
            time_pts = [0]
            self.cs_v_s = CubicSpline(s_pts,this_vel_targets)
            for xidx,x in enumerate(s_pts):
                if xidx == 0:
                    continue
                else:
                    _u = self.cs_v_s(s_pts[xidx-1])
                    _v =  self.cs_v_s(s_pts[xidx])
                    _S = s_pts[xidx]-s_pts[xidx-1]
                    t = 2*_S/(_u+_v)
                    if t+time_pts[-1] <= time_pts[-1]:
                        brk=1
                    time_pts.append(t+time_pts[-1])
            try:        
                self.cs_t_s = CubicSpline(time_pts,s_pts)
            except ValueError:
                f=1
                raise
            
            if time_pts[-1] > 8:
                brk = 1
            self.t_s_map = {t:self.cs_t_s(t) for t in time_st}
            
            ''' fit the time scaled velocity curve'''
            #self.cs_v = CubicSpline(time_pts,this_vel_targets)
            ord = min(len(time_pts)-1,3)
            #self.cs_v = UnivariateSpline(time_pts,this_vel_targets,w=np.arange(len(time_pts)+1,1,-1),k=ord)
            if time_pts[-1] < horizon:
                _addl_timepts = np.arange(time_pts[-1]+.5,horizon+.5,.5).tolist()
                _addl_velpts = [this_vel_targets[-1]]*len(_addl_timepts)
                time_pts += _addl_timepts
                this_vel_targets += _addl_velpts
            self.cs_v = UnivariateSpline(time_pts,this_vel_targets,k=ord)
            #plt.plot(time_st,[self.cs_v(z) for z in time_st])
            #plt.plot(time_pts,this_vel_targets,'x')
            #plt.show()
                              
            
            
            max_vel = self.cs_v(find_maxima_minima(True, self.cs_v, horizon))
            min_vel = self.cs_v(find_maxima_minima(False, self.cs_v, horizon))
            
            if self.show_plots:
                plt.plot(np.linspace(time_pts[0],horizon,100),[self.cs_v(x) for x in np.linspace(time_pts[0],horizon,100)])
                plt.show()
                
            
            if ord > 1:
                self.cs_a = self.cs_v.derivative(1)
            else:
                self.cs_a = lambda x : 0
            if ord > 2:
                self.cs_j = self.cs_a.derivative(1)
            else:
                self.cs_j = lambda x : 0
            '''
            max_acc = self.cs_a(find_maxima_minima(True, self.cs_a, horizon))
            f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/arcl)
            max_lat_acc = f_lat_acc_wrt_time(find_maxima_minima(True, f_lat_acc_wrt_time, horizon))
            max_jerk = self.cs_j(find_maxima_minima(True, self.cs_j, horizon,1))
            '''
            max_acc = max([self.cs_a(x) for x in np.arange(horizon)])
            f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/self.arcl)
            max_lat_acc = max([f_lat_acc_wrt_time(x) for x in np.arange(horizon)])
            max_jerk = max([self.cs_j(x) for x in np.arange(horizon)])
            category = self.assign_mode(max_acc,max_lat_acc,max_vel,max_jerk)
            if min_vel <= 0:
                category = 'infeasible'
            if category != 'infeasible':
                entry = {'func':copy.deepcopy(self.cs_v),
                         'target vels':this_vel_targets,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                         }
                if category not in self.velocity_profiles:
                    self.velocity_profiles[category] = []
                self.velocity_profiles[category].append(entry)
            if self.print_console:
                print('target vels',this_vel_targets,'max_vel',max_vel,'max_acc',max_acc,'max_lat_acc',max_lat_acc,'max_jerk',max_jerk,category)
        ''' randomly sample 10 velocity profiles from each category. '''
        indices = {k:np.random.choice(len(v), size=min(10,len(v)), replace=False) for k,v in self.velocity_profiles.items()}
        self.velocity_profiles = {k:[x for idx,x in enumerate(v) if idx in indices[k]] for k,v in self.velocity_profiles.items()}
        if self.show_plots:
            plt.show()
           
    def generate_wait_velocity_profiles(self):
        horizon = self.horizon
        indx = self.indx
        ''' 
        calculate the arc length of the generated path 
        '''
        
        
        '''
        scale an axis with respect to the arc length
        '''
        s_pts = [self.arcl*x for x in indx]
        
        '''
        initial and target velocity points.
        same length as the index points
        '''
        v0 = self.v0
        self.velocity_profiles = dict()
        h_end, h_start = self.traj_constr_obj.stop_horizon_time_sampling_range[1], self.traj_constr_obj.stop_horizon_time_sampling_range[0]
        iter_attempts = 0
        while len(self.velocity_profiles) == 0 and iter_attempts < 10:
            h_end, h_start = h_end+(2*iter_attempts), max(h_start-(2*iter_attempts),0.1)
            for o_it,o_r in enumerate(np.linspace(h_end, h_start,5)):
                if self.print_console:
                    print('-------iter',o_it)
                
                
                
                ''' fit the time scaled velocity curve'''
                
                stop_horizon = o_r
                        
                tcs = TriangulationCurve(self.v0,stop_horizon,10)
                all_v_profiles = [(stop_horizon,x) for x in tcs.curves()]
                
                for st_h,v in all_v_profiles:
                    self.cs_v = v
                
                    max_vel = self.cs_v(find_maxima_minima(True, self.cs_v, horizon))
                    
                    stop_dist = scipy.integrate.quad(self.cs_v,0,stop_horizon)[0]
                    '''
                    if stop_dist > max_stop_dist:
                        continue
                    '''
                    if stop_dist > self.traj_constr_obj.stop_horizon_dist_sampling_range[1] or stop_dist < self.traj_constr_obj.stop_horizon_dist_sampling_range[0]:
                        continue
                    
                    self.cs_a = self.cs_v.derivative(1)
                    self.cs_j = self.cs_v.derivative(2)
                    '''
                    max_acc = self.cs_a(find_maxima_minima(True, self.cs_a, horizon))
                    f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/arcl)
                    max_lat_acc = f_lat_acc_wrt_time(find_maxima_minima(True, f_lat_acc_wrt_time, horizon))
                    max_jerk = self.cs_j(find_maxima_minima(True, self.cs_j, horizon,1))
                    '''
                    self.cs_t_s = lambda x : scipy.integrate.quad(self.cs_v,0,x)[0]
                    max_acc = max([self.cs_a(x) for x in np.arange(0,horizon,.5)])
                    f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/self.arcl)
                    max_lat_acc = max([f_lat_acc_wrt_time(x) for x in np.arange(0,horizon,.5)])
                    max_jerk = max([self.cs_j(x) for x in np.arange(0,horizon,.5)])
                    category = self.assign_mode(max_acc,max_lat_acc,max_vel,max_jerk)
                    if category != 'infeasible':
                        entry = {'func':copy.deepcopy(self.cs_v),
                                 'target stop pt':st_h,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                                 }
                        if category not in self.velocity_profiles:
                            self.velocity_profiles[category] = []
                        self.velocity_profiles[category].append(entry)
                    if self.print_console:       
                        print('target vels',st_h,'max_vel',max_vel,'max_acc',max_acc,'max_lat_acc',max_lat_acc,'max_jerk',max_jerk,category)
            iter_attempts += 1

class PedestrianTrajectoryPlanner(TrajectoryPlanner):
    
    def generate_proceed_velocity_profiles(self):
        horizon = self.horizon
        indx = self.indx
        ''' 
        calculate the arc length of the generated path 
        '''
        
        f_dx = self.cs_x.derivative(1)
        f_dy = self.cs_y.derivative(1)
        f = lambda x : math.hypot(f_dx(x),f_dy(x))
        self.arcl = scipy.integrate.quad(f,0,1)[0]
        '''
        scale an axis with respect to the arc length
        '''
        s_pts = [self.arcl*x for x in indx]
        
        '''
        initial and target velocity points.
        same length as the index points
        '''
        v0 = self.v0
        
        self.velocity_profiles = dict()
        
        for iter,vp in enumerate(self.all_velocity_profiles):
            
            this_vel_targets = list(vp)
            ''' time scaling i.e. mapping time to arc length'''
            t_max = horizon
            time_st = np.arange(0,t_max+.1,.1)
            time_pts = [t_max* (x/s_pts[-1]) for x in s_pts]
            time_pts = [0]
            self.cs_v_s = CubicSpline(s_pts,this_vel_targets)
            for xidx,x in enumerate(s_pts):
                if xidx == 0:
                    continue
                else:
                    _u = self.cs_v_s(s_pts[xidx-1])
                    _v =  self.cs_v_s(s_pts[xidx])
                    _S = s_pts[xidx]-s_pts[xidx-1]
                    t = 2*_S/(_u+_v)
                    time_pts.append(t+time_pts[-1])
            self.cs_t_s = CubicSpline(time_pts,s_pts)
            self.t_s_map = {t:self.cs_t_s(t) for t in time_st}
            
            ''' fit the time scaled velocity curve'''
            self.cs_v = CubicSpline(time_pts,this_vel_targets)
            
            max_vel = self.cs_v(find_maxima_minima(True, self.cs_v, horizon))
            
            self.cs_a = self.cs_v.derivative(1)
            self.cs_j = self.cs_a.derivative(1)
            '''
            max_acc = self.cs_a(find_maxima_minima(True, self.cs_a, horizon))
            f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/arcl)
            max_lat_acc = f_lat_acc_wrt_time(find_maxima_minima(True, f_lat_acc_wrt_time, horizon))
            max_jerk = self.cs_j(find_maxima_minima(True, self.cs_j, horizon,1))
            '''
            max_acc = max([self.cs_a(x) for x in np.arange(horizon)])
            f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/self.arcl)
            max_lat_acc = max([f_lat_acc_wrt_time(x) for x in np.arange(horizon)])
            max_jerk = max([self.cs_j(x) for x in np.arange(horizon)])
            category = self.assign_mode(max_acc,max_lat_acc,max_vel,max_jerk)
            if category != 'infeasible':
                entry = {'func':copy.deepcopy(self.cs_v),
                         'target vels':this_vel_targets,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                         }
                if category not in self.velocity_profiles:
                    self.velocity_profiles[category] = []
                self.velocity_profiles[category].append(entry)
            if self.print_console:        
                print('target vels',this_vel_targets,'max_vel',max_vel,'max_acc',max_acc,'max_lat_acc',max_lat_acc,'max_jerk',max_jerk,category)
            
                
    def generate_wait_velocity_profiles(self):
        horizon = self.horizon
        indx = self.indx
        ''' 
        calculate the arc length of the generated path 
        '''
        
        f_dx = self.cs_x.derivative(1)
        f_dy = self.cs_y.derivative(1)
        f = lambda x : math.hypot(f_dx(x),f_dy(x))
        self.arcl = scipy.integrate.quad(f,0,1)[0]
        '''
        scale an axis with respect to the arc length
        '''
        s_pts = [self.arcl*x for x in indx]
        
        '''
        initial and target velocity points.
        same length as the index points
        '''
        v0 = self.v0
        
        self.velocity_profiles = dict()
        for o_it,o_r in enumerate(np.linspace(1,3,5)):
            if self.print_console:
                print('-------iter',o_it)
            ''' fit the time scaled velocity curve'''
            
            stop_horizon = o_r
                    
            tcs = TriangulationCurve(self.v0,stop_horizon,10)
            all_v_profiles = [(stop_horizon,x) for x in tcs.curves()]
            
            for st_h,v in all_v_profiles:
                self.cs_v = v
            
                max_vel = self.cs_v(find_maxima_minima(True, self.cs_v, horizon))
                
                self.cs_a = self.cs_v.derivative(1)
                self.cs_j = self.cs_v.derivative(2)
             
                self.cs_t_s = lambda x : scipy.integrate.quad(self.cs_v,0,x)[0]
                '''
                max_acc = self.cs_a(find_maxima_minima(True, self.cs_a, horizon))
                f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/arcl)
                max_lat_acc = f_lat_acc_wrt_time(find_maxima_minima(True, f_lat_acc_wrt_time, horizon))
                max_jerk = self.cs_j(find_maxima_minima(True, self.cs_j, horizon,1))
                '''
                max_acc = max([self.cs_a(x) for x in np.arange(0,horizon,.5)])
                f_lat_acc_wrt_time = lambda x : (self.cs_v(x)**2)*self.curvature(self.cs_t_s(x)/self.arcl)
                max_lat_acc = max([f_lat_acc_wrt_time(x) for x in np.arange(0,horizon,.5)])
                max_jerk = max([self.cs_j(x) for x in np.arange(0,horizon,.5)])
                category = self.assign_mode(max_acc,max_lat_acc,max_vel,max_jerk)
                if category != 'infeasible':
                    entry = {'func':copy.deepcopy(self.cs_v),
                             'target stop pt':st_h,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                             }
                    if category not in self.velocity_profiles:
                        self.velocity_profiles[category] = []
                    self.velocity_profiles[category].append(entry)
                if self.print_console:      
                    print('target vels',st_h,'max_vel',max_vel,'max_acc',max_acc,'max_lat_acc',max_lat_acc,'max_jerk',max_jerk,category)
    
    def get_next_vel(self,iter):
        return self.vel_pts[0:-1] + [self.vel_pts[-1]]
    
    def check_bounds(self,max_acc,max_lat_acc,max_vel,max_jerk):
        if self.mode == 'aggressive':
            if max_vel > 1.38 and max_acc < 1:
                return True 
        else:
            if max_vel <= 1.38 and max_acc < 1:
                return True         
            
    def assign_mode(self,max_acc,max_lat_acc,max_vel,max_jerk):
        if max_vel > 1.38 and max_acc < 1:
            return 'aggressive' 
        elif max_vel <= 1.38 and max_acc < 1:
            return 'normal'
        else:
            return 'infeasible'