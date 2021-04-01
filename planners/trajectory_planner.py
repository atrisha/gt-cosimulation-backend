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



def find_maxima_minima(max,cs_v,thresh,guess=3):
    
    #lc1 = LinearConstraint(np.ones((1,1)),0,thresh)
    lc1 = Bounds(0,thresh)
    if not max:
        res = minimize(cs_v, x0=[guess], method='TNC', bounds=lc1)
    else:
        f = lambda x : -cs_v(x)
        res = minimize(f, x0=[guess], method='TNC', bounds=lc1)
    
    return res.x

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
    
    
    
    def __init__(self,centerline,vel_pts,maneuver,mode):
        self.v0 = vel_pts[0][0]
        self.target_vels = vel_pts
        self.vel_pts = None
        self.centerline = centerline
        self.maneuver = maneuver
        self.mode = mode
        if maneuver not in WAIT_MANEUVERS:
            self.build_velocity_lattice(vel_pts)
        self.horizon = 6
    
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
        self.all_velocity_profiles = []
        for vp in vel_profiles:
            _v = []
            for i,v in enumerate(vp):
                if v is not None:
                    _v.append(v)
                else:
                    nxt_valid_indx = next(i+idx for idx,item in enumerate(vp[i:]) if item is not None)
                    prev_valid_indx = i-1
                    prev_valid_prop = math.hypot(self.centerline[0][0]-self.centerline[prev_valid_indx][0], self.centerline[0][1]-self.centerline[prev_valid_indx][1]) / (math.hypot(self.centerline[prev_valid_indx][0]-self.centerline[nxt_valid_indx][0], self.centerline[prev_valid_indx][1]-self.centerline[nxt_valid_indx][1]) + math.hypot(self.centerline[0][0]-self.centerline[prev_valid_indx][0], self.centerline[0][1]-self.centerline[prev_valid_indx][1])) 
                    intpl_v = prev_valid_prop*vp[prev_valid_indx] + (1-prev_valid_prop)*vp[nxt_valid_indx]
                    _v.append(intpl_v)
            self.all_velocity_profiles.append(_v)
        
    
    def generate_path(self):
        ''' 
        generate path with an index [0,1] that will 
        be later scaled to the arc length
        '''
        seg_l = [0]+ [math.hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]) for x in zip(self.centerline[1:],self.centerline[:-1])]
        seg_l = [sum(seg_l[:i+1]) for i,x in enumerate(seg_l)]
        indx = [seg_l[i]/seg_l[-1] for i in np.arange(len(self.centerline))]
        print(indx)
        
        
        self.cs_x = UnivariateSpline(indx,[x[0] for x in self.centerline],k=2)
        self.cs_y = UnivariateSpline(indx,[x[1] for x in self.centerline],k=2)
        self.path = [(x,self.cs_x(x),self.cs_y(x)) for x in indx]
        xdd = self.cs_x.derivative(2) 
        ydd = self.cs_y.derivative(2) 
        xd = self.cs_x.derivative(1)
        yd = self.cs_y.derivative(1)
        plot_indx_x = np.linspace(indx[0],indx[-1],100)
        self.curvature = lambda x: abs(((xd(x)/yd(x))*(ydd(x)/xd(x)**2) - (yd(x)/xd(x))*(xdd(x)/yd(x)**2))) / np.power((xd(x)/yd(x))** 2 + (yd(x)/xd(x))** 2, 3 / 2)
        if self.show_plots:
            plt.figure()
            plt.title('path')
            plt.plot([self.cs_x(x) for x in plot_indx_x],[self.cs_y(x) for x in plot_indx_x])
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
        if self.maneuver in ['turn','walk']:
            self.generate_proceed_velocity_profiles()
        else:
            self.generate_wait_velocity_profiles()
        
        
        if not all:
            traj = []
            self.cs_v = self.velocity_profiles[self.mode][0]['func']
            self.cs_a = self.cs_v.derivative(1)
            self.cs_j = self.cs_v.derivative(2)
        
            stopped_traj = False
            time_st = np.arange(0,horizon+.1,.1)
            for t in time_st:
                if t <= horizon:
                    #s = self.t_s_map[t]
                    #s = self.cs_v(t) * t + 0.5 * abs(self.cs_a(t)) * t**2
                    s = scipy.integrate.quad(self.cs_v,0,t)[0]
                    if self.maneuver in ['wait'] and self.cs_v(t)==0:
                        stopped_traj = True
                    if not stopped_traj:
                        traj.append((t,self.cs_x(s/self.arcl),self.cs_y(s/self.arcl),self.cs_v(t),self.cs_a(t),self.cs_j(t),(self.cs_v(t)**2)*self.curvature(s/self.arcl)))
                    else:
                        traj.append((t,traj[-1][1],traj[-1][2],0,0,0,0))
                else:
                    break
            max_vel,max_acc,max_jerk,max_lat_acc = max([x[3] for x in traj]),max([x[4] for x in traj]),max([x[5] for x in traj]),max([x[6] for x in traj])
            
            print('max_vel:',max_vel)
            print('max_acc:',max_acc)
            print('max_lat_acc:',max_lat_acc)
            print('max_jerk:',max_jerk)
            
            if self.show_plots:
                time_ax_x = np.linspace(start=0, stop=time_st[-1], num=100)
                plt.figure()
                plt.plot(time_ax_x,[self.cs_v(x) for x in time_ax_x])
                plt.title('velocity')
                plt.figure()
                plt.plot(time_ax_x,[self.cs_a(x) for x in time_ax_x])
                plt.title('acceleration')
                plt.figure()
                plt.plot(time_ax_x,[self.cs_j(x) for x in time_ax_x])
                plt.title('jerk')
                plt.figure()
                plt.plot([x[0] for x in traj],[x[6] for x in traj])
                plt.title('lateral acc')
                plt.show()
            self.trajectory = [(x[0],x[1],x[2]) for x in traj]
        else:
            all_trajs = dict()
            if self.show_plots:
                plt.figure()
            for m,v_profiles in self.velocity_profiles.items():
                for vp_idx,v in enumerate(v_profiles):
                    traj = []
                    self.cs_v = v['func']
                    self.cs_a = self.cs_v.derivative(1)
                    self.cs_j = self.cs_v.derivative(2)
                
                    stopped_traj = False
                    time_st = np.arange(0,horizon+.1,.1)
                    for t in time_st:
                        if t <= horizon:
                            #s = self.t_s_map[t]
                            #s = self.cs_v(t) * t + 0.5 * abs(self.cs_a(t)) * t**2
                            s = scipy.integrate.quad(self.cs_v,0,t)[0]
                            if self.maneuver in ['wait'] and self.cs_v(t)==0:
                                stopped_traj = True
                            if not stopped_traj:
                                traj.append((t,self.cs_x(s/self.arcl),self.cs_y(s/self.arcl),self.cs_v(t),self.cs_a(t),self.cs_j(t),(self.cs_v(t)**2)*self.curvature(s/self.arcl)))
                            else:
                                traj.append((t,traj[-1][1],traj[-1][2],0,0,0,0))
                        else:
                            break
                    #max_vel,max_acc,max_jerk,max_lat_acc = max([x[3] for x in traj]),max([x[4] for x in traj]),max([x[5] for x in traj]),max([x[6] for x in traj])
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
            
    def print_category(self,max_acc,max_lat_acc,max_vel,max_jerk):
        if max_acc > 3.6 or max_jerk > 2 or max_lat_acc > 5.6:
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
                    time_pts.append(t+time_pts[-1])
                    if t <= time_pts[-1]:
                        brk=1
                
            self.cs_t_s = CubicSpline(time_pts,s_pts)
            if time_pts[-1] > 8:
                brk = 1
            self.t_s_map = {t:self.cs_t_s(t) for t in time_st}
            
            ''' fit the time scaled velocity curve'''
            #self.cs_v = CubicSpline(time_pts,this_vel_targets)
            self.cs_v = UnivariateSpline(time_pts,this_vel_targets)
            if self.show_plots:
                plt.plot(np.linspace(time_pts[0],time_pts[-1],100),[self.cs_v(x) for x in np.linspace(time_pts[0],time_pts[-1],100)])
                
            
            max_vel = self.cs_v(find_maxima_minima(True, self.cs_v, horizon))
            min_vel = self.cs_v(find_maxima_minima(False, self.cs_v, horizon))
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
            category = self.print_category(max_acc,max_lat_acc,max_vel,max_jerk)
            if min_vel <= 0:
                category = 'infeasible'
            if category != 'infeasible':
                entry = {'func':copy.deepcopy(self.cs_v),
                         'target vels':this_vel_targets,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                         }
                if category not in self.velocity_profiles:
                    self.velocity_profiles[category] = []
                self.velocity_profiles[category].append(entry)
                    
            print('target vels',this_vel_targets,'max_vel',max_vel,'max_acc',max_acc,'max_lat_acc',max_lat_acc,'max_jerk',max_jerk,category)
        if self.show_plots:
            plt.show()
           
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
        map = NYCMapInfo() 
        max_stop_dist = map.get_max_stop_dist(NYCMapInfo.veh_centerline[0])
        self.velocity_profiles = dict()
        for o_it,o_r in enumerate(np.linspace(horizon,1,5)):
            
            print('-------iter',o_it)
            
            
            
            ''' fit the time scaled velocity curve'''
            
            stop_horizon = o_r
                    
            tcs = TriangulationCurve(self.v0,stop_horizon,10)
            all_v_profiles = [(stop_horizon,x) for x in tcs.curves()]
            
            for st_h,v in all_v_profiles:
                self.cs_v = v
            
                max_vel = self.cs_v(find_maxima_minima(True, self.cs_v, horizon))
                
                stop_dist = scipy.integrate.quad(self.cs_v,0,stop_horizon)[0]
                
                if stop_dist > max_stop_dist:
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
                category = self.print_category(max_acc,max_lat_acc,max_vel,max_jerk)
                if category != 'infeasible':
                    entry = {'func':copy.deepcopy(self.cs_v),
                             'target stop pt':st_h,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                             }
                    if category not in self.velocity_profiles:
                        self.velocity_profiles[category] = []
                    self.velocity_profiles[category].append(entry)
                        
                print('target vels',st_h,'max_vel',max_vel,'max_acc',max_acc,'max_lat_acc',max_lat_acc,'max_jerk',max_jerk,category)


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
            category = self.print_category(max_acc,max_lat_acc,max_vel,max_jerk)
            if category != 'infeasible':
                entry = {'func':copy.deepcopy(self.cs_v),
                         'target vels':this_vel_targets,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                         }
                if category not in self.velocity_profiles:
                    self.velocity_profiles[category] = []
                self.velocity_profiles[category].append(entry)
                    
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
                category = self.print_category(max_acc,max_lat_acc,max_vel,max_jerk)
                if category != 'infeasible':
                    entry = {'func':copy.deepcopy(self.cs_v),
                             'target stop pt':st_h,'max_vel':max_vel,'max_acc':max_acc,'max_lat_acc':max_lat_acc,'max_jerk':max_jerk
                             }
                    if category not in self.velocity_profiles:
                        self.velocity_profiles[category] = []
                    self.velocity_profiles[category].append(entry)
                        
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
            
    def print_category(self,max_acc,max_lat_acc,max_vel,max_jerk):
        if max_vel > 1.38 and max_acc < 1:
            return 'aggressive' 
        elif max_vel <= 1.38 and max_acc < 1:
            return 'normal'
        else:
            return 'infeasible'