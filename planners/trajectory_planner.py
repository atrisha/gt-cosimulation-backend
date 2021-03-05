'''
Created on Mar 5, 2021

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.animation as animation
import math
import scipy.integrate
from maps.map_info import NYCMapInfo
import copy
import random

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
            _cs = np.random.uniform(low=0*stop_horizon, high=stop_horizon, size=N)
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
        elif self.c < x < self.b:
            return self.v0 * (1 - (((self.b-x)**2) / ((self.b-self.a)*(self.b-self.c))))
        else:
            return self.v0
        
    def derivative(self,deg):
        
        def _d(_x):
            x = self.b - _x
            if x <= 0:
                return 0
            elif self.a < x <= self.c:
                return self.v0 * ((2*(x-self.a))/((self.b-self.a)*(self.c-self.a)))
            elif self.c < x < self.b:
                return self.v0 * ((2*(self.b-x))/((self.b-self.a)*(self.b-self.c)))
            else:
                return 0
            
        def _dd(_x):
            x = self.b - _x
            if x <= 0:
                return 0
            elif self.a < x <= self.c:
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
tcs = TriangulationCurve(5,4,10)
X = np.linspace(0,4,1000)
for tc in tcs.curves():
    Y = [tc(x) for x in X]
    plt.plot(X,Y)
plt.show() 

for tc in tcs.curves():
    tcd = tc.derivative(1)
    Y = [tcd(x) for x in X]
    plt.plot(X,Y)
plt.show()
'''  
    
        


class TrajectoryPlanner:
    
    show_plots = True
    
    def __init__(self,centerline,vel_pts,maneuver):
        self.v0 = vel_pts[0]
        self.vel_pts = vel_pts
        self.centerline = centerline
        self.maneuver = maneuver
        
        
    def generate_path(self):
        
        ''' 
        generate path with an index [0,1] that will 
        be later scaled to the arc length
        '''
        seg_l = [0]+ [math.hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]) for x in zip(self.centerline[1:],self.centerline[:-1])]
        seg_l = [sum(seg_l[:i+1]) for i,x in enumerate(seg_l)]
        indx = [seg_l[i]/seg_l[-1] for i in np.arange(len(self.centerline))]
        self.cs_x = CubicSpline(indx,[x[0] for x in self.centerline])
        self.cs_y = CubicSpline(indx,[x[1] for x in self.centerline])
        self.path = [(x,self.cs_x(x),self.cs_y(x)) for x in indx]
        xdd = self.cs_x.derivative(2)
        ydd = self.cs_y.derivative(2)
        xd = self.cs_x.derivative(1)
        yd = self.cs_y.derivative(1)
        plot_indx_x = np.linspace(indx[0],indx[-1],100)
        self.curvature = lambda x: abs((xd(x)*ydd(x) - yd(x)*xdd(x))) / np.power(xd(x)** 2 + yd(x)** 2, 3 / 2)
        if self.show_plots:
            plt.title('curvature')
            plt.plot(plot_indx_x,[self.curvature(x) for x in plot_indx_x])
            plt.show()
        self.indx = indx
        return self.path
    
    def generate_trajectory(self,horizon):
        self.generate_path()
        
        indx = self.indx
        ''' 
        calculate the arc length of the generated path 
        '''
        
        f_dx = self.cs_x.derivative(1)
        f_dy = self.cs_y.derivative(1)
        f = lambda x : math.hypot(f_dx(x),f_dy(x))
        arcl = scipy.integrate.quad(f,0,1)[0]
        '''
        scale an axis with respect to the arc length
        '''
        s_pts = [arcl*x for x in indx]
        
        '''
        initial and target velocity points.
        same length as the index points
        '''
        v0 = self.v0
        vel_pts_targ = self.vel_pts
        
        ''' main iteration loop '''
        for iter in np.arange(0,100):
            ''' some logic to find an alternate target 
            velocity points since the previous iteration 
            would have generated an infeasible trajectory 
            '''
            vel_pts = [vel_pts_targ[0]]+[x+iter for x in vel_pts_targ[1:]]
            
            print('-------iter',iter)
            print('target vels',vel_pts)
            
            ''' time scaling i.e. mapping time to arc length'''
            t_max = horizon
            time_st = np.arange(0,t_max+.1,.1)
            time_pts = [t_max* (x/s_pts[-1]) for x in s_pts]
            self.cs_t_s = CubicSpline(time_pts,s_pts)
            self.t_s_map = {t:self.cs_t_s(t) for t in time_st}
            
            ''' fit the time scaled velocity curve'''
            if self.maneuver in ['right_turn_wait']:
                stop_horizon = t_max
                for i,v in enumerate(vel_pts):
                    if v == 0:
                        stop_horizon = time_pts[i]
                        break
                        
                tcs = TriangulationCurve(self.v0,stop_horizon,1)
                self.cs_v = tcs.curves()[0]
            else:
                self.cs_v = CubicSpline(time_pts,vel_pts)
            self.cs_a = self.cs_v.derivative(1)
            self.cs_j = self.cs_v.derivative(2)
            
            
            traj = []
            stopped_traj = False
            for t in time_st:
                if t <= horizon:
                    s = self.t_s_map[t]
                    if self.maneuver in ['right_turn_wait'] and self.cs_v(t)==0:
                        stopped_traj = True
                    if not stopped_traj:
                        traj.append((t,self.cs_x(s/arcl),self.cs_y(s/arcl),self.cs_v(t),self.cs_a(t),self.cs_j(t),(self.cs_v(t)**2)*self.curvature(s/arcl)))
                    else:
                        traj.append((t,traj[-1][1],traj[-1][2],0,0,0,0))
                else:
                    break
            max_vel,max_acc,max_jerk = max([x[3] for x in traj]),max([x[4] for x in traj]),max([x[5] for x in traj])
            
            print('max_vel:',max_vel)
            print('max_acc:',max_acc)
            print('max_jerk:',max_jerk)
            
            if max_acc < 3.6 and max_jerk < 4:
                break 
        
        if self.show_plots:
            time_ax_x = np.linspace(start=0, stop=time_pts[-1], num=100)
            plt.plot(time_ax_x,[self.cs_v(x) for x in time_ax_x])
            plt.title('velocity')
            plt.show()
            plt.plot(time_ax_x,[self.cs_a(x) for x in time_ax_x])
            plt.title('acceleration')
            plt.show()
            plt.plot(time_ax_x,[self.cs_j(x) for x in time_ax_x])
            plt.title('jerk')
            plt.show()
            plt.plot([x[0] for x in traj],[x[6] for x in traj])
            plt.title('lateral acc')
            plt.show()
        self.trajectory = [(x[0],x[1],x[2]) for x in traj]
        f=1
        
        '''
        dsdt = self.cs_t_s.derivative(1)
        ts_x = np.linspace(0,time_st[-1],200)
        f = dsdt(0)
        vel = [float(dsdt(x))*arcl for x in np.linspace(0,time_st[-1],200)]
        #plt.plot(ts_x,vel_y)
        #plt.title('velocity')
        #plt.show()
        #plt.plot([x[0] for x in self.t_s_map],[x[1]*arcl for x in self.t_s_map])
        #plt.show()
        self.time_st = time_st
        '''