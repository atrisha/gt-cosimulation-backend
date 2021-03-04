'''
Created on Feb 26, 2021

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


class TrajectoryPlanner:
    
    def __init__(self,centerline,vel_pts):
        self.v0 = vel_pts[0]
        self.vel_pts = vel_pts
        self.centerline = centerline
        
        
    def generate_path(self):
        
        ''' 
        generate path with an index [0,1] that will 
        be later scaled to the arc length
        '''
        
        indx = [round(i/(len(self.centerline)-1),1) for i in np.arange(len(self.centerline))]
        self.cs_x = CubicSpline(indx,[x[0] for x in self.centerline])
        self.cs_y = CubicSpline(indx,[x[1] for x in self.centerline])
        self.path = [(x,self.cs_x(x),self.cs_y(x)) for x in indx]
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
            vel_pts = [vel_pts_targ[0],vel_pts_targ[1]+iter,vel_pts_targ[2]+iter]
            
            print('-------iter',iter)
            print('target vels',vel_pts)
            
            ''' time scaling i.e. mapping time to arc length'''
            t_max = horizon
            time_st = np.arange(0,t_max+.1,.1)
            time_pts = np.linspace(start=0, stop=t_max, num=len(s_pts))
            self.cs_t_s = CubicSpline(time_pts,s_pts)
            self.t_s_map = {t:self.cs_t_s(t) for t in time_st}
            
            ''' fit the time scaled velocity curve'''
            self.cs_v = CubicSpline(time_pts,vel_pts)
            self.cs_a = self.cs_v.derivative(1)
            self.cs_j = self.cs_v.derivative(2)
            
            
            traj = []
            for t in time_st:
                if t <= horizon:
                    s = self.t_s_map[t]
                    traj.append((t,self.cs_x(s/23),self.cs_y(s/23),self.cs_v(t),self.cs_a(t),self.cs_j(t)))
                else:
                    break
            max_vel,max_acc,max_jerk = max([x[3] for x in traj]),max([x[4] for x in traj]),max([x[5] for x in traj])
            
            print('max_vel:',max_vel)
            print('max_acc:',max_acc)
            print('max_jerk:',max_jerk)
            
            if max_acc < 3.6 and max_jerk < 4:
                break 
            
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
        

        
        
    
        
    
    
def draw_canvas():
    
    vehicle_lane = [(0,4), (13.5,4), (13.5,24), (6.5,24), (6.5,12), (0,12), (0,4)]
    ped_lane = [(3,4), (6.5,4), (6.5,12), (3,12), (3,4)]
    veh_centerline = [(8.25,24), (8.25,10), (0,10)]
    ped_centerline = [(4.75,0),(4.75,8),(4.75,16)]
    
    lane_divider = [(0,8),(5,8)]
    indx = np.arange(len(veh_centerline))
    veh_motion = TrajectoryPlanner(veh_centerline,[10,3,6])
    veh_motion.generate_trajectory(6)
    
    ped_motion = TrajectoryPlanner(ped_centerline,[3,2,3])
    ped_motion.generate_trajectory(6)
    
    
    #path_indx = np.arange(indx[0],indx[-1]+.01,.01)
    rect = patches.Circle((veh_motion.trajectory[0][1], veh_motion.trajectory[0][1]), 1)
    #rectm = patches.Rectangle((cs_x(path_indx[0])-.4, cs_y(path_indx[0])), .5, .75, linewidth=1, edgecolor='r', facecolor='none', fill= True)
    #rectl = patches.Rectangle((cs_x(path_indx[0])-.4, cs_y(path_indx[0])), .5, .75, linewidth=1, edgecolor='r', facecolor='none', fill= True)
    
    
    fig, ax = plt.subplots()
    #plt.plot([x[0] for x in ped_centerline], [x[1] for x in ped_centerline],'-')
    plt.plot([x[0] for x in vehicle_lane], [x[1] for x in vehicle_lane],'--')
    plt.plot([x[0] for x in ped_lane], [x[1] for x in ped_lane],'--')
    plt.plot([x[0] for x in lane_divider], [x[1] for x in lane_divider],'--')
    plt.plot([x[0] for x in veh_centerline], [x[1] for x in veh_centerline],'x')
    #plt.plot([cs_x(x) for x in path_indx],[cs_y(x) for x in path_indx])
    veh_path, = ax.plot([], [])
    veh_pathm, = ax.plot([], [])
    veh_pathl, = ax.plot([], [])
    ped_path, = ax.plot([], [])
    #plt.plot(ped_centerline[0],ped_centerline[1],marker="^") 
    hinge, = ax.plot(ped_motion.trajectory[0][1],ped_motion.trajectory[0][2],'^')
    #ax.add_patch(rect)
    plt.axis('equal')
    
    
    
    
    def init():
        """initialize animation"""
        veh_path.set_data([x[1] for x in veh_motion.trajectory],[x[2] for x in veh_motion.trajectory])
        '''
        veh_pathm.set_data([cs_x(x) for x in path_indx],[cs_y(x) for x in path_indx])
        veh_pathl.set_data([cs_x(x) for x in path_indx],[cs_y(x) for x in path_indx])
        '''
        ped_path.set_data([x[1] for x in ped_motion.trajectory],[x[2] for x in ped_motion.trajectory])
        hinge.set_data(ped_motion.trajectory[0][1],ped_motion.trajectory[0][2])
        ax.add_patch(rect)
        '''
        ax.add_patch(rectm)
        ax.add_patch(rectl)
        '''
        return veh_path,rect,ped_path,hinge,

    def animate(i):
        """perform animation step"""
        
        if i < len(veh_motion.trajectory)-1:
            yaw_line = np.asarray([veh_motion.trajectory[i+1][1]-veh_motion.trajectory[i][1],veh_motion.trajectory[i+1][2]-veh_motion.trajectory[i][2]])
            xax = np.asarray([1,0])
            cp = np.dot(yaw_line,xax)
            ang = np.rad2deg(np.arccos(cp/(np.linalg.norm(yaw_line)*np.linalg.norm(xax))))
            '''
            yaw_linem = np.asarray([cs_x(path_indx[i//2+1])-cs_x(path_indx[i//2]),cs_y(path_indx[i//2+1])-cs_y(path_indx[i//2])])
            xaxm = np.asarray([1,0])
            cpm = np.dot(yaw_linem,xaxm)
            angm = np.rad2deg(np.arccos(cpm/(np.linalg.norm(yaw_linem)*np.linalg.norm(xaxm))))
            
            yaw_linel = np.asarray([cs_x(path_indx[i//3+1])-cs_x(path_indx[i//3]),cs_y(path_indx[i//3+1])-cs_y(path_indx[i//3])])
            xaxl = np.asarray([1,0])
            cpl = np.dot(yaw_linel,xaxl)
            angl = np.rad2deg(np.arccos(cpl/(np.linalg.norm(yaw_linel)*np.linalg.norm(xaxl))))
            '''
        rot_ang = 180-((ang-90)%360)
        new_corn = veh_motion.trajectory[i][1]+.4*np.cos(np.deg2rad(rot_ang)), veh_motion.trajectory[i][2]+.4*np.sin(np.deg2rad(rot_ang))
        '''
        rot_angm = 180-((angm-90)%360)
        rot_angl = 180-((angl-90)%360)
        
        new_cornm = cs_x(path_indx[i//2])+.4*np.cos(np.deg2rad(rot_angm)), cs_y(path_indx[i//2])+.4*np.sin(np.deg2rad(rot_angm))
        new_cornl = cs_x(path_indx[i//3])+.4*np.cos(np.deg2rad(rot_angl)), cs_y(path_indx[i//3])+.4*np.sin(np.deg2rad(rot_angl))
        '''
        print('time:',round(i/10,1),'s')
        #t2 = mpl.transforms.Affine2D().rotate_deg_around(new_corn[0],new_corn[1],(90-ang)) + ax.transData
        '''
        #t2 = mpl.transforms.Affine2D().rotate_deg_around(cs_x(path_indx[i]), cs_y(path_indx[i]),(90-ang)) + ax.transData
        
        t2m = mpl.transforms.Affine2D().rotate_deg_around(new_cornm[0],new_cornm[1],(90-angm)) + ax.transData
        t2l = mpl.transforms.Affine2D().rotate_deg_around(new_cornl[0],new_cornl[1],(90-angl)) + ax.transData
        '''
        #rect.set_transform(t2)
        
        rect.center = veh_motion.trajectory[i][1], veh_motion.trajectory[i][2]
        #rectm.set_transform(t2m)
        #rectm.set_xy((cs_x(path_indx[i//2])+(.4*np.cos(np.deg2rad(rot_angm))), cs_y(path_indx[i//2])+(.4*np.sin(np.deg2rad(rot_angm)))))
        #rectl.set_transform(t2l)
        #rectl.set_xy((cs_x(path_indx[i//3])+(.4*np.cos(np.deg2rad(rot_angl))), cs_y(path_indx[i//3])+(.4*np.sin(np.deg2rad(rot_angl)))))
        veh_path.set_data([],[])
        veh_path.set_data([x[1] for x in veh_motion.trajectory[i:]],[x[2] for x in veh_motion.trajectory[i:]])
        '''
        veh_pathm.set_data([],[])
        veh_pathm.set_data([cs_x(x) for x in path_indx[i//2:]],[cs_y(x) for x in path_indx[i//2:]])
        veh_pathl.set_data([],[])
        veh_pathl.set_data([cs_x(x) for x in path_indx[i//3:]],[cs_y(x) for x in path_indx[i//3:]])
        '''
        ped_path.set_data([],[])
        ped_path.set_data([x[1] for x in ped_motion.trajectory[i:]],[x[2] for x in ped_motion.trajectory[i:]])
        hinge.set_data(None,None)
        hinge.set_data(ped_motion.trajectory[i][1], ped_motion.trajectory[i][2])
        return veh_path,rect,ped_path,hinge,

    ani = animation.FuncAnimation(fig, animate, frames=len(ped_motion.trajectory),
                              interval=100, blit=True, init_func=init)
    plt.show()
    

draw_canvas()