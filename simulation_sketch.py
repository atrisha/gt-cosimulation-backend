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
from planners.trajectory_planner import TrajectoryPlanner
from planners.maneuvers import right_turn_wait
from maps.map_info import NYCMapInfo



        

        
        
    
        
    
    
def draw_canvas():
    
    
    
    lane_divider = [(0,8),(5,8)]
    indx = np.arange(len(NYCMapInfo.veh_centerline))
    veh_motion = TrajectoryPlanner(NYCMapInfo.veh_centerline,right_turn_wait(3, 'lb'),'right_turn_wait')
    veh_motion.generate_trajectory(6)
    
    ped_motion = TrajectoryPlanner(NYCMapInfo.ped_centerline,[3,2,3],'pedestrian_walk')
    ped_motion.generate_trajectory(6)
    
    dist_gaps = [math.hypot(x[0]-x[1], y[0]-y[1]) for x,y in zip(ped_motion.trajectory,veh_motion.trajectory)]
    print('min distance gap',min(dist_gaps))
    
    #path_indx = np.arange(indx[0],indx[-1]+.01,.01)
    rect = patches.Circle((veh_motion.trajectory[0][1], veh_motion.trajectory[0][1]), 1)
    #rectm = patches.Rectangle((cs_x(path_indx[0])-.4, cs_y(path_indx[0])), .5, .75, linewidth=1, edgecolor='r', facecolor='none', fill= True)
    #rectl = patches.Rectangle((cs_x(path_indx[0])-.4, cs_y(path_indx[0])), .5, .75, linewidth=1, edgecolor='r', facecolor='none', fill= True)
    
    
    fig, ax = plt.subplots()
    #plt.plot([x[0] for x in ped_centerline], [x[1] for x in ped_centerline],'-')
    plt.plot([x[0] for x in NYCMapInfo.vehicle_lane], [x[1] for x in NYCMapInfo.vehicle_lane],'--')
    plt.plot([x[0] for x in NYCMapInfo.ped_lane], [x[1] for x in NYCMapInfo.ped_lane],'--')
    plt.plot([x[0] for x in lane_divider], [x[1] for x in lane_divider],'--')
    plt.plot([x[0] for x in NYCMapInfo.veh_centerline], [x[1] for x in NYCMapInfo.veh_centerline],'x')
    #plt.plot([cs_x(x) for x in path_indx],[cs_y(x) for x in path_indx])
    veh_path, = ax.plot([], [])
    veh_pathm, = ax.plot([], [])
    veh_pathl, = ax.plot([], [])
    ped_path, = ax.plot([], [])
    #plt.plot(ped_centerline[0],ped_centerline[1],marker="^") 
    hinge, = ax.plot(ped_motion.trajectory[0][1],ped_motion.trajectory[0][2],'^')
    #ax.add_patch(rect)
    plt.axis('equal')
    #plt.show()
    
    
    
    
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
        print('time:',round(i/10,1),'s','dist_gaps',dist_gaps[i])
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