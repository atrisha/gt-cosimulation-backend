'''
Created on Apr 13, 2021

@author: Atrisha
'''
import unittest
from planners.trajectory_planner import TrajectoryPlanner, VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner
from planners.trajectory_planner import WaitTrajectoryConstraints, ProceedTrajectoryConstraints
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

WAIT_ACTIONS = ['yield-to-merging','wait_for_lead_to_cross','wait-for-oncoming','decelerate-to-stop','wait-on-red','wait-for-pedestrian']

class TestVehicleTurn(unittest.TestCase):
    
    def test_ws_freeturn(self):
        rt_manv = 'wait-for-oncoming'
        st_manv = 'track_speed'
        """
        Test that trajectory generation works for west to south free turn
        """
        
        ''' Get the track of a representative right turn vehicle to construct a path centerline '''
        
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+'769'+'\\uni_weber_'+'769'+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID=2 ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        rt_path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
        
        ''' Get the track of a representative straight through vehicle to construct a path centerline '''
        q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID=28 ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        st_path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
        
        ''' Construct the velocity range samples. First point is the initial velocity (v0,)
        The mid point velocity is a reasonable sampling range for a turning trajectory 
        The exit point velocity is (5,8.3) 
        '''
        rt_vel_pts = [(6,)] + [(None,) if i != len(np.arange(1,len(rt_path)-1))//2 else (2,4) for i in np.arange(1,len(rt_path)-1)] + [(5,8.3)]
        
        ''' 
        Straight through vehicles can just have an initial and exit velocity point constraints.
        '''
        st_vel_pts = [(8,)] + [(None,)]*(len(st_path)-2) + [(8,14)]
        
        rt_trajs,st_trajs = dict(), dict()
        
        ''' Some housekeeping to translate the internal maneuver codes '''
        if rt_manv in WAIT_ACTIONS:
            rt_manv = 'wait'
        else:
            rt_manv = 'turn'
        if st_manv in WAIT_ACTIONS:
            st_manv = 'wait'
        
        ''' Setup the constraints object to generate the trajectory '''
        
        if rt_manv == 'wait':
            rt_traj_constr = WaitTrajectoryConstraints(init_vel=8,waypoints=rt_path,stop_horizon_dist_sampling_range=(20,30),stop_horizon_time_sampling_range=(4,8))
        else:
            rt_traj_constr = ProceedTrajectoryConstraints(waypoints=rt_path,waypoint_vel_sampling_range=rt_vel_pts)
        rt_traj_constr.set_limit_constraints()
        rt_motion = VehicleTrajectoryPlanner(rt_traj_constr,rt_manv,None,6)
        rt_motion.generate_trajectory(True)
        rt_trajs[rt_manv] = rt_motion.all_trajectories
        assert len(rt_trajs[rt_manv]) > 0
        
        
        
        plt.figure()
        for traj_obj,traj_list in rt_trajs[rt_manv].items():
            for traj in traj_list:
                plt.plot([x[0] for x in traj], [x[3] for x in traj])
                f=1
        plt.title('right turn velocities')
        print('right turn trajectories generated')
        sel_rt_traj = list(zip([float(x[1]) for x in traj], [float(x[2]) for x in traj]))
        
        st_traj_constr = ProceedTrajectoryConstraints(waypoints=st_path,waypoint_vel_sampling_range=st_vel_pts)
        st_traj_constr.set_limit_constraints()
        st_motion = VehicleTrajectoryPlanner(st_traj_constr,st_manv,None,6)
        st_motion.generate_trajectory(True)
        st_trajs[st_manv] = st_motion.all_trajectories
        
        assert len(st_trajs[st_manv]) > 0
        
        plt.figure()
        for traj_obj,traj_list in st_trajs[st_manv].items():
            for traj in traj_list:
                plt.plot([x[0] for x in traj], [x[3] for x in traj])
                f=1
        plt.title('straight through velocities')
        print('straight through trajectories generated')
        
        sel_st_traj = list(zip([float(x[1]) for x in traj], [float(x[2]) for x in traj]))
        
        
        plt.show()
        
        ''' Just take one example, the one in sel_xx_traj, and show the animation '''
        
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(min([x[0] for x in sel_st_traj]+[x[0] for x in sel_rt_traj])-10, max([x[0] for x in sel_st_traj]+[x[0] for x in sel_rt_traj])+10), ylim=(min([x[1] for x in sel_st_traj]+[x[1] for x in sel_rt_traj])-10, max([x[1] for x in sel_st_traj]+[x[1] for x in sel_rt_traj])+10))
        line1, = ax.plot([], [], lw=2)
        line2, = ax.plot([], [], lw=2)
        
        # initialization function: plot the background of each frame
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1,line2,
        
        # animation function.  This is called sequentially
        def animate(i):
            line1.set_data([x[0] for x in sel_rt_traj[:i]], [x[1] for x in sel_rt_traj[:i]])
            line2.set_data([x[0] for x in sel_st_traj[:i]], [x[1] for x in sel_st_traj[:i]])
            return line1,line2,
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(sel_st_traj), interval=20, blit=True, repeat = False) 
        plt.show()
        
        
    def test_se_leftturn(self):
        lt_manv = 'proceed-turn'
        st_manv = 'track_speed'
        """
        Test that trajectory generation works for south to east free turn
        """
        
        ''' Get the track of a representative left turn vehicle to construct a path centerline '''
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+'769'+'\\uni_weber_'+'769'+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID=11 ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        lt_path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
        
        
        ''' Get the track of a representative straight through vehicle to construct a path centerline '''
        q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID=23 ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        st_path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
        
        ''' Construct the velocity range samples. First point is the initial velocity (v0,)
        The mid point velocity is a reasonable sampling range for a turning trajectory 
        The exit point velocity range is (8,11) 
        '''
        lt_vel_pts = [(3,)] + [(None,) if i != len(np.arange(1,len(lt_path)-1))//2 else (3,8) for i in np.arange(1,len(lt_path)-1)] + [(8,11)]
        st_vel_pts = [(12,)] + [(None,)]*(len(st_path)-2) + [(8,14)]
        
        ''' Some housekeeping to translate the internal maneuver codes '''
        if lt_manv in WAIT_ACTIONS:
            lt_manv = 'wait'
        else:
            lt_manv = 'turn'
        if st_manv in WAIT_ACTIONS:
            st_manv = 'wait'
        
        
        ''' Setup the constraints object to generate the trajectory '''
        lt_trajs,st_trajs = dict(), dict()
        
        if lt_manv == 'wait':
            lt_traj_constr = WaitTrajectoryConstraints(init_vel=8,waypoints=lt_path,stop_horizon_dist_sampling_range=(20,30),stop_horizon_time_sampling_range=(4,8))
        else:
            lt_traj_constr = ProceedTrajectoryConstraints(waypoints=lt_path,waypoint_vel_sampling_range=lt_vel_pts)
        lt_traj_constr.set_limit_constraints()
        lt_motion = VehicleTrajectoryPlanner(lt_traj_constr,lt_manv,None,6)
        lt_motion.generate_trajectory(True)
        lt_trajs[lt_manv] = lt_motion.all_trajectories
        assert len(lt_trajs[lt_manv]) > 0
        
        
        
        plt.figure()
        for traj_obj,traj_list in lt_trajs[lt_manv].items():
            for traj in traj_list:
                plt.plot([x[0] for x in traj], [x[3] for x in traj])
        plt.title('left turn velocities')
        print('left turn trajectories generated')
        sel_lt_traj = list(zip([float(x[1]) for x in traj], [float(x[2]) for x in traj]))
        
        st_traj_constr = ProceedTrajectoryConstraints(waypoints=st_path,waypoint_vel_sampling_range=st_vel_pts)
        st_traj_constr.set_limit_constraints()
        st_motion = VehicleTrajectoryPlanner(st_traj_constr,st_manv,None,6)
        st_motion.generate_trajectory(True)
        st_trajs[st_manv] = st_motion.all_trajectories
        
        assert len(st_trajs[st_manv]) > 0
        
        plt.figure()
        for traj_obj,traj_list in st_trajs[st_manv].items():
            for traj in traj_list:
                plt.plot([x[0] for x in traj], [x[3] for x in traj])
                f=1
        plt.title('straight through velocities')
        print('straight through trajectories generated')
        
        sel_st_traj = list(zip([float(x[1]) for x in traj], [float(x[2]) for x in traj]))
        
        
        plt.show()
        
        ''' Just take one example, the one in sel_xx_traj, and show the animation '''
        
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(min([x[0] for x in sel_st_traj]+[x[0] for x in sel_lt_traj])-10, max([x[0] for x in sel_st_traj]+[x[0] for x in sel_lt_traj])+10), ylim=(min([x[1] for x in sel_st_traj]+[x[1] for x in sel_lt_traj])-10, max([x[1] for x in sel_st_traj]+[x[1] for x in sel_lt_traj])+10))
        line1, = ax.plot([], [], lw=2)
        line2, = ax.plot([], [], lw=2)
        
        # initialization function: plot the background of each frame
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1,line2,
        
        # animation function.  This is called sequentially
        def animate(i):
            line1.set_data([x[0] for x in sel_lt_traj[:i]], [x[1] for x in sel_lt_traj[:i]])
            line2.set_data([x[0] for x in sel_st_traj[:i]], [x[1] for x in sel_st_traj[:i]])
            return line1,line2,
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(sel_st_traj), interval=20, blit=True, repeat = False) 
        plt.show()
        
if __name__ == '__main__':
    unittest.main()