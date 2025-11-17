'''
Created on Apr 13, 2021

@author: Atrisha
'''
import unittest
from planners.trajectory_planner import TrajectoryPlanner, VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner
from planners.trajectory_planner import WaitTrajectoryConstraints, ProceedTrajectoryConstraints
from planners.planning_objects import TrajectoryConstraintsFactory
from motion_planners.planning_objects import VehicleState
from maps.States import SyntheticScenarioDef
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import constants
from maps.States import ScenarioDef
from equilibrium.game_tree import Actions
from visualizer.visualizer import plot_traffic_regions
from rg_visualizer import UniWeberAnalytics
import all_utils


WAIT_ACTIONS = ['yield-to-merging','wait_for_lead_to_cross','wait-for-oncoming','decelerate-to-stop','wait-on-red','wait-for-pedestrian']

class TestVehicleTurn():
    @unittest.skip
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
        
        
        analytics_obj = UniWeberAnalytics('769')
        trajs_list = [sel_st_traj,sel_lt_traj]
        analytics_obj.animate_scene(trajs_list)
        
        
        
        ''' Just take one example, the one in sel_xx_traj, and show the animation '''
        '''
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(min([x[0] for x in sel_st_traj]+[x[0] for x in sel_st_traj])-10, max([x[0] for x in sel_st_traj]+[x[0] for x in sel_lt_traj])+10), ylim=(min([x[1] for x in sel_st_traj]+[x[1] for x in sel_lt_traj])-10, max([x[1] for x in sel_st_traj]+[x[1] for x in sel_lt_traj])+10))
        line1, = ax.plot([], [], lw=2)
        line2, = ax.plot([], [], lw=2)
        #plot_traffic_regions(ax)
        plt.xlim(538780, 538890)
        plt.ylim(4813970, 4814055)
        img = plt.imread("D:\\behavior modeling\\background.jpg")
        ax.imshow(img, extent=[538780, 538890, 4813970, 4814055])
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
                                   frames=len(sel_st_traj), interval=100, blit=True, repeat = True) 
        plt.show()
        '''

class TestScenario():
    
    def test_scenario(self):
        
        dataset_file_id = '769'
        init_time = 0
        scene_def = ScenarioDef(2,28,'769',initialize_db=True,start_ts=4.004)
        maneuver_constraints = scene_def.setup_trajectory_constraints()
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        agent1_init_vel = maneuver_constraints['agent_1']['agent_state'].velocity
        agent2_init_vel = maneuver_constraints['agent_2']['agent_state'].velocity
        time_horizon = 6
        acts = Actions(maneuver_constraints)
        acts.generate_actions(init_time,agent1_init_vel,agent2_init_vel,time_horizon,True)
        
    
    def test_scene_769_1(self):
        
        scene_def = ScenarioDef(agent_1_id=229, agent_2_id=None,file_id='775',initialize_db=False,start_ts= 227.227,freq=0.5)
        if scene_def.time_crossed:
            print('failed')
            raise Exception()
        else:
            ag_obj = scene_def.agent
        '''
        scene_def = ScenarioDef(agent_1_id=43, agent_2_id=None,file_id='769',initialize_db=False,start_ts=43.043,freq=0.5)
        if scene_def.time_crossed:
            print('failed')
            raise Exception()
        else:
            lead_ag_obj = scene_def.agent
        '''
        lead_ag_obj = None
        constr = TrajectoryConstraintsFactory.get_constraint_object(maneuver='track_speed', ag_obj=ag_obj, lead_ag_obj=lead_ag_obj)
        #constr.set_limit_constraints()
        constr.set_limit_constraints(max_lat_acc_lims=5.6,max_vel_lims=22,max_acc_lims=6,max_jerk_lims=3)
        agent1_motion = VehicleTrajectoryPlanner(traj_constr_obj=constr,maneuver= 'track_speed', mode=None, horizon=6)
        agent1_motion.generate_trajectory(True)
        assert hasattr(agent1_motion, 'all_trajectories') and len(agent1_motion.all_trajectories) > 0
        f=1
            
    def test_synthetic(self):
        occluding_vehicle_initial_speed = 9.981109108284521
        agent_waypoints = [(538835.8104362666, 4813998.205484587), (538837.0054616176, 4813996.489007992), (538838.2005760442, 4813994.771817474), (538839.3670737282, 4813993.090424765), (538841.0003698844, 4813990.715326011), (538842.407808143, 4813988.626150823), (538843.7656073741, 4813986.498339003)]
        agent_waypoint_segments = ['l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r', 'l_n_s_r']
        initialize_db = False
        freq = 0.5
        scene_def = SyntheticScenarioDef(-1, occluding_vehicle_initial_speed, agent_waypoints, agent_waypoint_segments, 'L_N_S', 769, initialize_db, 0, freq)
        ag_obj = scene_def.agent
             
        constr = TrajectoryConstraintsFactory.get_constraint_object(maneuver='track_speed', ag_obj=ag_obj, lead_ag_obj=None)
        constr.set_limit_constraints()
        agent1_motion = VehicleTrajectoryPlanner(traj_constr_obj=constr,maneuver= 'track_speed', mode=None, horizon=6)
        agent1_motion.generate_trajectory(True)
        assert hasattr(agent1_motion, 'all_trajectories') and len(agent1_motion.all_trajectories) > 0     
        
    def test_single_trajectory_error(self):
        initialize_db = False
        freq = 0.5
        occluding_vehicle_initial_speed = 3.2471666666666668
        agent_waypoints = [(538893.3, 4814021.84), (538893.36, 4814027.04), (538872.38, 4814030.39), (538856.54, 4814022.62), (538846.5, 4813984.48)]
        agent_waypoint_segments = ['ln_e_1', 'ln_e_1', 'ln_e_1', 'ln_e_1', 'exec-turn_e']
        # Uncomment this set of waypoint and wapoint segments to see that removing 'exec-turn_e' allows this to work
        # agent_waypoints = [(538893.3, 4814021.84), (538893.36, 4814027.04), (538872.38, 4814030.39), (538856.54, 4814022.62)]
        # agent_waypoint_segments = ['ln_e_1', 'ln_e_1', 'ln_e_1', 'ln_e_1']
        scene_def = SyntheticScenarioDef(75, occluding_vehicle_initial_speed, agent_waypoints, agent_waypoint_segments, 'L_E_S', 782, initialize_db, 0, freq)
        constants.CURRENT_FILE_ID = '769'
        vs = all_utils.utils.setup_vehicle_state(124, 110.11)
        p,g,d = all_utils.utils.get_path_gates_direction(None,124)
        vs.set_gates(g)
        gate_crossing_times = all_utils.utils.gate_crossing_times(vs)
        vs.set_gate_crossing_times(gate_crossing_times)
        r = all_utils.utils.get_relevant_agents(vs)
        ag_obj = scene_def.agent
        initialize_db = False
        freq = 0.5
        occluding_vehicle_initial_speed = 0.9428055555555555
        agent_waypoints = [(538851.32, 4814020.17), (538849.41, 4814019.3), (538849.42, 4814019.3), (538849.45, 4814019.31), (538848.21, 4814018.72), (538847.04, 4814018.11), (538845.84, 4814017.46), (538844.65, 4814016.72), (538840.13, 4814007.88), (538849.95, 4813982.52)]
        agent_waypoint_segments = ['prep-turn_e', 'prep-turn_e', 'prep-turn_e', 'prep-turn_e', 'prep-turn_e', 'prep-turn_e', 'prep-turn_e', 'prep-turn_e', 'exec-turn_e', 'ln_s_-1']
        scene_def1 = SyntheticScenarioDef(75, occluding_vehicle_initial_speed, agent_waypoints, agent_waypoint_segments, 'L_E_S', 782, initialize_db, 0, freq)
        lead_ag_obj = scene_def1.agent
        constr = TrajectoryConstraintsFactory.get_constraint_object(maneuver='follow_lead_into_intersection', ag_obj=ag_obj, lead_ag_obj=lead_ag_obj)
        constr.set_limit_constraints(max_lat_acc_lims=5.6,max_vel_lims=22,max_acc_lims=6,max_jerk_lims=3)
        
        agent_motion = VehicleTrajectoryPlanner(traj_constr_obj=constr,maneuver='follow_lead_into_intersection', mode=None, horizon=6)
        agent_motion.generate_trajectory(True)
        assert hasattr(agent_motion, 'all_trajectories') and len(agent_motion.all_trajectories) > 0
                                    
                        
  

    

if __name__ == '__main__':
    #unittest.main()
    test = TestScenario()
    test.test_single_trajectory_error()