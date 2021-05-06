'''
Created on Mar 31, 2021

@author: Atrisha
'''
import sqlite3
import numpy as np
import constants
from all_utils.utils import interpolate_track_info
from motion_planners.planning_objects import VehicleState as OneshotRepoVehicleState
import ast
from planners.planning_objects import VehicleState
from planners.trajectory_planner import WaitTrajectoryConstraints, ProceedTrajectoryConstraints
import math
from operator import itemgetter
import matplotlib.pyplot as plt

class ScenarioDef:
    
    def _setup_1shotrepo_state(self,veh_track):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+'769'+'\\uni_weber_'+'769'+'.db')
        c = conn.cursor()
        oneshot_vehstate = OneshotRepoVehicleState()
        oneshot_vehstate.id = veh_track[0][0]
        q_string = "select TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ FROM TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRACK_ID="+str(oneshot_vehstate.id)
        c.execute(q_string)
        res =c.fetchone()
        oneshot_vehstate.segment_seq = ast.literal_eval(res[0])
        q_string = "select * from v_times where track_id="+str(oneshot_vehstate.id)
        c.execute(q_string)
        res = c.fetchone()
        oneshot_vehstate.entry_exit_time = (res[1],res[2])
        return oneshot_vehstate
    
    def setup_database(self,file_id): 
        conn = sqlite3.connect('D:\\repeated_games_data\\'+file_id+'.db')
        c = conn.cursor()
        q_string = "CREATE TABLE IF NOT EXISTS TRAJECTORIES ( `TRACK_ID` INTEGER, `X` NUMERIC, `Y` NUMERIC, `SPEED` NUMERIC, `TAN_ACC` NUMERIC, `LAT_ACC` NUMERIC, `TIME` NUMERIC, `ANGLE` NUMERIC, `TRAFFIC_REGIONS` TEXT )"
        c.execute(q_string)
        q_string = "CREATE TABLE IF NOT EXISTS TRAJECTORY_METADATA ( `TRAJ_ID` INTEGER, `INIT_POS_X` NUMERIC, `INIT_POS_Y` NUMERIC, `INIT_VEL` NUMERIC, `INIT_ACC` NUMERIC, `FINAL_VEL` NUMERIC, `MANEUVER` TEXT, `MANEUVER_MODE` TEXT, `AGENT_TYPE` TEXT, `INIT_TIME` INTEGER, `PARENT_TRAJ_ID` INTEGER )"
        c.execute(q_string)
        q_string = "CREATE TABLE IF NOT EXISTS TRAJ_INTERACTIONS ( `INTERAC_ID` INTEGER, `TRAJ_1_ID` INTEGER, `TRAJ_2_ID` INTEGER, `DISTANCE_GAP` NUMERIC, `TIME_GAP` NUMERIC, `TRAJ_1_LENGTH` NUMERIC, `TRAJ_2_LENGTH` NUMERIC )"
        c.execute(q_string)
        q_string = "CREATE INDEX IF NOT EXISTS `trajectories_trajid_time` ON `TRAJECTORIES` (`TRACK_ID` ,`TIME` )"
        conn.commit()
        q_string = "DELETE FROM TRAJECTORIES"
        c.execute(q_string)
        q_string = "DELETE FROM TRAJECTORY_METADATA"
        c.execute(q_string)
        q_string = "DELETE FROM TRAJ_INTERACTIONS"
        c.execute(q_string)
        conn.commit()
        
    def __init__(self,agent_1_id, agent_2_id,file_id,initialize_db):
        
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" WHERE TRAJECTORIES_0769.TRACK_ID="+str(agent_1_id)+" ORDER BY TIME"
        c.execute(q_string)
        agent1_res = c.fetchall()
        agent1_path = [(x[1],x[2]) for idx,x in enumerate(agent1_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent1_res)-1, num=10)]]
        
        ''' Get the track of a representative straight through vehicle to construct a path centerline '''
        q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" WHERE TRAJECTORIES_0769.TRACK_ID="+str(agent_2_id)+" ORDER BY TIME"
        c.execute(q_string)
        agent2_res = c.fetchall()
        agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
        agent2_start_ts = agent2_res[0][6]
        agent1_start_ts = agent1_res[0][6]
        agent_1_attribs = {'x':agent1_res[0][1], 'y':agent1_res[0][2], 'velocity':agent1_res[0][3]/3.6, 'waypoints':agent1_path, 'file_time':agent1_start_ts, 'id':agent_1_id}
        if agent2_start_ts > agent1_start_ts:
            oneshot_vehstate = self._setup_1shotrepo_state(agent2_res)
            oneshot_vehstate.current_time = agent1_start_ts
            interpolated_track = interpolate_track_info(veh_state = oneshot_vehstate, forward = False, backward = True, partial_track = None)
            agent2_path = [(interpolated_track[1],interpolated_track[2])] + agent2_path
            agent_2_attribs = {'x':interpolated_track[1], 'y':interpolated_track[2], 'velocity':interpolated_track[3], 'waypoints':agent2_path, 'file_time':agent1_start_ts, 'id':agent_2_id}
        elif agent2_start_ts == agent1_start_ts:
            agent_2_attribs = {'x':agent2_res[0][1], 'y':agent2_res[0][2], 'velocity':agent2_res[0][3]/3.6, 'waypoints':agent2_path, 'file_time':agent1_start_ts, 'id':agent_2_id}
        else:
            agent2_res_trunc = None
            for idx,pt in enumerate(agent2_res):
                if abs(pt[6]-agent1_start_ts) < 0.3:
                    agent2_res_trunc = agent2_res[idx:]
                    break
            agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res_trunc) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res_trunc)-1, num=10)]]
            agent_2_attribs = {'x':agent2_res_trunc[0][1], 'y':agent2_res_trunc[0][2], 'velocity':agent2_res_trunc[0][3]/3.6, 'waypoints':agent2_path, 'file_time':agent1_start_ts, 'id':agent_2_id}
        self.agent1 = VehicleState(agent_1_attribs)
        self.agent2 = VehicleState(agent_2_attribs)
        file_id = constants.CURRENT_FILE_ID+'_'+str(agent_1_id)+'_'+str(agent_2_id)+'_'+str(agent1_start_ts).replace('.',',')
        if initialize_db:
            self.setup_database(file_id)
            
    def setup_trajectory_constraints(self):
        maneuver_constraints = {'agent_1':{'maneuvers':{'wait':None,'turn':None}, 'agent_state':self.agent1},'agent_2':{'maneuvers':{'wait':None,'track_speed':None}, 'agent_state':self.agent2}}
        agent1_vel_pts_proc = [(self.agent1.velocity,)] + [(None,) if i != len(np.arange(1,len(self.agent1.waypoints)-1))//2 else (2,4) for i in np.arange(1,len(self.agent1.waypoints)-1)] + [(4,8.3)]
        agent2_vel_pts_proc = [(self.agent2.velocity,)] + [(None,)]*(len(self.agent2.waypoints)-2) + [(8,17)]
        min_distgp_indx = min(enumerate([math.hypot(x[0]-y[1], x[0]-y[1]) for x,y in zip(self.agent1.waypoints,self.agent2.waypoints)]), key=itemgetter(1))[0] 
        agent_2_dist_2_stop = math.hypot(self.agent2.waypoints[min_distgp_indx][0]-self.agent2.waypoints[0][0], self.agent2.waypoints[min_distgp_indx][1]-self.agent2.waypoints[0][1])
        agent_2_time_2_stop = agent_2_dist_2_stop/self.agent2.velocity if self.agent2.velocity !=0 else 2
        
        
        agent2_traj_constr_wait = WaitTrajectoryConstraints(init_vel=self.agent2.velocity,waypoints=self.agent2.waypoints,stop_horizon_dist_sampling_range=(agent_2_dist_2_stop-5,agent_2_dist_2_stop+5),stop_horizon_time_sampling_range=(agent_2_time_2_stop-2,agent_2_time_2_stop+2))
        agent2_traj_constr_proc = ProceedTrajectoryConstraints(waypoints=self.agent2.waypoints,waypoint_vel_sampling_range=agent2_vel_pts_proc)
        maneuver_constraints['agent_2']['maneuvers']['wait'] = agent2_traj_constr_wait
        maneuver_constraints['agent_2']['maneuvers']['track_speed'] = agent2_traj_constr_proc
        
        agent_1_dist_1_stop = math.hypot(self.agent1.waypoints[min_distgp_indx][0]-self.agent1.waypoints[0][0], self.agent1.waypoints[min_distgp_indx][1]-self.agent1.waypoints[0][1])
        agent_1_time_1_stop = agent_1_dist_1_stop/self.agent1.velocity if self.agent1.velocity !=0 else 2
        agent1_traj_constr_wait = WaitTrajectoryConstraints(init_vel=self.agent1.velocity,waypoints=self.agent1.waypoints,stop_horizon_dist_sampling_range=(1,5),stop_horizon_time_sampling_range=(1,4))
        agent2_traj_constr_proc = ProceedTrajectoryConstraints(waypoints=self.agent1.waypoints,waypoint_vel_sampling_range=agent1_vel_pts_proc)
        maneuver_constraints['agent_1']['maneuvers']['wait'] = agent1_traj_constr_wait
        maneuver_constraints['agent_1']['maneuvers']['turn'] = agent2_traj_constr_proc
        '''
        plt.plot([x[0] for x in self.agent1.waypoints], [x[1] for x in self.agent1.waypoints], c = 'blue',marker='o')
        plt.plot([x[0] for x in self.agent2.waypoints], [x[1] for x in self.agent2.waypoints] , c = 'red',marker='o')
        plt.axis('equal')
        plt.show()
        '''
        return maneuver_constraints
        
        
                
                
        