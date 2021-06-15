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
from collections import defaultdict
import constants, rg_constants
import all_utils
import code_utils.utils as rg_utils
from equilibrium.gametree_objects import UnsupportedLatticeException, UnsupportedAgentObservationException

class ScenarioDef:
    
    def _setup_1shotrepo_state(self,veh_track):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
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
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+file_id+'.db')
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
    
    
    
    def _remove_duplicate(self,path):
        dup_indxs = []
        for dup in sorted(self.list_duplicates([x[0] for x in path])):
            dup_indxs += dup[1][1:]
        _newpath = [x for idx,x in enumerate(path) if idx not in dup_indxs]
        return _newpath,dup_indxs
    
    def get_reasonable_velocities(self,seg,direction=None):
        return rg_utils.get_reasonable_velocities(seg, direction) 
        
    
    ''' from https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list'''
    def list_duplicates(self,seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return ((key,locs) for key,locs in tally.items() 
                                if len(locs)>1)

    def __init__(self,agent_1_id, agent_2_id,file_id,initialize_db,start_ts,freq):
        if agent_1_id is not None and agent_2_id is not None:
            self._two_agent_scenedef(agent_1_id, agent_2_id,file_id,initialize_db,start_ts,freq)
        else:
            ag = agent_1_id if agent_1_id is not None else agent_2_id
            self._one_agent_scenedef(ag, file_id,initialize_db,start_ts,freq)
        
    
    def _one_agent_scenedef(self,agent_2_id,file_id,initialize_db,start_ts,freq):
        self.freq = freq
        self.horizon = int(3/self.freq)
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        ''' Get the track of a representative straight through vehicle to construct a path centerline '''
        q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" T INNER JOIN TRAJECTORIES_0"+constants.CURRENT_FILE_ID+"_EXT E using(track_id,time) WHERE TRACK_ID="+str(agent_2_id)+" AND TIME >= "+str(start_ts)+"  ORDER BY TIME"
        c = conn.cursor()
        c.execute(q_string)
        agent2_res = c.fetchall()
        agent2_path_gates_dir = all_utils.utils.get_path_gates_direction(agent_track=None, agent_id=agent_2_id)
        if len(agent2_res) == 0:
            self.time_crossed = True
        else:
            agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
            agent2_path_segments = [x[9] for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
            agent2_path,dup_indxs = self._remove_duplicate(agent2_path)
            agent2_path_segments = [x for idx,x in enumerate(agent2_path_segments) if idx not in dup_indxs]
                
            if len(agent2_path) == 1:
                self.time_crossed = False
                agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
                agent2_path_segments = [x[9] for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
                agent2_path,dup_indxs = self._remove_duplicate(agent2_path)
                agent2_path_segments = [x for idx,x in enumerate(agent2_path_segments) if idx not in dup_indxs]
                agent2_start_ts = agent2_res[0][6]
                
                oneshot_vehstate = self._setup_1shotrepo_state(agent2_res)
                oneshot_vehstate.current_time = start_ts
                interpolated_track = interpolate_track_info(veh_state = oneshot_vehstate, forward = True, backward = False, partial_track = None)
                agent2_path = agent2_path + [(interpolated_track[1],interpolated_track[2])]
                agent2_path_segments = agent2_path_segments + [agent2_path_segments[-1]] 
                agent_2_attribs = {'x':agent2_res[0][1], 'y':agent2_res[0][1], 'velocity':agent2_res[0][3]/3.6, 'waypoints':agent2_path, 'file_time':start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
                self.agent = VehicleState(agent_2_attribs)
                self.all_agent_trajectories = [(x[1],x[2]) for idx,x in enumerate(agent2_res)]
            else:
                self.time_crossed = False
                agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
                agent2_path_segments = [x[9] for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
                agent2_path,dup_indxs = self._remove_duplicate(agent2_path)
                agent2_path_segments = [x for idx,x in enumerate(agent2_path_segments) if idx not in dup_indxs]
                agent2_start_ts = agent2_res[0][6]
                if agent2_start_ts > start_ts:
                    oneshot_vehstate = self._setup_1shotrepo_state(agent2_res)
                    oneshot_vehstate.current_time = start_ts
                    interpolated_track = interpolate_track_info(veh_state = oneshot_vehstate, forward = False, backward = True, partial_track = None)
                    agent2_path = [(interpolated_track[1],interpolated_track[2])] + agent2_path
                    agent2_path_segments = [agent2_path_segments[0]] + agent2_path_segments 
                    agent_2_attribs = {'x':interpolated_track[1], 'y':interpolated_track[2], 'velocity':interpolated_track[3], 'waypoints':agent2_path, 'file_time':start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
                elif agent2_start_ts == start_ts:
                    agent_2_attribs = {'x':agent2_res[0][1], 'y':agent2_res[0][2], 'velocity':agent2_res[0][3]/3.6, 'waypoints':agent2_path, 'file_time':start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
                else:
                    agent2_res_trunc = None
                    for idx,pt in enumerate(agent2_res):
                        if abs(pt[6]-start_ts) < 0.3:
                            agent2_res_trunc = agent2_res[idx:]
                            break
                    agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res_trunc) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res_trunc)-1, num=10)]]
                    agent2_path_segments = [x[9] for idx,x in enumerate(agent2_res_trunc) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res_trunc)-1, num=10)]]
                    agent2_path,dup_indxs = self._remove_duplicate(agent2_path)
                    agent2_path_segments = [x for idx,x in enumerate(agent2_path_segments) if idx not in dup_indxs]
                    agent_2_attribs = {'x':agent2_res_trunc[0][1], 'y':agent2_res_trunc[0][2], 'velocity':agent2_res_trunc[0][3]/3.6, 'waypoints':agent2_path, 'file_time':start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
                self.agent = VehicleState(agent_2_attribs)
                self.all_agent_trajectories = [(x[1],x[2]) for idx,x in enumerate(agent2_res)]
                
    
    
    
    def _two_agent_scenedef(self,agent_1_id, agent_2_id,file_id,initialize_db,start_ts,freq):
        self.freq = freq
        self.horizon = int(3/self.freq)
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" T INNER JOIN TRAJECTORIES_0"+constants.CURRENT_FILE_ID+"_EXT E using(track_id,time) WHERE TRACK_ID="+str(agent_1_id)+" AND TIME >= "+str(start_ts)+" ORDER BY TIME"
        c.execute(q_string)
        agent1_res = c.fetchall()
        agent1_path = [(x[1],x[2]) for idx,x in enumerate(agent1_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent1_res)-1, num=10)]]
        agent_1_path_segments = [x[9] for idx,x in enumerate(agent1_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent1_res)-1, num=10)]]
        agent1_path,dup_indxs = self._remove_duplicate(agent1_path)
        agent_1_path_segments = [x for idx,x in enumerate(agent_1_path_segments) if idx not in dup_indxs]
        ''' Get the track of a representative straight through vehicle to construct a path centerline '''
        q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" T INNER JOIN TRAJECTORIES_0"+constants.CURRENT_FILE_ID+"_EXT E using(track_id,time) WHERE TRACK_ID="+str(agent_2_id)+" AND TIME >= "+str(start_ts)+"  ORDER BY TIME"
        c.execute(q_string)
        agent2_res = c.fetchall()
        agent1_path_gates_dir = all_utils.utils.get_path_gates_direction(agent_track=None, agent_id=agent_1_id)
        agent2_path_gates_dir = all_utils.utils.get_path_gates_direction(agent_track=None, agent_id=agent_2_id)
        if len(agent1_res)==0 or len(agent2_res)==0:
            self.time_crossed = True
        else:
            self.time_crossed = False
            agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
            agent2_path_segments = [x[9] for idx,x in enumerate(agent2_res) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res)-1, num=10)]]
            agent2_path,dup_indxs = self._remove_duplicate(agent2_path)
            agent2_path_segments = [x for idx,x in enumerate(agent2_path_segments) if idx not in dup_indxs]
            agent2_start_ts = agent2_res[0][6]
            agent1_start_ts = agent1_res[0][6]
            agent_1_attribs = {'x':agent1_res[0][1], 'y':agent1_res[0][2], 'velocity':agent1_res[0][3]/3.6, 'waypoints':agent1_path, 'file_time':agent1_start_ts, 'id':agent_1_id, 'waypoint_segments':agent_1_path_segments, 'direction':agent1_path_gates_dir[-1]}
            if agent2_start_ts > agent1_start_ts:
                oneshot_vehstate = self._setup_1shotrepo_state(agent2_res)
                oneshot_vehstate.current_time = agent1_start_ts
                interpolated_track = interpolate_track_info(veh_state = oneshot_vehstate, forward = False, backward = True, partial_track = None)
                agent2_path = [(interpolated_track[1],interpolated_track[2])] + agent2_path
                agent2_path_segments = [agent2_path_segments[0]] + agent2_path_segments 
                agent_2_attribs = {'x':interpolated_track[1], 'y':interpolated_track[2], 'velocity':interpolated_track[3], 'waypoints':agent2_path, 'file_time':agent1_start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
            elif agent2_start_ts == agent1_start_ts:
                agent_2_attribs = {'x':agent2_res[0][1], 'y':agent2_res[0][2], 'velocity':agent2_res[0][3]/3.6, 'waypoints':agent2_path, 'file_time':agent1_start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
            else:
                agent2_res_trunc = None
                for idx,pt in enumerate(agent2_res):
                    if abs(pt[6]-agent1_start_ts) < 0.3:
                        agent2_res_trunc = agent2_res[idx:]
                        break
                agent2_path = [(x[1],x[2]) for idx,x in enumerate(agent2_res_trunc) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res_trunc)-1, num=10)]]
                agent2_path_segments = [x[9] for idx,x in enumerate(agent2_res_trunc) if idx in [int(y) for y in np.linspace(start=0, stop=len(agent2_res_trunc)-1, num=10)]]
                agent2_path,dup_indxs = self._remove_duplicate(agent2_path)
                agent2_path_segments = [x for idx,x in enumerate(agent2_path_segments) if idx not in dup_indxs]
                agent_2_attribs = {'x':agent2_res_trunc[0][1], 'y':agent2_res_trunc[0][2], 'velocity':agent2_res_trunc[0][3]/3.6, 'waypoints':agent2_path, 'file_time':agent1_start_ts, 'id':agent_2_id, 'waypoint_segments':agent2_path_segments, 'direction':agent2_path_gates_dir[-1]}
            self.agent1 = VehicleState(agent_1_attribs)
            self.agent2 = VehicleState(agent_2_attribs)
            file_id = constants.CURRENT_FILE_ID+'_'+str(agent_1_id)+'_'+str(agent_2_id)+'_'+str(agent1_start_ts).replace('.',',')
            self.assign_emp_traj_length(agent1_res, agent2_res, agent1_start_ts, agent2_start_ts)
            self.all_agent_trajectories = [[(x[1],x[2]) for idx,x in enumerate(agent1_res)],[(x[1],x[2]) for idx,x in enumerate(agent2_res)]]
            if initialize_db:
                self.setup_database(file_id)
    
    def assign_emp_traj_length(self,agent1_res,agent2_res,agent1_start_ts,agent2_start_ts):
        self.agent1_emp_traj, self.agent2_emp_traj = [], []
        start_time,one_step_t,two_step_t,three_step_t = 0,int(1/self.freq),int(2*int(1/self.freq)),self.horizon
        ag1_times = []
        for tp in agent1_res:
            if one_step_t - (tp[6]-agent1_start_ts) < 0.1 and len(self.agent1_emp_traj)==0:
                traj_l = math.hypot(tp[1]-agent1_res[0][1], tp[2]-agent1_res[0][2])
                self.agent1_emp_traj.append(traj_l)
                ag1_times.append(tp[6])
            if two_step_t - (tp[6]-agent1_start_ts) < 0.1 and len(self.agent1_emp_traj)==1:
                traj_l = math.hypot(tp[1]-agent1_res[0][1], tp[2]-agent1_res[0][2])
                self.agent1_emp_traj.append(traj_l)
                ag1_times.append(tp[6])
            if three_step_t - (tp[6]-agent1_start_ts) < 0.1 and len(self.agent1_emp_traj)==2:
                traj_l = math.hypot(tp[1]-agent1_res[0][1], tp[2]-agent1_res[0][2])
                self.agent1_emp_traj.append(traj_l)
                ag1_times.append(tp[6])
                break
        
        ag1_times = [agent1_start_ts] + ag1_times[:-1]
        ag2_emp_tl = []
        agent2_end_ts = agent2_res[-1][6]
        for agt in ag1_times:
            if agt < agent2_start_ts and agt+one_step_t<agent2_start_ts:
                chunk_1,chunk_2 = (agent2_res[0][3]/3.6)*(one_step_t), 0
                ag2_emp_tl.append(chunk_1+chunk_2)
            elif agt <= agent2_start_ts and agt+one_step_t <= agent2_end_ts:
                chunk_1,chunk_2 = (agent2_res[0][3]/3.6)*(agent2_start_ts-agt), 0
                for tp in agent2_res:
                    if (agt+one_step_t - tp[6]) < 0.1:
                        chunk_2 = math.hypot(tp[1]-agent2_res[0][1], tp[2]-agent2_res[0][2])
                        break
                ag2_emp_tl.append(chunk_1+chunk_2)
            elif agent2_start_ts <= agt and agt+one_step_t <= agent2_end_ts:
                chunk_1_idx, chunk_2_idx = 0, 0
                for idx,tp in enumerate(agent2_res):
                    if (agt - tp[6]) < 0.1:
                        chunk_1_idx = idx
                        break
                for idx,tp in enumerate(agent2_res):
                    if (agt+one_step_t - tp[6]) < 0.1:
                        chunk_2_idx = idx
                        break
                trajl = math.hypot(agent2_res[chunk_2_idx][1]-agent2_res[chunk_1_idx][1], agent2_res[chunk_2_idx][2]-agent2_res[chunk_1_idx][2])
                ag2_emp_tl.append(trajl)
            elif agt <= agent2_end_ts and agent2_end_ts <= agt+one_step_t:
                chunk_1,chunk_2 = 0, (agent2_res[-1][3]/3.6)*(agt+one_step_t-agent2_end_ts)
                for tp in agent2_res:
                    if (agt - tp[6]) < 0.1:
                        chunk_1 = math.hypot(tp[1]-agent2_res[-1][1], tp[2]-agent2_res[-1][2])
                        break
                ag2_emp_tl.append(chunk_1+chunk_2)
            elif agt <= agent2_start_ts and agent2_end_ts <= agt+one_step_t:
                chunk_1,chunk_3 = (agent2_res[0][3]/3.6)*(agent2_start_ts-agt), (agent2_res[-1][3]/3.6)*(agt+one_step_t-agent2_end_ts)
                chunk_2 = math.hypot(agent2_res[0][1]-agent2_res[-1][1], agent2_res[0][2]-agent2_res[-1][2])
                ag2_emp_tl.append(chunk_1+chunk_2+chunk_3)
            elif agent2_start_ts == agt and agent2_end_ts == agt+one_step_t:
                traj_l = math.hypot(agent2_res[0][1]-agent2_res[-1][1], agent2_res[0][2]-agent2_res[-1][2])
                ag2_emp_tl.append(trajl)
            elif agent2_start_ts < agt and agent2_end_ts < agt:
                chunk = (agent2_res[-1][3]/3.6)*(one_step_t)
                ag2_emp_tl.append(chunk)
            else:
                raise UnsupportedAgentObservationException(str((agt,one_step_t))+'|'+str(agent2_start_ts)+'|'+str(agent2_end_ts))
        self.agent2_emp_traj =  np.cumsum(ag2_emp_tl).tolist()
        f=1
        
                        
            
            
            
            
        
            
    def setup_trajectory_constraints(self):
        maneuver_constraints = {'agent_1':{'maneuvers':{'wait':None,'turn':None}, 'agent_state':self.agent1},'agent_2':{'maneuvers':{'wait':None,'track_speed':None}, 'agent_state':self.agent2}}
        if len(self.agent1.waypoints) < 5:
            agent1_vel_pts_proc = [(self.agent1.velocity,)] + [(None,) if i != len(np.arange(1,len(self.agent1.waypoints)-1))//2 else self.get_reasonable_velocities(self.agent1.waypoint_segments[i], self.agent1.direction) for i in np.arange(1,len(self.agent1.waypoints)-1)] + [self.get_reasonable_velocities(self.agent1.waypoint_segments[-1], self.agent1.direction)]
        else:
            agent1_vel_pts_proc = [(self.agent1.velocity,)] + [(None,) if i != len(np.arange(1,len(self.agent1.waypoints)-1))//2 else self.get_reasonable_velocities(self.agent1.waypoint_segments[i], self.agent1.direction) for i in np.arange(1,len(self.agent1.waypoints)-1)] + [self.get_reasonable_velocities(self.agent1.waypoint_segments[-1], self.agent1.direction)]
        #agent1_vel_pts_proc = [(self.agent1.velocity,)] + [(None,) for i in np.arange(1,len(self.agent1.waypoints)-1)] + [(4,8.3)]
        agent2_vel_pts_proc = [(self.agent2.velocity,)] + [(None,)]*(len(self.agent2.waypoints)-2) + [(8,17)]
        if len(self.agent2.waypoints) < 5:
            agent2_vel_pts_proc = [(self.agent2.velocity,)] + [(None,) if i != len(np.arange(1,len(self.agent2.waypoints)-1))//2 else self.get_reasonable_velocities(self.agent2.waypoint_segments[i], self.agent2.direction) for i in np.arange(1,len(self.agent2.waypoints)-1)] + [self.get_reasonable_velocities(self.agent2.waypoint_segments[-1], self.agent2.direction)]
        else:
            agent2_vel_pts_proc = [(self.agent2.velocity,)] + [(None,) if i != len(np.arange(1,len(self.agent2.waypoints)-1))//2 else self.get_reasonable_velocities(self.agent2.waypoint_segments[i], self.agent2.direction) for i in np.arange(1,len(self.agent2.waypoints)-1)] + [self.get_reasonable_velocities(self.agent2.waypoint_segments[-1], self.agent2.direction)]
        
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
        agent1_traj_constr_proc = ProceedTrajectoryConstraints(waypoints=self.agent1.waypoints,waypoint_vel_sampling_range=agent1_vel_pts_proc)
        maneuver_constraints['agent_1']['maneuvers']['wait'] = agent1_traj_constr_wait
        maneuver_constraints['agent_1']['maneuvers']['turn'] = agent1_traj_constr_proc
        '''
        plt.plot([x[0] for x in self.agent1.waypoints], [x[1] for x in self.agent1.waypoints], c = 'blue',marker='o')
        plt.plot([x[0] for x in self.agent2.waypoints], [x[1] for x in self.agent2.waypoints] , c = 'red',marker='o')
        plt.axis('equal')
        plt.show()
        '''
        return maneuver_constraints
        
        
                
class SyntheticScenarioDef:
    
    def __init__(self,agent_id, agent_init_velocity_mps, agent_waypoints,agent_waypoint_segments, direction, file_id,initialize_db,start_ts,freq):
        assert len(agent_waypoints[0]) == 2, "Agent waypoints should contain (x,v) information"
        agent_attribs = {'x':agent_waypoints[0][0], 'y':agent_waypoints[0][1], 'velocity':agent_init_velocity_mps, 'waypoints':agent_waypoints, 'file_time':start_ts, 'id':agent_id, 'waypoint_segments':agent_waypoint_segments, 'direction':direction}
        self.agent = VehicleState(agent_attribs)
        
        
        
                       
        