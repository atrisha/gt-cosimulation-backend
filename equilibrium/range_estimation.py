'''
Created on Apr 7, 2021

@author: Atrisha
'''
import numpy as np
import sqlite3
import itertools
from repeated_game import Utilities
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from equilibrium.gametree_objects import TrajectoryCache, TrajectoryFragment

        
show_plots = False

class RangeEstimationModel:
    
    
    '''
    This is a model for estimating the range of the minimum and maximum distance gaps 
    given the 2s, 4s, 6s distances of a trajectory.
    
    The basic idea of the model is to predict the range based on a k-NN type model.
    '''
    
    def fit(self,distgap_data):
        self.model_2sec, self.model_4sec, self.model_6sec = dict(), dict(), dict()
        for manv,dg_data in distgap_data.items():
            self.distgap_data = dg_data
            dist_2sec = np.round(np.asarray([x[0][1] for x in self.distgap_data]),1)
            dist_4sec = np.round(np.asarray([x[0][1]+x[0][3] for x in self.distgap_data]),1)
            dist_6sec = np.round(np.asarray([x[0][1]+x[0][3]+x[0][5] for x in self.distgap_data]),1)
            Y = [x[1] for x in self.distgap_data]
            dist_2sec_points = np.arange(np.amin(dist_2sec),np.amax(dist_2sec)+.1,.1)
            dist_2sec_model = np.full((len(dist_2sec_points),2 ), np.nan)
            dist_2sec_model[:,0] = np.inf
            dist_2sec_model[:,1] = -np.inf
            
            dist_4sec_points = np.arange(np.amin(dist_4sec),np.amax(dist_4sec)+.1,.1)
            dist_4sec_model = np.full((len(dist_4sec_points),2), np.nan)
            dist_4sec_model[:,0] = np.inf
            dist_4sec_model[:,1] = -np.inf
            
            dist_6sec_points = np.arange(np.amin(dist_6sec),np.amax(dist_6sec)+.1,.1)
            dist_6sec_model = np.full((len(dist_6sec_points),2), np.nan)
            dist_6sec_model[:,0] = np.inf
            dist_6sec_model[:,1] = -np.inf
            
            for d in self.distgap_data:
                val_2sec = round(d[0][1],1)
                try:
                    val_2sec_idx = 0 if val_2sec-np.amin(dist_2sec_points)==0 or np.amax(dist_2sec_points)-np.amin(dist_2sec_points)==0 else int((val_2sec-np.amin(dist_2sec_points)) / (np.amax(dist_2sec_points)-np.amin(dist_2sec_points)) * (dist_2sec_model.shape[0]-1))
                    dist_2sec_model[val_2sec_idx] = np.asarray([min(dist_2sec_model[val_2sec_idx,0],d[1]), max(dist_2sec_model[val_2sec_idx,1],d[1])])
                    val_4sec = round(d[0][3]+d[0][1],1)
                    val_4sec_idx = 0 if val_4sec-np.amin(dist_4sec_points) == 0 or np.amax(dist_4sec_points)-np.amin(dist_4sec_points) == 0 else int((val_4sec-np.amin(dist_4sec_points)) / (np.amax(dist_4sec_points)-np.amin(dist_4sec_points)) * (dist_4sec_model.shape[0]-1))
                    dist_4sec_model[val_4sec_idx] = np.asarray([min(dist_4sec_model[val_4sec_idx,0],d[1]), max(dist_4sec_model[val_4sec_idx,1],d[1])])
                
                    val_6sec = round(d[0][5]+d[0][3]+d[0][1],1)
                    val_6sec_idx = 0 if val_6sec-np.amin(dist_6sec_points)==0 or np.amax(dist_6sec_points)-np.amin(dist_6sec_points)==0 else int((val_6sec-np.amin(dist_6sec_points)) / (np.amax(dist_6sec_points)-np.amin(dist_6sec_points)) * (dist_6sec_model.shape[0]-1))
                    dist_6sec_model[val_6sec_idx] = np.asarray([min(dist_6sec_model[val_6sec_idx,0],d[1]), max(dist_6sec_model[val_6sec_idx,1],d[1])])
                except:
                    f=1
                    raise
                
                
            self.model_2sec[manv] = ((np.amin(dist_2sec),np.amax(dist_2sec)),dist_2sec_model)
            self.model_4sec[manv] = ((np.amin(dist_4sec),np.amax(dist_4sec)),dist_4sec_model)
            self.model_6sec[manv] = ((np.amin(dist_6sec),np.amax(dist_6sec)),dist_6sec_model)   
        
    def _interpolate(self,idx,data_arr):
        next_pt, prev_pt = None, None
        pred_pt = None
        if idx < 0:
            for bidx in np.arange(0,data_arr.shape[0]):
                if data_arr[bidx,0] != np.inf and data_arr[bidx,1] != -np.inf: 
                    prev_pt = data_arr[bidx]
                    break
        elif idx >= data_arr.shape[0]:
            for fidx in np.arange(data_arr.shape[0]-1,-1,-1):
                if data_arr[fidx,0] != np.inf and data_arr[fidx,1] != -np.inf: 
                    next_pt = data_arr[fidx]
                    break
        else:  
            for fidx in np.arange(idx,data_arr.shape[0]):
                if data_arr[fidx,0] != np.inf and data_arr[fidx,1] != -np.inf: 
                    next_pt = data_arr[fidx]
                    break
            for bidx in np.arange(idx,-1,-1):
                if data_arr[bidx,0] != np.inf and data_arr[bidx,1] != -np.inf: 
                    prev_pt = data_arr[bidx]
                    break
        assert prev_pt is not None or next_pt is not None, "No interpolation reference found"
        if next_pt is not None and prev_pt is not None:
            pred_pt = np.asarray([prev_pt[0] + (((idx-bidx)/(fidx-bidx))*(next_pt[0]-prev_pt[0])),prev_pt[1] + (((idx-bidx)/(fidx-bidx))*(next_pt[1]-prev_pt[1]))])
        else:
            pred_pt = prev_pt if prev_pt is not None else next_pt
        return pred_pt
        
        
    def predict(self,inp,manv_map):
        X,manvX = [x[0] for x in inp], [x[1] for x in inp]
        assert 1 <= len(X) <= 2, "Length of X should be 1 or 2 with 2,4 sec distances in meters" 
        val_2sec_idx, val_4sec_idx, val_6sec_idx = None, None, None
        pred_range_2s, pred_range_4s, pred_range_6s = None, None, None
        val_2sec = round(X[0],1)
        prediction_details = dict()
        
        try:
            wait_model = self.model_2sec[(manv_map['wait'],manv_map['wait'],manv_map['wait'])]
            proceed_model = self.model_2sec[(manv_map['proceed'],manv_map['proceed'],manv_map['proceed'])]
            val_2sec_idx_wait = 0 if val_2sec-wait_model[0][0]==0 or wait_model[0][1]==wait_model[0][0] else int((val_2sec-wait_model[0][0]) / (wait_model[0][1]-wait_model[0][0]) * (wait_model[1].shape[0]-1))
            val_2sec_idx_proceed = 0 if val_2sec-proceed_model[0][0]==0 or proceed_model[0][1]==proceed_model[0][0] else int((val_2sec-proceed_model[0][0]) / (proceed_model[0][1]-proceed_model[0][0]) * (proceed_model[1].shape[0]-1))
            if val_2sec_idx_wait>0 and val_2sec_idx_wait<wait_model[1].shape[0] and wait_model[1][val_2sec_idx_wait,0] != np.inf and wait_model[1][val_2sec_idx_wait,1] != -np.inf:
                pred_range_2s_wait = wait_model[1][val_2sec_idx_wait]
            else:
                pred_range_2s_wait = self._interpolate(val_2sec_idx_wait, wait_model[1])
            if val_2sec_idx_proceed>0 and val_2sec_idx_proceed<proceed_model[1].shape[0] and proceed_model[1][val_2sec_idx_proceed,0] != np.inf and proceed_model[1][val_2sec_idx_proceed,1] != -np.inf:
                pred_range_2s_proceed = proceed_model[1][val_2sec_idx_proceed]
            else:
                pred_range_2s_proceed = self._interpolate(val_2sec_idx_proceed, proceed_model[1])
            prediction_details = {manv_map['wait'] : pred_range_2s_wait, 
                                       manv_map['proceed'] : pred_range_2s_proceed}
        except KeyError:
            ''' some of the manuver combination may not have a model so assign nans'''
            prediction_details = {manv_map['wait'] : (np.nan,np.nan), 
                                       manv_map['proceed'] : (np.nan,np.nan)}
        if len(X) > 1:
            try:
                val_4sec = round(X[1],1)
                wait_model = self.model_4sec[(manvX[0],manv_map['wait'],manv_map['wait'])]
                proceed_model = self.model_4sec[(manvX[0],manv_map['proceed'],manv_map['proceed'])]
                val_4sec_idx_wait = 0 if val_4sec-wait_model[0][0]==0 or wait_model[0][1]==wait_model[0][0] else int((val_4sec-wait_model[0][0]) / (wait_model[0][1]-wait_model[0][0]) * (wait_model[1].shape[0]-1))
                val_4sec_idx_proceed = 0 if val_4sec-proceed_model[0][0]==0 or proceed_model[0][1]==proceed_model[0][0]  else int((val_4sec-proceed_model[0][0]) / (proceed_model[0][1]-proceed_model[0][0]) * (proceed_model[1].shape[0]-1))
                if val_4sec_idx_wait>0 and val_4sec_idx_wait<wait_model[1].shape[0] and wait_model[1][val_4sec_idx_wait,0] != np.inf and wait_model[1][val_4sec_idx_wait,1] != -np.inf:
                    pred_range_4s_wait = wait_model[1][val_4sec_idx_wait]
                else:
                    pred_range_4s_wait = self._interpolate(val_4sec_idx_wait, wait_model[1])
                if val_4sec_idx_proceed>0 and val_4sec_idx_proceed<proceed_model[1].shape[0] and proceed_model[1][val_4sec_idx_proceed,0] != np.inf and proceed_model[1][val_4sec_idx_proceed,1] != -np.inf:
                    pred_range_4s_proceed = proceed_model[1][val_4sec_idx_proceed]
                else:
                    pred_range_4s_proceed = self._interpolate(val_4sec_idx_proceed, proceed_model[1])
                if not np.isnan(prediction_details[manv_map['wait']][0]):
                    prediction_details = {manv_map['wait'] : (min(pred_range_4s_wait[0],prediction_details[manv_map['wait']][0]), min(pred_range_4s_wait[1],prediction_details[manv_map['wait']][1])), 
                                           manv_map['proceed'] : (min(pred_range_4s_proceed[0],prediction_details[manv_map['proceed']][0]), min(pred_range_4s_proceed[1],prediction_details[manv_map['proceed']][1]))}
                else:
                    prediction_details = {manv_map['wait'] : pred_range_4s_wait, 
                                           manv_map['proceed'] : pred_range_4s_proceed}
            except KeyError:
                ''' some of the manuver combination may not have a model so assign nans'''
                prediction_details = {manv_map['wait'] : (np.nan,np.nan), 
                                           manv_map['proceed'] : (np.nan,np.nan)}
        '''
        if len(X) > 2:
            val_6sec = round(X[2],1)
            val_6sec_idx = int((val_6sec-self.model_6sec[0][0]) / (self.model_6sec[0][1]-self.model_6sec[0][0]) * (self.model_6sec[1].shape[0]-1))
        if val_4sec_idx is not None:
            if self.model_4sec[1][val_4sec_idx,0] != np.inf and self.model_4sec[1][val_4sec_idx,1] != -np.inf:
                pred_range_4s = self.model_4sec[1][val_4sec_idx]
            else:
                pred_range_4s = self._interpolate(val_4sec_idx, self.model_4sec[1])
        if val_6sec_idx is not None:
            if self.model_6sec[1][val_6sec_idx,0] != np.inf and self.model_6sec[1][val_6sec_idx,1] != -np.inf:
                pred_range_6s = self.model_6sec[1][val_6sec_idx]
            else:
                pred_range_6s = self._interpolate(val_6sec_idx, self.model_6sec[1])
        if len(X) == 1:
            return pred_range_2s
        elif len(X) == 2:
            return (min(pred_range_2s[0],pred_range_4s[0]), max(pred_range_2s[1],pred_range_4s[1]))
        else:
            return (min(pred_range_2s[0],pred_range_4s[0],pred_range_6s[0]), max(pred_range_2s[1],pred_range_4s[1],pred_range_6s[1]))
        '''
        return prediction_details    
            
        
        
        
    
class MinDistanceGapModel:
    
    def __init__(self, file_id):
        self.file_id = file_id
    
    
    def build_agent_trajectories(self,ag_type):
        trajectories = dict()
        trajectories = dict()
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+self.file_id+'.db')
        c = conn.cursor()
        ''' get all vehicle trajectories '''
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID, TRAJECTORY_METADATA.MANEUVER,TRAJECTORY_METADATA.MANEUVER_MODE from TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.INIT_TIME=0 AND TRAJECTORY_METADATA.AGENT_TYPE='"+ag_type+"'"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            tf0_2 = TrajectoryFragment((0,2),row[0],row[1],row[2],0)
            tf2_4 = TrajectoryFragment((2,4),row[0],row[1],row[2],0)
            tf4_6 = TrajectoryFragment((4,6),row[0],row[1],row[2],0)
            tf2_4.set_next_fragment(tf4_6)
            tf0_2.set_next_fragment(tf2_4)
            #trajectories.append(tf0_2)
            if (row[1],row[1],row[1]) not in trajectories:
                trajectories[(row[1],row[1],row[1])] = [tf0_2]
            else:
                trajectories[(row[1],row[1],row[1])].append(tf0_2) 
        
        ''' one change trajectories '''
        
        q_string = "select * from TRAJECTORY_METADATA M1 INNER JOIN TRAJECTORY_METADATA M2 ON M1.TRAJ_ID = M2.PARENT_TRAJ_ID and M1.INIT_TIME=0 and M2.INIT_TIME=2 AND M1.AGENT_TYPE='"+ag_type+"' \
                    UNION \
                    select * from TRAJECTORY_METADATA M1 INNER JOIN TRAJECTORY_METADATA M2 ON M1.TRAJ_ID = M2.PARENT_TRAJ_ID and M1.INIT_TIME=0 and M2.INIT_TIME=4 AND M1.AGENT_TYPE='"+ag_type+"'"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            frag_2_init_time = row[20]
            frag_1_traj_id = row[0]
            frag_1_manv = row[6]
            frag_1_manv_mode = row[7]
            frag_2_traj_id = row[11]
            frag_2_manv = row[17]
            frag_2_manv_mode = row[18]
            frag_1_init_time = 0
            frag_2_init_time = row[20]
            if frag_2_init_time == 2:
                tf0_2 = TrajectoryFragment((0,2),frag_1_traj_id,frag_1_manv,frag_1_manv_mode,frag_1_init_time)
                tf2_4 = TrajectoryFragment((2,4),frag_2_traj_id,frag_2_manv,frag_2_manv_mode,frag_2_init_time)
                tf4_6 = TrajectoryFragment((4,6),frag_2_traj_id,frag_2_manv,frag_2_manv_mode,frag_2_init_time)
                tf2_4.set_next_fragment(tf4_6)
                tf0_2.set_next_fragment(tf2_4)
                #trajectories.append(tf0_2)
                if (frag_1_manv,frag_2_manv,frag_2_manv) not in trajectories:
                    trajectories[(frag_1_manv,frag_2_manv,frag_2_manv)] = [tf0_2]
                else:
                    trajectories[(frag_1_manv,frag_2_manv,frag_2_manv)].append(tf0_2) 
            else:
                tf0_2 = TrajectoryFragment((0,2),frag_1_traj_id,frag_1_manv,frag_1_manv_mode,frag_1_init_time)
                tf2_4 = TrajectoryFragment((2,4),frag_1_traj_id,frag_1_manv,frag_1_manv_mode,frag_1_init_time)
                tf4_6 = TrajectoryFragment((4,6),frag_2_traj_id,frag_2_manv,frag_2_manv_mode,frag_2_init_time)
                tf2_4.set_next_fragment(tf4_6)
                tf0_2.set_next_fragment(tf2_4)
                #trajectories.append(tf0_2)
                if (frag_1_manv,frag_1_manv,frag_2_manv) not in trajectories:
                    trajectories[(frag_1_manv,frag_1_manv,frag_2_manv)] = [tf0_2]
                else:
                    trajectories[(frag_1_manv,frag_1_manv,frag_2_manv)].append(tf0_2) 
                
        ''' two change trajectories '''
        
        q_string = "select M1.TRAJ_ID,M1.MANEUVER,M1.MANEUVER_MODE, M2.TRAJ_ID,M2.MANEUVER,M2.MANEUVER_MODE, M3.TRAJ_ID,M3.MANEUVER,M3.MANEUVER_MODE,M1.INIT_TIME,M2.INIT_TIME,M3.INIT_TIME from TRAJECTORY_METADATA M1 INNER JOIN TRAJECTORY_METADATA M2 INNER JOIN TRAJECTORY_METADATA M3 ON M1.TRAJ_ID = M2.PARENT_TRAJ_ID and M2.TRAJ_ID = M3.PARENT_TRAJ_ID AND M1.AGENT_TYPE='"+ag_type+"'"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            tf0_2 = TrajectoryFragment((0,2),row[0],row[1],row[2],row[9])
            tf2_4 = TrajectoryFragment((2,4),row[3],row[4],row[5],row[10])
            tf4_6 = TrajectoryFragment((4,6),row[6],row[7],row[8],row[11])
            tf2_4.set_next_fragment(tf4_6)
            tf0_2.set_next_fragment(tf2_4)
            #trajectories.append(tf0_2)
            if (row[1],row[4],row[7]) not in trajectories:
                trajectories[(row[1],row[4],row[7])] = [tf0_2]
            else:
                trajectories[(row[1],row[4],row[7])].append(tf0_2) 
        return trajectories
    
    def populate_trajectories(self):
        pedestrian_trajectories = self.build_agent_trajectories('agent_2')
        vehicle_trajectories = self.build_agent_trajectories('agent_1')
        ''' sample the trajectories to build the model taking into account
            adequate coverage of maneuver combinations 
        '''
        traj_cache = dict()
        for x in [(0,0,2),(0,2,4),(0,4,6),(2,2,4),(2,4,6),(4,4,6)]:
            traj_cache[x] = TrajectoryCache(init_time=x[0],time_range=(x[1],x[2]),ag_type=None,file_id=self.file_id)
        for m,t in pedestrian_trajectories.items():
            if len(t) > 50:
                len_t = len(t)
                #print('agent_2',m,len_t)
                selected_indices = np.random.uniform(low=0,high=len_t, size=min(50,len_t),)
                pedestrian_trajectories[m] = [t[int(idx)] for idx in selected_indices]
        for m,t in vehicle_trajectories.items():
            if len(t) > 50:
                len_t = len(t)
                #print('agent_1',m,len_t)
                selected_indices = np.random.uniform(low=0,high=len_t, size=min(50,len_t),)
                vehicle_trajectories[m] = [t[int(idx)] for idx in selected_indices]
        for k,v in pedestrian_trajectories.items():
            #print('agent_2',k,len(v))
            for traj in v:
                traj.load(self.file_id,traj_cache)
        for k,v in vehicle_trajectories.items():
            #print('agent_1',k,len(v))
            for traj in v:
                traj.load(self.file_id,traj_cache)
        self.pedestrian_trajectories = pedestrian_trajectories
        self.vehicle_trajectories = vehicle_trajectories
        '''
        plt.figure()
        plt.plot([x[1] for x in vehicle_trajectories[('turn', 'turn', 'turn')][0].loaded_traj],[x[2] for x in vehicle_trajectories[('wait', 'wait', 'wait')][0].loaded_traj],color='red')
        plt.plot([x[1] for x in pedestrian_trajectories[('wait', 'wait', 'wait')][0].loaded_traj],[x[2] for x in pedestrian_trajectories[('wait', 'wait', 'wait')][0].loaded_traj],color='blue')
        plt.axis('equal')
        plt.show()
        '''
        
    def _Xvector(self,traj_obj):
        ''' 2sec_vel, 2sec_dist, 4sec_vel, 4sec_dist, 6sec_vel, 6sec_dist'''
        vel_2sec = traj_obj.loaded_traj_frag[-1][3]
        dist_2sec = math.hypot(traj_obj.loaded_traj_frag[0][1]-traj_obj.loaded_traj_frag[-1][1], traj_obj.loaded_traj_frag[0][2]-traj_obj.loaded_traj_frag[-1][2])
        vel_4sec = traj_obj.next_fragment.loaded_traj_frag[-1][3]
        dist_4sec = math.hypot(traj_obj.next_fragment.loaded_traj_frag[0][1]-traj_obj.next_fragment.loaded_traj_frag[-1][1], traj_obj.next_fragment.loaded_traj_frag[0][2]-traj_obj.next_fragment.loaded_traj_frag[-1][2])
        vel_6sec = traj_obj.next_fragment.next_fragment.loaded_traj_frag[-1][3]
        dist_6sec = math.hypot(traj_obj.next_fragment.next_fragment.loaded_traj_frag[0][1]-traj_obj.next_fragment.next_fragment.loaded_traj_frag[-1][1], traj_obj.next_fragment.next_fragment.loaded_traj_frag[0][2]-traj_obj.next_fragment.next_fragment.loaded_traj_frag[-1][2])
        _X = (vel_2sec,dist_2sec,vel_4sec,dist_4sec,vel_6sec,dist_6sec)
        return _X
        
        
        
    def build_model(self):
        utl_obj = Utilities()
        self.populate_trajectories()
        veh_distgap_data,ped_distgap_data = dict(), dict()
        ''' It's okay, this whole loop is limited to (8*50)^2 = 1,60,000 iterations'''
        ct,N = 0,160000
        for vt_manv,vt_obj_list in self.vehicle_trajectories.items():
            for vt_obj in vt_obj_list:
                vt = [(x[1],x[2]) for x in vt_obj.loaded_traj]
                vt_X = self._Xvector(vt_obj)
                for pt_manv,pt_obj_list in self.pedestrian_trajectories.items():
                    for pt_obj in pt_obj_list:
                        ct += 1
                        #print(ct,'/',N)
                        pt = [(x[1],x[2]) for x in pt_obj.loaded_traj]
                        min_dist_gap = utl_obj.calc_dist_gap(vt, pt, (0,1))
                        pt_X = self._Xvector(pt_obj)
                        if vt_manv not in veh_distgap_data:
                            veh_distgap_data[vt_manv] = [(vt_X,min_dist_gap)]
                        else:
                            veh_distgap_data[vt_manv].append((vt_X,min_dist_gap))
                        if pt_manv not in ped_distgap_data:
                            ped_distgap_data[pt_manv] = [(pt_X,min_dist_gap)]
                        else:
                            ped_distgap_data[pt_manv].append((pt_X,min_dist_gap))
        if show_plots:
            '''
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plt.title('vehicle 2 sec')
            cmhot = plt.get_cmap("hot")
            ax.scatter([x[0][1] for x in veh_distgap_data],[x[0][0] for x in veh_distgap_data],[x[1] for x in veh_distgap_data], c=[x[0][0] for x in veh_distgap_data], cmap=cmhot)
            '''
            plt.figure()
            plt.title('vehicle 2 sec')
            plt.plot([x[0][1] for x in veh_distgap_data],[x[1] for x in veh_distgap_data],'.')
            plt.figure()
            plt.title('vehicle 4 sec')
            plt.plot([x[0][1]+x[0][3] for x in veh_distgap_data],[x[1] for x in veh_distgap_data],'.')
            plt.figure()
            plt.title('vehicle 6 sec')
            plt.plot([x[0][1]+x[0][3]++x[0][5] for x in veh_distgap_data],[x[1] for x in veh_distgap_data],'.')
            plt.figure()
            plt.title('pedestrian 2 sec')
            plt.plot([x[0][1] for x in ped_distgap_data],[x[1] for x in ped_distgap_data],'.')
            plt.figure()
            plt.title('pedestrian 4 sec')
            plt.plot([x[0][1]+x[0][3] for x in ped_distgap_data],[x[1] for x in ped_distgap_data],'.')
            plt.figure()
            plt.title('pedestrian 6 sec')
            plt.plot([x[0][1]+x[0][3]++x[0][5] for x in ped_distgap_data],[x[1] for x in ped_distgap_data],'.')
            
            plt.show()
        '''
        selected_indices = np.random.randint(low=0,high=len(veh_distgap_data), size=10000)
        veh_X = np.asarray([list(x[0]) for i,x in enumerate(veh_distgap_data) if i in selected_indices])
        veh_Y = np.asarray([x[1] for i,x in enumerate(veh_distgap_data) if i in selected_indices])
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(veh_X, veh_Y)
        sc = gpr.score(veh_X, veh_Y)
        pr = gpr.predict(veh_X[:2,:], return_std=True)
        print(sc)
        print(pr)
        '''
        r_est_obj_veh = RangeEstimationModel()
        r_est_obj_veh.fit(veh_distgap_data)
        self.distgap_model_agent_1 = r_est_obj_veh
        
        r_est_obj_ped = RangeEstimationModel()
        r_est_obj_ped.fit(ped_distgap_data)
        self.distgap_model_agent_2 = r_est_obj_ped
        
        
        
        
if __name__ == '__main__':         
    m = MinDistanceGapModel()
    m.build_model()           
        