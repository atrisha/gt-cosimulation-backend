'rt_exec-turn_w''''
from all_utils import utils
Created on Feb 8, 2021

@author: Atrisha
'''
import sqlite3
import numpy as np
import constants
import ast 
from collections import OrderedDict
from collections import Counter
from all_utils import utils
import csv
import rg_constants
from shapely.geometry import LineString, Polygon, Point
import os
from maps.parse_inD import parse_scenes
import copy

def right_turn_interaction_scenarios():
    tot = 0
    for file_id in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from v_TIMES"
        c.execute(q_string)
        res = c.fetchall()
        entry_exit_times = {row[0]:(row[1],row[2]) for row in res}
        q_string = "select * from TRAJECTORIES_0"+str(file_id)+"_EXT WHERE ASSIGNED_SEGMENT IN ('ln_s_-2' ,'ln_s_-1','ln_e_-2' ,'ln_e_-1') ORDER BY TRACK_ID,TIME"
        c.execute(q_string)
        res = c.fetchall()
        exit_lane_entry_time =  dict()
        for row in res:
            if row[0] not in exit_lane_entry_time:
                exit_lane_entry_time[row[0]] = row[1]
        
        q_string = "SELECT * from TRAJECTORIES_0"+str(file_id)+"_EXT ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        scenario_dict = OrderedDict()
        ct,N = 0,len(res)
        print('processing',file_id)
        for row in res:
            ct += 1
            #print(file_id,ct,'/',N)
            if row[1] not in scenario_dict:
                scenario_dict[row[1]] = [(row[0],row[2])]
            else:
                if row[0] not in [x[0] for x in scenario_dict[row[1]]]:
                    scenario_dict[row[1]].append((row[0],row[2]))
        agent_entries = []
        
        #print(scenario_dict.keys())
        for k in list(scenario_dict.keys()):
            if k==110.7106:
                _e = scenario_dict[k]
                f=1
            #pedestrian_info = utils.setup_pedestrian_info(k)
            el_ctr = Counter([x[1] for x in scenario_dict[k]])
            #print(k)
            for rtag in scenario_dict[k]:
                if ((rtag[1] == 'rt_prep-turn_w' and el_ctr['rt_exec-turn_w']==0 and el_ctr['ln_s_-2']==0) or \
                    (rtag[1] == 'rt_exec-turn_w' and el_ctr['ln_s_-2']==0) or \
                    (rtag[1] == 'ln_w_4' and el_ctr['rt_prep-turn_w']==0 and el_ctr['rt_exec-turn_w']==0 and el_ctr['ln_s_-2']==0)):
                    for stag in scenario_dict[k]:
                        #vs = utils.setup_vehicle_state(rtag[0], k)
                        #ra = utils.get_relevant_pedestrians(vs, pedestrian_info)
                        if((stag[1] == 'ln_n_3' and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0) or \
                            (stag[1] == 'l_n_s_r' and el_ctr['ln_s_-2']==0)):
                                '''
                                (stag[1] == 'ln_n_2' and el_ctr['ln_n_3']==0 and el_ctr['l_n_s_l']==0 and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0 and el_ctr['ln_s_-1']==0) or \
                                (stag[1] == 'l_n_s_l' and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0 and el_ctr['ln_s_-1']==0))
                                '''
                                if (rtag[0],stag[0]) not in [(x[0],x[1]) for x in agent_entries]:
                                    if rtag[0] in [x[0] for x in agent_entries]:
                                        _l = [x[0] for x in agent_entries]
                                        last_index = len(_l) - 1 - _l[::-1].index(rtag[0])
                                        if agent_entries[last_index][1] not in exit_lane_entry_time:
                                            continue
                                        else:
                                            ra_ex_time = exit_lane_entry_time[agent_entries[last_index][1]]
                                    else:
                                        ra_ex_time = entry_exit_times[stag[0]][0]-2
                                    agent_entries.append((rtag[0],stag[0],max(entry_exit_times[rtag[0]][0],ra_ex_time),'ws'))
                                    #print((rtag[0],stag[0],k,'rt_st_w_s'))
                elif ((rtag[1] == 'rt_prep-turn_s' and el_ctr['rt_exec-turn_s']==0 and el_ctr['ln_e_-2']==0) or \
                    (rtag[1] == 'rt_exec-turn_s' and el_ctr['ln_e_-2']==0) or \
                    (rtag[1] == 'ln_s_4' and el_ctr['rt_prep-turn_s']==0 and el_ctr['rt_exec-turn_s']==0 and el_ctr['ln_e_-2']==0)):
                    for stag in scenario_dict[k]:
                        #vs = utils.setup_vehicle_state(rtag[0], k)
                        #ra = utils.get_relevant_pedestrians(vs, pedestrian_info)
                        if((stag[1] == 'ln_w_3' and el_ctr['l_w_e_r']==0 and el_ctr['ln_e_-2']==0) or \
                            (stag[1] == 'l_w_e_r' and el_ctr['ln_e_-2']==0)):
                                '''
                                (stag[1] == 'ln_n_2' and el_ctr['ln_n_3']==0 and el_ctr['l_n_s_l']==0 and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0 and el_ctr['ln_s_-1']==0) or \
                                (stag[1] == 'l_n_s_l' and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0 and el_ctr['ln_s_-1']==0))
                                '''
                                if (rtag[0],stag[0]) not in [(x[0],x[1]) for x in agent_entries]:
                                    if rtag[0] in [x[0] for x in agent_entries]:
                                        _l = [x[0] for x in agent_entries]
                                        last_index = len(_l) - 1 - _l[::-1].index(rtag[0])
                                        if agent_entries[last_index][1] not in exit_lane_entry_time:
                                            continue
                                        else:
                                            ra_ex_time = exit_lane_entry_time[agent_entries[last_index][1]]
                                    else:
                                        ra_ex_time = entry_exit_times[stag[0]][0]-2
                                    agent_entries.append((rtag[0],stag[0],max(entry_exit_times[rtag[0]][0],ra_ex_time),'se'))
                                    #print((rtag[0],stag[0],k,'rt_st_s_e'))                
        #for entr in agent_entries:
        #    print(entr)
        conn.close()
        with open(rg_constants.SCENE_OUT_PATH, mode='a') as scene_file:
            sc_writer = csv.writer(scene_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for entr in agent_entries:
                sc_writer.writerow([file_id,'rt',entr[3],entr[0],entr[1],entr[2]])
        print('Total',file_id, len(agent_entries))
        tot += len(agent_entries)
        
    print('grand total',tot)
    
def can_exclude(file_id,agent_id,task,direction,curr_time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    q_string = "SELECT "+direction+", TIME FROM TRAFFIC_LIGHTS WHERE TIME - "+str(curr_time)+" <= 0 AND FILE_ID = 769 ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    prev_signal = (res[-1][0],res[-1][1]) if len(res) > 0 else None
    q_string = "SELECT "+direction+", TIME FROM TRAFFIC_LIGHTS WHERE TIME - "+str(curr_time)+" > 0 AND FILE_ID = 769 ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    next_signal = (res[0][0],res[0][1]) if len(res) > 0 else None
    excl = False
    tol = 2
    if prev_signal is not None:
        if prev_signal[0] == 'R':
            '''
            if (prev_signal[1]==0 or curr_time-prev_signal[1] > 2):
                if (next_signal is None) or (next_signal is not None and next_signal[1]-curr_time >2):
                    excl = True
            '''
            excl = True
    gate_map = {'L_N_S':(130,18),
                'L_W_E':(34,132),
                'L_E_W':(131,63),
                'L_S_W':(131,63),
                'L_W_N':(73,73),
                }
    if not excl:
        q_string = "select * FROM GATE_CROSSING_EVENTS WHERE GATE_CROSSING_EVENTS.GATE_ID="+str(gate_map[direction][0])+" OR GATE_CROSSING_EVENTS.GATE_ID="+str(gate_map[direction][1])+" ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        res_ag_list = [x[3] for x in res]
        ag_crossing_idx = res_ag_list.index(agent_id) if agent_id in res_ag_list else None
        if ag_crossing_idx is not None and ag_crossing_idx > 0:
            time_gap = res[ag_crossing_idx][6] - res[ag_crossing_idx-1][6]  
            if time_gap <= 2:
                excl = True
    return excl
    
        
    
def left_turn_interaction_scenarios():
    tot = 0
    for file_id in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from v_TIMES"
        c.execute(q_string)
        res = c.fetchall()
        entry_exit_times = {row[0]:(row[1],row[2]) for row in res}
        q_string = "select * from TRAJECTORIES_0"+str(file_id)+"_EXT WHERE ASSIGNED_SEGMENT IN ('ln_s_-2' ,'ln_s_-1','ln_w_-2' ,'ln_w_-1') ORDER BY TRACK_ID,TIME"
        c.execute(q_string)
        res = c.fetchall()
        exit_lane_entry_time =  dict()
        for row in res:
            if row[0] not in exit_lane_entry_time:
                exit_lane_entry_time[row[0]] = row[1]
        
        
        q_string = "SELECT * from TRAJECTORIES_0"+str(file_id)+"_EXT ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        scenario_dict = OrderedDict()
        ct,N = 0,len(res)
        print('processing',file_id)
        for row in res:
            ct += 1
            #print(file_id,ct,'/',N)
            if row[1] not in scenario_dict:
                scenario_dict[row[1]] = [(row[0],row[2])]
            else:
                if row[0] not in [x[0] for x in scenario_dict[row[1]]]:
                    scenario_dict[row[1]].append((row[0],row[2]))
        agent_entries = []
        
        #print(scenario_dict.keys())
        for k in list(scenario_dict.keys()):
            if k==110.7106:
                _e = scenario_dict[k]
                f=1
            #pedestrian_info = utils.setup_pedestrian_info(k)
            el_ctr = Counter([x[1] for x in scenario_dict[k]])
            #print(k)
            for rtag in scenario_dict[k]:
                if ((rtag[1] == 'prep-turn_s' and el_ctr['prep-turn_s']==1 and el_ctr['exec-turn_s']==0) or \
                    (rtag[1] == 'exec-turn_s' and el_ctr['ln_w_-2']==0 and el_ctr['ln_w_-1']==0) or \
                    (rtag[1] == 'ln_s_1' and el_ctr['ln_s_1'] == 1 and el_ctr['prep-turn_s']==0 and el_ctr['exec-turn_s']==0) or\
                    (rtag[1] == 'ln_w_-1' and el_ctr['ln_w_-1']==1) or (rtag[1] == 'ln_w_-2' and el_ctr['ln_w_-2']==1)):
                    if not can_exclude(file_id, rtag[0],'lt','L_S_W', k):
                        for stag in scenario_dict[k]:
                            #vs = utils.setup_vehicle_state(rtag[0], k)
                            #ra = utils.get_relevant_pedestrians(vs, pedestrian_info)
                            if((stag[1] == 'ln_n_3' and el_ctr['l_n_s_r']==0 ) or \
                                (stag[1] == 'l_n_s_r' and el_ctr['ln_s_-2']==0) or \
                                (stag[1] == 'ln_n_2' and el_ctr['l_n_s_l']==0 ) or \
                                (stag[1] == 'l_n_s_l' and el_ctr['ln_s_-1']==0)):
                                    '''
                                    (stag[1] == 'ln_n_2' and el_ctr['ln_n_3']==0 and el_ctr['l_n_s_l']==0 and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0 and el_ctr['ln_s_-1']==0) or \
                                    (stag[1] == 'l_n_s_l' and el_ctr['l_n_s_r']==0 and el_ctr['ln_s_-2']==0 and el_ctr['ln_s_-1']==0))
                                    '''
                                    if not can_exclude(file_id, stag[0], 'st', 'L_N_S', k):
                                        if (rtag[0],stag[0]) not in [(x[0],x[1]) for x in agent_entries]:
                                            if rtag[0] in [x[0] for x in agent_entries]:
                                                _l = [x[0] for x in agent_entries]
                                                last_index = len(_l) - 1 - _l[::-1].index(rtag[0])
                                                if agent_entries[last_index][1] not in exit_lane_entry_time:
                                                    continue
                                                else:
                                                    ra_ex_time = exit_lane_entry_time[agent_entries[last_index][1]]
                                            else:
                                                ra_ex_time = entry_exit_times[stag[0]][0]-2
                                            agent_entries.append((rtag[0],stag[0],max(entry_exit_times[rtag[0]][0],ra_ex_time),'se'))
                                            print((rtag[0],stag[0],max(entry_exit_times[rtag[0]][0],ra_ex_time),'se'))  
                elif ((rtag[1] == 'prep-turn_w' and el_ctr['exec-turn_w']==0) or \
                    (rtag[1] == 'exec-turn_w' and el_ctr['ln_n_-2']==0 and el_ctr['ln_n_-1']==0) or \
                    (rtag[1] == 'ln_w_1' and el_ctr['prep-turn_w']==0 and el_ctr['exec-turn_w']==0) or\
                    (rtag[1] == 'ln_n_-1' and el_ctr['ln_n_-1']==1) or (rtag[1] == 'ln_n_-2' and el_ctr['ln_n_-2']==1)):
                    if not can_exclude(file_id, rtag[0],'lt','L_W_N', k):
                        for stag in scenario_dict[k]:
                            #vs = utils.setup_vehicle_state(rtag[0], k)
                            #ra = utils.get_relevant_pedestrians(vs, pedestrian_info)
                            if((stag[1] == 'ln_e_3' and el_ctr['l_e_w_r']==0 ) or \
                                (stag[1] == 'l_e_w_r' and el_ctr['ln_w_-2']==0) or \
                                (stag[1] == 'ln_e_2' and el_ctr['l_e_w_l']==0 ) or \
                                (stag[1] == 'l_e_w_l' and el_ctr['ln_w_-1']==0)):
                                    if rtag[0]==14 and (stag[0]==6 or stag[0]==4 or stag[0]==12):
                                        f=1
                                    if not can_exclude(file_id, stag[0],'st','L_E_W', k):
                                        if (rtag[0],stag[0]) not in [(x[0],x[1]) for x in agent_entries]:
                                            if rtag[0] in [x[0] for x in agent_entries]:
                                                _l = [x[0] for x in agent_entries]
                                                last_index = len(_l) - 1 - _l[::-1].index(rtag[0])
                                                if agent_entries[last_index][1] not in exit_lane_entry_time:
                                                    continue
                                                else:
                                                    ra_ex_time = exit_lane_entry_time[agent_entries[last_index][1]]
                                            else:
                                                ra_ex_time = entry_exit_times[stag[0]][0]-2
                                            agent_entries.append((rtag[0],stag[0],max(entry_exit_times[rtag[0]][0],ra_ex_time),'wn'))
                                            print((rtag[0],stag[0],max(entry_exit_times[rtag[0]][0],ra_ex_time),'wn'))
        #for entr in agent_entries:
        #    print(entr)
        conn.close()
        print('Total',file_id, len(agent_entries))
        tot += len(agent_entries)
        with open(rg_constants.SCENE_OUT_PATH, mode='a') as scene_file:
            sc_writer = csv.writer(scene_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for entr in agent_entries:
                sc_writer.writerow([file_id,'lt',entr[3],entr[0],entr[1],entr[2]])
    print('grand total',tot)
    
    

    
def pedestrian_interaction_scenarios():
    tot = 0
    all_scenes = []
    for file_id in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "SELECT * FROM L1_SOLUTIONS WHERE L1_SOLUTIONS.MODEL_PARMS LIKE '%l3_sampling=BASELINE,%' and model='maxmax';"
        c.execute(q_string)
        res = c.fetchall()
        N = len(res)
        file_scenes = []
        for runidx,row in enumerate(res):
            print('processing',file_id,runidx,'/',N)
            rule_strat = ast.literal_eval(row[6])
            row_agid = int(row[1])
            if len(rule_strat) >0:
                agidx = None
                if len(rule_strat) > 0:
                    for _idx,_es in enumerate(rule_strat[0]):
                        _thisagid = int(_es[3:6]) if int(_es[6:9]) == 0 else int(_es[6:9])
                        if _thisagid == row_agid:
                            agidx = _idx
                            break
                    rule_acts = list(set([int(x[agidx][9:11]) for x in rule_strat]))
                    if 11 in rule_acts:
                        if row[1] not in [x[1] for x in file_scenes]:
                            file_scenes.append((row[0],row[1],row[2]))
                        #print(scenes[-1])
        tot += len(file_scenes)
        all_scenes.extend(file_scenes)
    for sc in all_scenes:
        print(sc)
    print('Total', tot)


class inD_Scenarios:
    
    def __init__(self,init_scenefile,vehicle_ids):
        track_map = OrderedDict()
        self.init_scenefile = init_scenefile
        self.file_init_end = {2:7, 7:18, 30:33}
        
    def setup_database(self,file_id): 
        conn = sqlite3.connect(os.path.join(rg_constants.ind_dataset_path,str(file_id)+'.db'))
        c = conn.cursor()
        q_string = 'CREATE TABLE TRAJECTORIES_'+str(file_id)+' ( `TRACK_ID` INTEGER, `X` NUMERIC, `Y` NUMERIC, `SPEED` NUMERIC, `TAN_ACC` NUMERIC, `LAT_ACC` NUMERIC, `TIME` NUMERIC, `ANGLE` NUMERIC, `TRAFFIC_REGIONS` TEXT )'
        try:
            c.execute(q_string)
        except sqlite3.OperationalError:
            q_string = 'DELETE FROM TRAJECTORIES_'+str(file_id)
            c.execute(q_string)
        q_string = 'CREATE TABLE TRAJECTORIES_'+str(file_id)+'_EXT ( `TRACK_ID` INTEGER, `TIME` NUMERIC, `ASSIGNED_SEGMENT` TEXT, `L1_ACTION` TEXT, `L2_ACTION` TEXT )'
        try:
            c.execute(q_string)
        except sqlite3.OperationalError:
            q_string = 'DELETE FROM TRAJECTORIES_'+str(file_id)+'_EXT'
            c.execute(q_string)
        conn.commit()
        
    def insert_data(self,file_id,i_string,ins_list):
        conn = sqlite3.connect(os.path.join(rg_constants.ind_dataset_path,str(file_id)+'.db'))
        c = conn.cursor()
        c.executemany(i_string,ins_list)
        conn.commit()
        conn.close()
    
    def interpolate_segments(self,file_id):
        conn = sqlite3.connect(os.path.join(rg_constants.ind_dataset_path,str(file_id)+'.db'))
        c = conn.cursor()
        q_string = 'SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_'+str(file_id)+'_EXT WHERE ASSIGNED_SEGMENT IS NULL ORDER BY TRACK_ID;'
        c.execute(q_string)
        ids_2_interpolate = c.fetchall()
        up_list = []
        for _e in ids_2_interpolate:
            id = _e[0]
            q_string = 'SELECT TIME,ASSIGNED_SEGMENT FROM TRAJECTORIES_'+str(file_id)+'_EXT WHERE TRACK_ID='+str(id)+' AND ASSIGNED_SEGMENT IS NOT NULL ORDER BY TIME'
            c.execute(q_string)
            res = c.fetchone()
            f_time, seg = res[0], res[1]
            q_string = 'SELECT TIME,ASSIGNED_SEGMENT FROM TRAJECTORIES_'+str(file_id)+'_EXT WHERE TRACK_ID='+str(id)+' AND ASSIGNED_SEGMENT IS NULL and TIME < '+str(f_time)+' ORDER BY TIME;'
            c.execute(q_string)
            res = c.fetchall()
            for row in res:
                up_list.append((seg,row[0],id))
        u_string = 'UPDATE TRAJECTORIES_'+str(file_id)+'_EXT SET ASSIGNED_SEGMENT=? WHERE TIME=? AND TRACK_ID=?'
        c.executemany(u_string,up_list)
        conn.commit()
        conn.close()
                
    
    def load_tracks(self):
        scene_data = dict()
        seg_regions = {(30,32):{'ag1':{'ln_n_1':None,'prep-turn_n':None,'exec-turn_n':None,'ln_e_-1':None},'ag2':{'ln_s_2':None,'l_n_s_l':None,'ln_n_-1':None}},
                              (7,17):{'ag1':{'ln_n_1':None,'prep-turn_n':None,'exec-turn_n':None,'ln_e_-2':None},'ag2':{'ln_s_2':None,'l_s_n_l':None,'ln_n_-1':None}},
                              (2,6):{'ag1':{'ln_s_4':None,'rt_prep-turn_s':None,'rt_exec-turn_s':None,'ln_e_-1':None},'ag2':{'ln_w_3':None,'l_w_e_r':None,'ln_e_-2':None}}}
        for k,v in seg_regions.items():
            for ag,segs in v.items():
                for seg in segs.keys():
                    reg = parse_scenes(k[0],[seg])
                    segs[seg] = copy.deepcopy(reg[0])
            
        with open('D:\\repeated_games_data\\intersection_dataset\\ind_scenario_files.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['file_id'] not in scene_data:
                    scene_data[row['file_id']] = [[row['ag1_id']],[row['ag2_id']]]
                else:
                    scene_data[row['file_id']][0].append(row['ag1_id'])
                    scene_data[row['file_id']][1].append(row['ag2_id'])
        
        for file_id,agids in scene_data.items():
            traj_ins_data,traj_ext_ins_data = [], []
            scene_fileid = file_id
            xUtmOrigin,yUtmOrigin,frame_rate = None, None, None
            last_seg_info = dict()
            with open('D:\\datasets\\inD-tools\\drone-dataset-tools-master\\drone-dataset-tools-master\\data\\'+scene_fileid+'_recordingMeta.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    #print(f'Column names are {", ".join(row)}')
                    xUtmOrigin,yUtmOrigin,frame_rate = float(row['xUtmOrigin']), float(row['yUtmOrigin']), float(row['frameRate'])
                    break
            with open('D:\\datasets\\inD-tools\\drone-dataset-tools-master\\drone-dataset-tools-master\\data\\'+scene_fileid+'_tracks.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if row['trackId'] in agids[0] or row['trackId'] in agids[1]:
                        f=1
                        track_id = int(row['trackId'])
                        x,y = xUtmOrigin + float(row['xCenter']),yUtmOrigin + float(row['yCenter']) 
                        speed, tan_acc, lat_acc = float(row['lonVelocity'])*3.6, float(row['lonAcceleration']), float(row['latAcceleration'])
                        t = float(row['frame'])/frame_rate
                        ang = np.deg2rad(float(row['heading']))
                        traf_reg = None
                        assigned_seg = None
                        pt = Point(x,y)
                        
                        for k,v in seg_regions.items():
                            if k[0] <= int(file_id) <= k[1]:
                                if row['trackId'] in agids[0]:
                                    for seg, reg in v['ag1'].items():
                                        if pt.within(reg):
                                            assigned_seg = seg
                                else:
                                    for seg, reg in v['ag2'].items():
                                        if pt.within(reg):
                                            assigned_seg = seg
                        if assigned_seg is not None:
                            last_seg_info[track_id] = assigned_seg
                        else:
                            if track_id in last_seg_info:
                                assigned_seg = last_seg_info[track_id]
                        traj_ins_data.append((track_id,x,y,speed,tan_acc,lat_acc,t,ang,traf_reg))
                        traj_ext_ins_data.append((track_id,t,assigned_seg,None,None))
                        f=1
            self.setup_database(file_id)
            i_string = 'REPLACE INTO TRAJECTORIES_'+str(file_id)+' VALUES (?,?,?,?,?,?,?,?,?)'
            self.insert_data(file_id, i_string, traj_ins_data)
            i_string = 'REPLACE INTO TRAJECTORIES_'+str(file_id)+'_EXT VALUES (?,?,?,?,?)'
            self.insert_data(file_id, i_string, traj_ext_ins_data)
            self.interpolate_segments(file_id)
            f=1
    
    def right_turn_scenarios(self):
        track_meta_map = dict()
        lt_focus_regions, st_focus_regions = parse_scenes(30,['ln_n_1']), parse_scenes(30,['ln_s_2'])
        
        tot = 0
        for scene_fileid in ['0'+str(x) if x < 10 else str(x) for x in  np.arange(self.init_scenefile,self.file_init_end[self.init_scenefile])]:
            vehicles = dict()
            track_info = dict()
            #print('processing', scene_fileid)
            xUtmOrigin,yUtmOrigin = None, None
            with open('D:\\datasets\\inD-tools\\drone-dataset-tools-master\\drone-dataset-tools-master\\data\\'+scene_fileid+'_recordingMeta.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    #print(f'Column names are {", ".join(row)}')
                    xUtmOrigin,yUtmOrigin = float(row['xUtmOrigin']), float(row['yUtmOrigin'])
                    break
                    
            with open('D:\\datasets\\inD-tools\\drone-dataset-tools-master\\drone-dataset-tools-master\\data\\'+scene_fileid+'_tracksMeta.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if row['class'] not in ['pedestrian','bicycle']:
                        track_meta_map[row['trackId']] = (row['initialFrame'], row['finalFrame']) 
                    line_count += 1
            vehicles[scene_fileid] = {'lt':dict(),'st':dict()}
            
            #print('UTM origins are',xUtmOrigin,yUtmOrigin)
            with open('D:\\datasets\\inD-tools\\drone-dataset-tools-master\\drone-dataset-tools-master\\data\\'+scene_fileid+'_tracks.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        #print(f'Column names are {", ".join(row)}')
                        f=1
                    if row['trackId'] in track_meta_map:
                        for lt_region in lt_focus_regions:
                            if Point(float(row['xCenter']) + xUtmOrigin, float(row['yCenter']) + yUtmOrigin).within(lt_region):
                                if row['trackId'] not in vehicles[scene_fileid]['lt']:
                                    vehicles[scene_fileid]['lt'][row['trackId']] = track_meta_map[row['trackId']] 
                        for st_region in st_focus_regions:
                            pt = Point(float(row['xCenter']) + xUtmOrigin, float(row['yCenter']) + yUtmOrigin)
                            if pt.within(st_region):
                                if row['trackId'] not in vehicles[scene_fileid]['st']:
                                    vehicles[scene_fileid]['st'][row['trackId']] = (track_meta_map[row['trackId']], pt) 
                                 
                    line_count += 1
        
            scene_map = dict()
            #print('left turning')
            for k,v in vehicles.items():
                for k1,v1 in v['lt'].items():
                    for k2,v2 in v['st'].items():
                        if v1[0] <= v2[0][0] <= v1[1]:
                            if k not in scene_map:
                                scene_map[k] = dict()
                                
                            if k1 not in scene_map[k]:
                                scene_map[k][k1] = OrderedDict()
                                track_info[k1] = dict()
                            if k2 not in scene_map[k][k1]:
                                current_vehs = [x[2] for x in scene_map[k][k1].values()]
                                if len(current_vehs) > 0:
                                    if min([LineString([v2[1],x]).length for x in current_vehs]) > 20:
                                        scene_map[k][k1][k2] = OrderedDict()
                                        scene_map[k][k1][k2] = (v1,v2[0],v2[1],min([LineString([v2[1],x]).length for x in current_vehs])) 
                                        track_info[k2] = dict()
                                        tot += 1
                                else:
                                    scene_map[k][k1][k2] = OrderedDict()
                                    scene_map[k][k1][k2] = (v1,v2[0],v2[1],None)
                                    track_info[k2] = dict() 
                                    tot += 1
                                
            for k,v in scene_map.items():
                for k1,v1 in v.items():
                    for k2,v2 in v1.items():
                        print(k,k1,k2)
        print('total',tot)
        
if __name__ == '__main__':
    sc = inD_Scenarios(30,[])
    #sc.right_turn_scenarios()
    sc.load_tracks()