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
from maps.parse_inD import parse_scenes

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
    
    def right_turn_scenarios(self):
        track_meta_map = dict()
        lt_region = parse_scenes(30)
        scenes = dict()
        for scene_fileid in ['0'+str(x) if x < 10 else str(x) for x in np.arange(30,33)]:
            print('processing', scene_fileid)
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
                    if line_count == 0:
                        #print(f'Column names are {", ".join(row)}')
                        f=1
                    if row['class'] not in ['pedestrian','bicycle']:
                        track_meta_map[row['trackId']] = (row['initialFrame'], row['finalFrame']) 
                    line_count += 1
            scenes[scene_fileid] = dict()
            print('UTM origins are',xUtmOrigin,yUtmOrigin)
            with open('D:\\datasets\\inD-tools\\drone-dataset-tools-master\\drone-dataset-tools-master\\data\\'+scene_fileid+'_tracks.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        #print(f'Column names are {", ".join(row)}')
                        f=1
                    if row['trackId'] in track_meta_map:
                        if Point(float(row['xCenter']) + xUtmOrigin, float(row['yCenter']) + yUtmOrigin).within(lt_region):
                            if row['trackId'] not in scenes[scene_fileid]:
                                scenes[scene_fileid][row['trackId']] = track_meta_map[row['trackId']] 
                                 
                    line_count += 1
        tot = 0
        for k,v in scenes.items():
            tot += len(v)
            for k1,v1 in v.items():
                print(k,k1,v1)
        print('total',tot)
        
if __name__ == '__main__':
    sc = inD_Scenarios()
    sc.right_turn_scenarios()