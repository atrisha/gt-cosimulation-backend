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
            if (prev_signal[1]==0 or curr_time-prev_signal[1] > 2):
                if (next_signal is None) or (next_signal is not None and next_signal[1]-curr_time >2):
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
        agent_entries = dict()
        
        #print(scenario_dict.keys())
        for k in list(scenario_dict.keys()):
            if k==110.7106:
                _e = scenario_dict[k]
                f=1
            #pedestrian_info = utils.setup_pedestrian_info(k)
            el_ctr = Counter([x[1] for x in scenario_dict[k]])
            #print(k)
            for rtag in scenario_dict[k]:
                if ((rtag[1] == 'prep-turn_s' and el_ctr['exec-turn_s']==0) or \
                    (rtag[1] == 'exec-turn_s' and el_ctr['ln_w_-2']==0 and el_ctr['ln_w_-1']==0) or \
                    (rtag[1] == 'ln_s_1' and el_ctr['prep-turn_s']==0 and el_ctr['exec-turn_s']==0) or\
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
                                        if rtag[0] not in agent_entries:
                                            agent_entries[rtag[0]] = [(stag[0],k)]
                                        else:
                                            if stag[0] not in [x[0] for x in agent_entries[rtag[0]]]:
                                                agent_entries[rtag[0]].append((stag[0],k))
                                        #print((rtag[0],stag[0],k,'lt_st_s_e'))  
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
                                        if rtag[0] not in agent_entries:
                                            agent_entries[rtag[0]] = [(stag[0],k)]
                                        else:
                                            if stag[0] not in [x[0] for x in agent_entries[rtag[0]]]:
                                                agent_entries[rtag[0]].append((stag[0],k))
                                        #print((rtag[0],stag[0],k,'lt_st_w_n'))
        #for entr in agent_entries:
        #    print(entr)
        conn.close()
        print('Total',file_id, len(agent_entries))
        tot += len(agent_entries)
    print('grand total',tot)
    
    
def left_turn_scenarios():
    tot = 0
    for file_id in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_s_1%ln_w_-1%'\
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_s_1%ln_w_-2%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_e_1%ln_s_-1%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_e_1%ln_s_-2%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_w_1%ln_n_-1%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_w_1%ln_n_-2%' \
                    ;"
        c.execute(q_string)
        res = c.fetchall()
        print(file_id,len(res))
        tot += len(res)
    print('Total', tot)
    
def right_turn_scenarios():
    tot = 0
    for file_id in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_s_4%ln_e_-1%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_s_4%ln_e_-2%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_w_4%ln_s_-1%' \
                    UNION \
                    select * from TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_w_4%ln_s_-2%'"
        c.execute(q_string)
        res = c.fetchall()
        print(file_id,len(res))
        tot += len(res)
    print('Total', tot)




if __name__ == '__main__':
    right_turn_interaction_scenarios()    