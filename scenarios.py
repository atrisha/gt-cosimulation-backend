'''
Created on Feb 8, 2021

@author: Atrisha
'''
import sqlite3
import numpy as np
import constants
import ast 

def right_turn_interaction_scenarios():
    tot = 0
    for file_id in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "SELECT * from TRAJECTORIES_0"+str(file_id)+"_EXT"
        c.execute(q_string)
        res = c.fetchall()
        scenario_dict = dict()
        ct,N = 0,len(res)
        for row in res:
            ct += 1
            #print(file_id,ct,'/',N)
            if round(row[1]) not in scenario_dict:
                scenario_dict[round(row[1])] = [(row[0],row[2])]
            else:
                scenario_dict[round(row[1])].append((row[0],row[2]))
        for k in list(scenario_dict.keys()):
            if 'l_n_s_r' in [x[1] for x in scenario_dict[k]] and ('rt_prep-turn_w' in [x[1] for x in scenario_dict[k]] or 'rt_exec-turn_w' in [x[1] for x in scenario_dict[k]]):
                continue
            else:
                del scenario_dict[k]
        print(file_id)
        #print(list([x*30 for x in scenario_dict.keys()]))
        vehicles = []
        for k,v in scenario_dict.items():
            for v_id in [x[0] for x in v]:
                if v_id not in vehicles:
                    vehicles.append(v_id)
        print(vehicles,len(vehicles))
        tot += len(vehicles)
        conn.close()
    print('Total', tot)
    
    
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


right_turn_scenarios()    