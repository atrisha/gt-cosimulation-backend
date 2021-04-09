'''
Created on Feb 8, 2021

@author: Atrisha
'''
import sqlite3
import numpy as np
import constants
import ast 

def right_turn_scenarios():
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
        print(list([x*30 for x in scenario_dict.keys()]))
        conn.close()
        
right_turn_scenarios()    