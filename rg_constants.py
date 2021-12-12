'''
Created on May 12, 2021

@author: Atrisha
'''
import constants
import os
import sys

DATASET = None
SCENE_TYPE = ('REAL',None)
SCENE_OUT_PATH = 'D:\\repeated_games_data\\intersection_dataset\\scenario_files.csv' if SCENE_TYPE[0] == 'REAL' else 'D:\\repeated_games_data\\intersection_dataset\\'+SCENE_TYPE[0]+'\\'+SCENE_TYPE[1]+'\\'+'scenario_files.csv'
FAILED_FILES_PATH = 'D:\\repeated_games_data\\intersection_dataset\\failed_scenario_files.csv' if SCENE_TYPE[0] == 'REAL' else 'D:\\repeated_games_data\\intersection_dataset\\'+SCENE_TYPE[0]+'\\'+SCENE_TYPE[1]+'\\'+'failed_scenario_files.csv'
TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\game_trees' if SCENE_TYPE[0] == 'REAL' else 'D:\\repeated_games_data\\intersection_dataset\\'+SCENE_TYPE[0]+'\\'+SCENE_TYPE[1]+'\\'+'game_trees'
RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\results' if SCENE_TYPE[0] == 'REAL' else 'D:\\repeated_games_data\\intersection_dataset\\'+SCENE_TYPE[0]+'\\'+SCENE_TYPE[1]+'\\'+'results'
UNI_WEBER_DB_HOME = "D:\\intersections_dataset\\dataset\\"
CURRENT_RG_FILE_ID = None
MAP_FILE = None

PROCEED_VEL_RANGES = {'prep-left-turn':(0.5,12),
                      'exec-left-turn':(0.5,12),
                      'prep-right-turn':(0.5,8),
                      'exec-right-turn':(0.5,8),
                      'exit-lane':(8,18),
                      'left-turn-lane':(5,18),
                      'through-lane-entry':(5,18),
                      'through-lane':(5,18),
                      'right-turn-lane':(5,18)}
def get_db_path(file_id = None):
    uni_weber_db_fileid = constants.CURRENT_FILE_ID if file_id is None else file_id
    uni_weber_dbpath = os.path.join(UNI_WEBER_DB_HOME,uni_weber_db_fileid,'uni_weber_'+uni_weber_db_fileid+'.db')
    return uni_weber_dbpath

def get_rg_db_path(file_id):
    if DATASET == 'intersection_dataset':
        if SCENE_TYPE[0] == 'REAL':
            return 'D:\\repeated_games_data\\intersection_dataset\\db_files\\'+file_id+'.db'
        else:
            return 'D:\\repeated_games_data\\intersection_dataset\\'+SCENE_TYPE[0]+'\\'+SCENE_TYPE[1]+'\\'+'db_files\\'+file_id+'.db'
    elif DATASET == 'inD':
        if SCENE_TYPE[0] == 'REAL':
            return 'D:\\repeated_games_data\\rg_ind_run\\db_files\\'+file_id+'.db'
        else:
            return 'D:\\repeated_games_data\\rg_ind_run\\'+SCENE_TYPE[0]+'\\'+SCENE_TYPE[1]+'\\'+'db_files\\'+file_id+'.db'
    else:
        sys.exit()

    
ind_dataset_path = 'D:\\repeated_games_data\\ind_dataset'
ind_run_path = 'D:\\repeated_games_data\\rg_ind_run\\db_files'