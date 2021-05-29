'''
Created on May 12, 2021

@author: Atrisha
'''

SCENE_OUT_PATH = 'D:\\repeated_games_data\\intersection_dataset\\scenario_files.csv'
FAILED_FILES_PATH = 'D:\\repeated_games_data\\intersection_dataset\\failed_scenario_files.csv'
TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\game_trees'
RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\results'

PROCEED_VEL_RANGES = {'prep-left-turn':(0.5,12),
                      'exec-left-turn':(0.5,12),
                      'prep-right-turn':(0.5,8),
                      'exec-right-turn':(0.5,8),
                      'exit-lane':(8,17),
                      'left-turn-lane':(5,17),
                      'through-lane-entry':(5,17),
                      'through-lane':(5,17),
                      'right-turn-lane':(5,17)}