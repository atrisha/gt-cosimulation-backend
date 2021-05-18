'''
Created on May 13, 2021

@author: Atrisha
'''


import unittest
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from equilibrium.game_tree import GameTree
from maps.States import ScenarioDef
import constants
import time
from equilibrium.equilibria_calculation import SatisficingEquilibria
from equilibrium.range_estimation import MinDistanceGapModel


class TestManvSatisfPESolving(unittest.TestCase):
    
    def test_range_estimation(self):
        scene_def = ScenarioDef(8,23,'769',initialize_db=False)
        maneuver_constraints = scene_def.setup_trajectory_constraints()
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        m = MinDistanceGapModel(file_id)
        m.build_model()   
        print('predicting')
        predicted_range = m.distgap_model_agent_1.predict([3,4])
        print('predicted range after going 3mts in 2 secs',predicted_range)
        
        
if __name__ == '__main__':
    unittest.main()