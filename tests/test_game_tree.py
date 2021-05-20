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
from equilibrium.equilibria_calculation import SatisficingEquilibria, AutoStrategyResponse, RobustResponse
from equilibrium.range_estimation import MinDistanceGapModel
from all_utils.utils import setup_pedestrian_info
from equilibrium.game_tree import AssignDistRanges
from code_utils.code_util_objects import RunContext



class TestManvSatisfPESolving(unittest.TestCase):
    @unittest.skip
    def test_perfect_satisficing_equil(self):
        
        scene_def = ScenarioDef(8,28,'769',initialize_db=False)
        maneuver_constraints = scene_def.setup_trajectory_constraints()
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        gt = GameTree(file_id)
        gt.build_tree()
        type(gt.root).progress_ctr = 0
        type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
        start_time = time.time()
        gt.solve(SatisficingEquilibria())
        print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
        #pedestrian_info = setup_pedestrian_info(maneuver_constraints['agent_1']['agent_state'].file_time)
        #rule_obj = Rules(constants.CURRENT_FILE_ID)
        #rule_obj.resolve_rule(ag_obj, pedestrian_info)
        m = MinDistanceGapModel(file_id)
        m.build_model()   
        drassign_obj = AssignDistRanges()
        drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
        f=1
        #print('predicting')
        #predicted_range = m.distgap_model_agent_1.predict([3])
        

class TestRobustEqSolving(unittest.TestCase):
    
    def test_perfect_satisficing_equil(self):
        context = RunContext()
        scene_def = ScenarioDef(8,28,'769',initialize_db=False,start_ts=7.374033)
        maneuver_constraints = scene_def.setup_trajectory_constraints()
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
        gt = GameTree(file_id)
        gt.build_tree()
        type(gt.root).progress_ctr = 0
        type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
        m = MinDistanceGapModel(file_id)
        m.build_model()   
        context = RunContext()
        manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'track_speed'}}
        context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True})
        
        drassign_obj = AssignDistRanges()
        drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
        start_time = time.time()
        gt.solve(SatisficingEquilibria(context))
        print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
        #pedestrian_info = setup_pedestrian_info(maneuver_constraints['agent_1']['agent_state'].file_time)
        #rule_obj = Rules(constants.CURRENT_FILE_ID)
        #rule_obj.resolve_rule(ag_obj, pedestrian_info)
        
        start_time = time.time()
        gt.solve(RobustResponse(context))
        print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
        f=1
        #print('predicting')
        #predicted_range = m.distgap_model_agent_1.predict([3])
                
if __name__ == '__main__':
    unittest.main()