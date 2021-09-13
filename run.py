'''
Created on Jul 7, 2021

@author: Atrisha
'''
import numpy as np
import sqlite3
import itertools
from planners.planning_objects import VehicleState, PedestrianState
import time
from equilibrium.equilibria_calculation import *
import copy
from maps.map_info import NYCMapInfo
from mpl_toolkits.mplot3d import Axes3D
from planners.trajectory_planner import VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner, WaitTrajectoryConstraints, ProceedTrajectoryConstraints
from equilibrium.utilities import Utilities
from numpy import linalg as LA
import math
from maps.States import ScenarioDef, TwoAgentSyntheticScenarioDef
import constants
import csv
from all_utils.utils import pickle_dump_to_dir
import rg_constants
from code_utils.utils import get_all_level_nodes, get_nearest_node,\
    lighten_color
import os
import traceback
import sys
from equilibrium.range_estimation import MinDistanceGapModel
from equilibrium.gametree_objects import TrajectoryCache, TrajectoryFragment, UnsupportedAgentObservationException,UnsupportedLatticeException
from code_utils.code_util_objects import RunContext
import all_utils
import copy
from os import listdir
import matplotlib.pyplot as plt
import ast
from rg_visualizer import UniWeberAnalytics
from shapely.geometry import multipoint, point, linestring, multilinestring, GeometryCollection
from shapely.ops import nearest_points
from os.path import isfile, join
from equilibrium import game_tree
from maps.map_info import IntersectionClearanceMapInfo, MergeBeforeIntersection, ParkingPullout
from equilibrium.game_tree import *
import rg_constants
log = constants.common_logger
from equilibrium.automata_strategies import *



def run_intersection_clearance(run_id,agent1_id, agent2_id,agent1_vel, agent2_vel):
    ''' run the lt veh 1 trajectories '''
    ''' run the st veh 1 trajectories '''
    ''' run the st veh 2 trajectories '''
    ''' save lt-1 , st-1 dbs'''
    ''' save lt-1 , st-2 dbs'''
    ''' solve all dbs. set continuation utils correctly. '''
    ''' plot results for time to clearance for lt-1 '''
    '''
    plt.plot([x[0] for x in IntersectionClearanceMapInfo.st1_waypoints],[x[1] for x in IntersectionClearanceMapInfo.st1_waypoints],color='blue',marker='s')
    plt.plot([x[0] for x in IntersectionClearanceMapInfo.st2_waypoints],[x[1] for x in IntersectionClearanceMapInfo.st2_waypoints],color='green',marker='o')
    plt.plot([x[0] for x in IntersectionClearanceMapInfo.lt_waypoints],[x[1] for x in IntersectionClearanceMapInfo.lt_waypoints],color='red',marker='x')
    plt.show()
    '''
    
    initialize_db=True
    freq=0.5
    file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(agent1_vel).replace('.',',')+'_'+str(agent2_vel).replace('.',',')
    rg_constants.CURRENT_RG_FILE_ID = file_id
    rg_constants.SCENE_TYPE = ('synthetic','intersection_clearance')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    scene_def = TwoAgentSyntheticScenarioDef(initialize_db=initialize_db,file_id=file_id)
    scene_def.add_agent(agent_tag='agent_1',agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, agent_waypoints=IntersectionClearanceMapInfo.lt_waypoints,agent_waypoint_segments=IntersectionClearanceMapInfo.lt_waypoint_segments, direction='L_S_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    if agent2_id == 2:
        scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=IntersectionClearanceMapInfo.st1_waypoints,agent_waypoint_segments=IntersectionClearanceMapInfo.st1_waypoint_segments, direction='L_N_S', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    else:
        scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=IntersectionClearanceMapInfo.st2_waypoints,agent_waypoint_segments=IntersectionClearanceMapInfo.st2_waypoint_segments, direction='L_N_S', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    maneuver_constraints = scene_def.setup_trajectory_constraints()
    tree_builder = TreeBuilder(freq,initialize_db)
    tree_builder.build_complete_tree(maneuver_constraints)
    if rg_constants.SCENE_TYPE[0] == 'REAL':
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
    else:
        file_id = rg_constants.CURRENT_RG_FILE_ID
    gt = GameTree(file_id,freq)
    gt.build_tree(maneuver_constraints)
    type(gt.root).progress_ctr = 0
    type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
    m = MinDistanceGapModel(file_id,freq)
    m.build_model()   
    context = RunContext()
    context.gt_obj = gt
    manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'track_speed'}}
    context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True,'maneuver_constraints':maneuver_constraints})
    drassign_obj = AssignDistRanges()
    drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
    start_time = time.time()
    eq_obj = SatisficingEquilibria(context)
    gt.solve(eq_obj)
    print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(RobustResponse(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(Ql1Model(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    
    gt.scene_def = scene_def
    #assign_emp_nodes(gt,gt.scene_def)
    #gt.print_tree()
    #plot_velocity_profiles(gt,scene_def,freq)
    #gt.animate('mspe')
    gt.maneuver_constraints = None
    
    pickle_dump_to_dir(os.path.join(rg_constants.TREE_FILES,file_id+'.gt'), gt)
    f=1

def run_merge_before_intersection(run_id,agent1_id, agent2_id,agent1_vel, agent2_vel):
    initialize_db=True
    freq=0.5
    file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(agent1_vel).replace('.',',')+'_'+str(agent2_vel).replace('.',',')
    rg_constants.CURRENT_RG_FILE_ID = file_id
    rg_constants.SCENE_TYPE = ('synthetic','merge_before_intersection')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    scene_def = TwoAgentSyntheticScenarioDef(initialize_db=initialize_db,file_id=file_id)
    scene_def.add_agent(agent_tag='agent_1',agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, agent_waypoints=MergeBeforeIntersection.ol_waypoints,agent_waypoint_segments=MergeBeforeIntersection.ol_waypoint_segments, direction='L_S_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=MergeBeforeIntersection.mv_waypoints,agent_waypoint_segments=MergeBeforeIntersection.mv_waypoint_segments, direction='L_S_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    maneuver_map = {'agent_1':{'maneuvers':{'wait':None,'turn':None}, 'agent_state':scene_def.agent1},'agent_2':{'maneuvers':{'wait':None,'turn':None}, 'agent_state':scene_def.agent2}}
    maneuver_constraints = scene_def.setup_trajectory_constraints(maneuver_map=maneuver_map)
    for k,v in maneuver_constraints['agent_2']['maneuvers'].items():
        setattr(v, 'path_degree', 5)
    
    tree_builder = TreeBuilder(freq,initialize_db)
    maneuver_constraints['agent_1']['step_dist'],maneuver_constraints['agent_2']['step_dist'] = 2,2
    tree_builder.build_complete_tree(maneuver_constraints)
    if rg_constants.SCENE_TYPE[0] == 'REAL':
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
    else:
        file_id = rg_constants.CURRENT_RG_FILE_ID
    gt = GameTree(file_id,freq)
    gt.build_tree(maneuver_constraints)
    type(gt.root).progress_ctr = 0
    type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
    m = MinDistanceGapModel(file_id,freq)
    m.build_model()   
    context = RunContext()
    context.gt_obj = gt
    manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'turn'}}
    context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True,'maneuver_constraints':maneuver_constraints})
    drassign_obj = AssignDistRanges()
    drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
    start_time = time.time()
    eq_obj = SatisficingEquilibria(context)
    gt.solve(eq_obj)
    print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(RobustResponse(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(Ql1Model(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    
    gt.scene_def = scene_def
    #assign_emp_nodes(gt,gt.scene_def)
    #gt.print_tree()
    #plot_velocity_profiles(gt,scene_def,freq)
    #gt.animate('mspe')
    gt.maneuver_constraints = None
    
    pickle_dump_to_dir(os.path.join(rg_constants.TREE_FILES,file_id+'.gt'), gt)
    f=1


def run_parking_pullout(run_id,agent1_id, agent2_id,agent1_vel, agent2_vel):
    initialize_db=True
    freq=0.5
    file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(agent1_vel).replace('.',',')+'_'+str(agent2_vel).replace('.',',')
    rg_constants.CURRENT_RG_FILE_ID = file_id
    rg_constants.SCENE_TYPE = ('synthetic','parking_pullout')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    scene_def = TwoAgentSyntheticScenarioDef(initialize_db=initialize_db,file_id=file_id)
    scene_def.add_agent(agent_tag='agent_1',agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, agent_waypoints=ParkingPullout.pv_waypoints,agent_waypoint_segments=ParkingPullout.pv_waypoint_segments, direction='L_S_N', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=ParkingPullout.st_waypoints,agent_waypoint_segments=ParkingPullout.st_waypoint_segments, direction='L_S_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq)
    maneuver_constraints = scene_def.setup_trajectory_constraints()
    for k,v in maneuver_constraints['agent_1']['maneuvers'].items():
        setattr(v, 'path_degree', 3)
    tree_builder = TreeBuilder(freq,initialize_db)
    tree_builder.build_complete_tree(maneuver_constraints)
    if rg_constants.SCENE_TYPE[0] == 'REAL':
        file_id = constants.CURRENT_FILE_ID+'_'+str(maneuver_constraints['agent_1']['agent_state'].id)+'_'+str(maneuver_constraints['agent_2']['agent_state'].id)+'_'+str(maneuver_constraints['agent_1']['agent_state'].file_time).replace('.', ',')
    else:
        file_id = rg_constants.CURRENT_RG_FILE_ID
    gt = GameTree(file_id,freq)
    gt.build_tree(maneuver_constraints)
    type(gt.root).progress_ctr = 0
    type(gt.root).tree_size = gt.root.size(gt.last_decision_level)
    m = MinDistanceGapModel(file_id,freq)
    m.build_model()   
    context = RunContext()
    context.gt_obj = gt
    manv_map = {'agent_1':{'wait':'wait','proceed':'turn'}, 'agent_2':{'wait':'wait','proceed':'turn'}}
    context.set_attrib({'manv_map':manv_map,'acc_dynamic':True,'non_acc_dynamic':True,'maneuver_constraints':maneuver_constraints})
    drassign_obj = AssignDistRanges()
    drassign_obj.assign_distranges(node=gt.root, last_decision_level=gt.last_decision_level, model=m)
    start_time = time.time()
    eq_obj = SatisficingEquilibria(context)
    gt.solve(eq_obj)
    print('solving tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    gt.solve(RobustResponse(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    start_time = time.time()
    context.precision_parm = 1
    gt.solve(Ql1Model(context))
    print('solving autom. strategy tree....DONE','(%s secs)' % (time.time() - start_time),)
    
    gt.scene_def = scene_def
    #assign_emp_nodes(gt,gt.scene_def)
    #gt.print_tree()
    #plot_velocity_profiles(gt,scene_def,freq)
    im_type = 'parking_pullout'
    #gt.animate('mspe',im_type)
    gt.maneuver_constraints = None
    
    pickle_dump_to_dir(os.path.join(rg_constants.TREE_FILES,file_id+'.gt'), gt)
    f=1


def process_results_intersection_clearance():
    rg_constants.SCENE_TYPE = ('synthetic','intersection_clearance')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    rg_constants.RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'results'
    treefiles = [f for f in listdir(rg_constants.TREE_FILES) if isfile(join(rg_constants.TREE_FILES, f))]
    model_types = ['auto_resp','ql1_resp','robust_resp']
    results_map = dict()
    u = Utilities()
    for ctidx,resfile_name in enumerate(treefiles):
        #if ctidx > 10:
        #    break
        print('processing',ctidx+1,resfile_name)
        try:
            gt = all_utils.utils.pickle_load(os.path.join(rg_constants.TREE_FILES,resfile_name))
        except:
            continue
        l6_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=int(3/gt.freq))
        for l6n in l6_nodes:
            ag_1l = l6n.path_from_root['agent_1'].get_last().length
            ag_2l = l6n.path_from_root['agent_2'].get_last().length
            for i in np.arange(5):
                for j in np.arange(5):
                    for m_type in model_types:
                        if m_type == 'auto_resp':
                            resp_range_ag1 = (l6n.parent.auto_strategy_response['agent_1'][0][i,j]['traj_l'], l6n.parent.auto_strategy_response['agent_1'][1][i,j]['traj_l'])
                            resp_range_ag2 = (l6n.parent.auto_strategy_response['agent_2'][0][i,j]['traj_l'], l6n.parent.auto_strategy_response['agent_2'][1][i,j]['traj_l'])
                        elif m_type == 'ql1_resp':
                            resp_range_ag1 = (l6n.parent.ql1_response['response']['agent_1'][i,j]['traj_l'], l6n.parent.ql1_response['response']['agent_1'][i,j]['traj_l'])
                            resp_range_ag2 = (l6n.parent.ql1_response['response']['agent_2'][i,j]['traj_l'], l6n.parent.ql1_response['response']['agent_2'][i,j]['traj_l'])
                        elif m_type == 'robust_resp':
                            resp_range_ag1 = (l6n.parent.robust_response['agent_1'][i,0].veh_eq_acts[0], l6n.parent.robust_response['agent_1'][i,0].veh_eq_acts[1])
                            resp_range_ag2 = (l6n.parent.robust_response['agent_2'][i,0].peds_eq_acts[0], l6n.parent.robust_response['agent_2'][i,0].peds_eq_acts[1])
                        if min(resp_range_ag1) <= ag_1l <= max(resp_range_ag1) and min(resp_range_ag2) <= ag_2l <= max(resp_range_ag2):
                            ag_1l_p = l6n.parent.path_from_root['agent_1'].get_last().length
                            ag_2l_p = l6n.parent.path_from_root['agent_2'].get_last().length
                            if m_type == 'auto_resp':
                                resp_range_ag1 = (l6n.parent.parent.auto_strategy_response['agent_1'][0][i,j]['traj_l'], l6n.parent.parent.auto_strategy_response['agent_1'][1][i,j]['traj_l'])
                                resp_range_ag2 = (l6n.parent.parent.auto_strategy_response['agent_2'][0][i,j]['traj_l'], l6n.parent.parent.auto_strategy_response['agent_2'][1][i,j]['traj_l'])
                            elif m_type == 'ql1_resp':
                                resp_range_ag1 = (l6n.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'], l6n.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'])
                                resp_range_ag2 = (l6n.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'], l6n.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'])
                            elif m_type == 'robust_resp':
                                resp_range_ag1 = (l6n.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[0], l6n.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[1])
                                resp_range_ag2 = (l6n.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[0], l6n.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[1])
                        
                            if min(resp_range_ag1) <= ag_1l_p <= max(resp_range_ag1) and min(resp_range_ag2) <= ag_2l_p <= max(resp_range_ag2):
                                ag_1l_pp = l6n.parent.parent.path_from_root['agent_1'].get_last().length
                                ag_2l_pp = l6n.parent.parent.path_from_root['agent_2'].get_last().length
                                if m_type == 'auto_resp':
                                    resp_range_ag1 = (l6n.parent.parent.parent.auto_strategy_response['agent_1'][0][i,j]['traj_l'], l6n.parent.parent.parent.auto_strategy_response['agent_1'][1][i,j]['traj_l'])
                                    resp_range_ag2 = (l6n.parent.parent.parent.auto_strategy_response['agent_2'][0][i,j]['traj_l'], l6n.parent.parent.parent.auto_strategy_response['agent_2'][1][i,j]['traj_l'])
                                elif m_type == 'ql1_resp':
                                    resp_range_ag1 = (l6n.parent.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'], l6n.parent.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'])
                                    resp_range_ag2 = (l6n.parent.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'], l6n.parent.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'])
                                elif m_type == 'robust_resp':
                                    resp_range_ag1 = (l6n.parent.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[0], l6n.parent.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[1])
                                    resp_range_ag2 = (l6n.parent.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[0], l6n.parent.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[1])
                        
                                if min(resp_range_ag1) <= ag_1l_pp <= max(resp_range_ag1) and min(resp_range_ag2) <= ag_2l_pp <= max(resp_range_ag2):
                                    if resfile_name not in results_map:
                                        results_map[resfile_name] = {k:dict() for k in model_types}
                                    results_map[resfile_name][m_type][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
                    if resfile_name not in results_map:
                        results_map[resfile_name] = {k:dict() for k in model_types}
                    if hasattr(l6n, 'on_mspe') and l6n.on_mspe[i,j]:
                        if 'mspe' not in results_map[resfile_name]:
                            results_map[resfile_name]['mspe'] = dict() 
                        results_map[resfile_name]['mspe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
                    if hasattr(l6n, 'on_uspe') and l6n.on_uspe[i,j]:
                        if 'uspe' not in results_map[resfile_name]:
                            results_map[resfile_name]['uspe'] = dict() 
                        results_map[resfile_name]['uspe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
                        
                        
    print('****** RESULTS CSV START ********')
    with open(os.path.join(rg_constants.RESULTS_FILES,'all_results'+'.csv'), mode='w') as resfile:
        res_writer = csv.writer(resfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                              
        succ_ct = None
        succ_map_speed,succ_map_type,strat_types = OrderedDict(), OrderedDict(), OrderedDict()
        for k12 in results_map.keys():
            for k13 in results_map.keys():
                
                if k12 == k13 or k12.split('_')[1].split('-')[1] != '2' or k13.split('_')[1].split('-')[1] != '3':
                    continue
                lt_vel_1, lt_vel_2 = k12.split('_')[2], k13.split('_')[2]
                if lt_vel_1 == lt_vel_2:
                    for m_type in results_map[k12].keys():
                        if succ_ct is None:
                            succ_ct = {_k:0 for _k in results_map[k12].keys()}
                        for type_comb1 in results_map[k12][m_type].keys():
                            for type_comb2 in results_map[k13][m_type].keys():
                                if type_comb1[0] == type_comb2[0]:
                                    ''' agent 1's velocity and type matches, so we can check this scenario now '''
                                    ag1_l, ag2_l, ag1_traj_frag, ag2_traj_frag = results_map[k12][m_type][type_comb1]
                                    ag1_l2, ag3_l, ag1_traj_frag, ag3_traj_frag = results_map[k13][m_type][type_comb2]
                                    ag1_traj = ag1_traj_frag.loaded_traj
                                    ag2_traj = ag2_traj_frag.loaded_traj
                                    ag3_traj = ag3_traj_frag.loaded_traj
                                    dist_gap12,dist_gap13 = None, None
                                    if max([ag1_l,ag1_l2]) >= 20 and (ag2_l <= IntersectionClearanceMapInfo.st1_on_intersection_distance[0] or ag2_l >= IntersectionClearanceMapInfo.st1_on_intersection_distance[1]) \
                                        and (ag3_l <= 50 or ag3_l >= IntersectionClearanceMapInfo.st2_on_intersection_distance[1]):
                                        dist_gap12 = u.calc_dist_gap(veh_traj = ag1_traj, ped_traj = ag2_traj)
                                        dist_gap13 = u.calc_dist_gap(veh_traj = ag1_traj, ped_traj = ag3_traj)
                                        #print('dist_gap',m_type,type_comb1[0],type_comb1[1],type_comb2[1],min(dist_gap12,dist_gap13))
                                        if min(dist_gap12,dist_gap13) >= 1:
                                            succ_ct[m_type] += 1
                                            success = True
                                        else:
                                            success = False
                                    else:
                                        success = False
                                    if success:
                                        lt_speed = lt_vel_1.replace(',','.')
                                        st1_speed = k12.split('_')[3].split('.')[0].replace(',','.')
                                        st2_speed = k13.split('_')[3].split('.')[0].replace(',','.')
                                        lt_type, st1_type, st2_type = type_comb1[0],type_comb1[1],type_comb2[1]
                                        print(m_type,lt_speed,st1_speed,st2_speed,lt_type, st1_type, st2_type,max([ag1_l,ag1_l2]),ag2_l,ag3_l,'success' if success else 'fail')
                                        if (lt_speed,st1_speed,st2_speed) not in succ_map_speed:
                                            succ_map_speed[(lt_speed,st1_speed,st2_speed)] = dict()
                                        if m_type not in succ_map_speed[(lt_speed,st1_speed,st2_speed)]:
                                            succ_map_speed[(lt_speed,st1_speed,st2_speed)][m_type] = 0
                                        succ_map_speed[(lt_speed,st1_speed,st2_speed)][m_type] += 1
                                        
                                        if (lt_type, st1_type, st2_type) not in succ_map_type:
                                            succ_map_type[(lt_type, st1_type, st2_type)] = dict()
                                        if m_type not in succ_map_type[(lt_type, st1_type, st2_type)]:
                                            succ_map_type[(lt_type, st1_type, st2_type)][m_type] = 0
                                        succ_map_type[(lt_type, st1_type, st2_type)][m_type] += 1
                                        
                                        if m_type not in strat_types:
                                            strat_types[m_type] = dict()
                                        ag1_manvs = tuple(ag1_traj_frag.manv_from_root)
                                        ag2_manvs = tuple(ag2_traj_frag.manv_from_root)
                                        ag3_manvs = tuple(ag3_traj_frag.manv_from_root)
                                        ag1_trajl = round(ag1_traj_frag.total_length * 2) / 2
                                        ag2_trajl = round(ag2_traj_frag.total_length * 2) / 2
                                        ag3_trajl = round(ag3_traj_frag.total_length * 2) / 2
                                        
                                        if tuple([ag1_trajl,ag2_trajl,ag3_trajl]) not in strat_types[m_type]:
                                            strat_types[m_type][tuple([ag1_trajl,ag2_trajl,ag3_trajl])] = 1
                                        else:
                                            strat_types[m_type][tuple([ag1_trajl,ag2_trajl,ag3_trajl])] += 1
                                        print(m_type,lt_type, st1_type, st2_type,lt_speed,st1_speed,st2_speed,ag1_trajl,ag2_trajl,ag3_trajl,dist_gap12,dist_gap13, sep=",")
                                        res_writer.writerow([m_type,lt_type, st1_type, st2_type,lt_speed,st1_speed,st2_speed,ag1_trajl,ag2_trajl,ag3_trajl,dist_gap12,dist_gap13])
                                    
                                        
                                        
                                        
    pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'strat_types'+'.result'), strat_types)                              
    pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_type'+'.result'), succ_map_type)  
    pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_speed'+'.result'), succ_map_speed)  
    print('****** RESULTS CSV END ********')
    print('Total success counts:')
    for k,v in succ_ct.items():
        print(k,':',v)                        
    type_arr = list(itertools.product([0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]))
    #type_arr = [tuple([str(y) for y in list(x)]) for x in list(itertools.product([0,0.5,1,1.5,2],[10,12,15,17,20],[10,12,15,17,20]))]
    '''
    for k,v in strat_types.items():
        plt.figure()
        plt.title(k+' - strat type')
        plt.gcf().subplots_adjust(bottom=.5)
        plt.bar(np.arange(len(v)), list(v.values()), align='center', alpha=0.5)
        plt.xticks(np.arange(len(v)), list(v.keys()),fontsize=6)
        plt.xticks(rotation=90)
        
        
        
    all_models = model_types + ['mspe','uspe']
    N = len(type_arr)
    all_data = []
    for midx,m in enumerate(all_models):
        data = []
        for t in type_arr:
            if t in succ_map_type:
                if m in succ_map_type[t]:
                    data.append(succ_map_type[t][m])
                else:
                    data.append(0)
            else:
                data.append(0)
        all_data.append(data)
    fig, axs = plt.subplots(2)
    #fig.suptitle('pooling map-'+k,y=1.12)
    mdl_colors = ['r','g','b','c','black']
    barplots = [None] * len(all_models)
    for midx,m in enumerate(all_models):
        print(midx, all_data[midx])
        if midx == 0:
            barplots[midx] = axs[0].bar(np.arange(N), all_data[midx], color=mdl_colors[midx])
            print(midx, all_data[midx])
        else:
            if midx > 1:
                _arr = np.asarray(all_data[:midx-1])
                _bottom_sum = np.sum(_arr,axis=0)
            else:
                _bottom_sum = all_data[0]
                
            barplots[midx] = axs[0].bar(np.arange(N), all_data[midx], color=mdl_colors[midx], bottom = _bottom_sum)
            print(midx,_bottom_sum)
    axs[0].legend (tuple(barplots), tuple(all_models))
    xax_legend = np.asarray(type_arr).T
    rg_utils.plot_heatmap(plt, fig, axs[1], xax_legend)
    plt.show()
    f=1
    '''
    
class ProcessResults:
    
    def generate_results_map(self):
        treefiles = [f for f in listdir(rg_constants.TREE_FILES) if isfile(join(rg_constants.TREE_FILES, f))]
        if hasattr(self, 'visualization_only'):
            incl_treefiles = []
            for x in treefiles:
                this_tree_file = '_'.join(x.split('.')[0].split('_')[1:])
                if this_tree_file in self.visualization_only:
                    incl_treefiles.append(x)
            treefiles = incl_treefiles
        model_types = ['auto_resp','ql1_resp','robust_resp']
        results_map = dict()
        for ctidx,resfile_name in enumerate(treefiles):
            #if ctidx > 5:
            #    break
            print('processing',ctidx+1,resfile_name)
            try:
                gt = all_utils.utils.pickle_load(os.path.join(rg_constants.TREE_FILES,resfile_name))
            except:
                continue
            l6_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=int(3/gt.freq))
            for l6n in l6_nodes:
                ag_1l = l6n.path_from_root['agent_1'].get_last().length
                ag_2l = l6n.path_from_root['agent_2'].get_last().length
                for i in np.arange(5):
                    for j in np.arange(5):
                        for m_type1,m_type2 in itertools.product(model_types,model_types):
                            if m_type1 == 'auto_resp':
                                resp_range_ag1 = (l6n.parent.auto_strategy_response['agent_1'][0][i,j]['traj_l'], l6n.parent.auto_strategy_response['agent_1'][1][i,j]['traj_l'])
                            elif m_type1 == 'ql1_resp':
                                resp_range_ag1 = (l6n.parent.ql1_response['response']['agent_1'][i,j]['traj_l'], l6n.parent.ql1_response['response']['agent_1'][i,j]['traj_l'])
                            elif m_type1 == 'robust_resp':
                                resp_range_ag1 = (l6n.parent.robust_response['agent_1'][i,0].veh_eq_acts[0], l6n.parent.robust_response['agent_1'][i,0].veh_eq_acts[1])
                            if m_type2 == 'auto_resp':
                                resp_range_ag2 = (l6n.parent.auto_strategy_response['agent_2'][0][i,j]['traj_l'], l6n.parent.auto_strategy_response['agent_2'][1][i,j]['traj_l'])
                            elif m_type2 == 'ql1_resp':
                                resp_range_ag2 = (l6n.parent.ql1_response['response']['agent_2'][i,j]['traj_l'], l6n.parent.ql1_response['response']['agent_2'][i,j]['traj_l'])
                            elif m_type2 == 'robust_resp':
                                resp_range_ag2 = (l6n.parent.robust_response['agent_2'][i,0].peds_eq_acts[0], l6n.parent.robust_response['agent_2'][i,0].peds_eq_acts[1])
                            
                            if min(resp_range_ag1) <= ag_1l <= max(resp_range_ag1) and min(resp_range_ag2) <= ag_2l <= max(resp_range_ag2):
                                ag_1l_p = l6n.parent.path_from_root['agent_1'].get_last().length
                                ag_2l_p = l6n.parent.path_from_root['agent_2'].get_last().length
                                if m_type1 == 'auto_resp':
                                    resp_range_ag1 = (l6n.parent.parent.auto_strategy_response['agent_1'][0][i,j]['traj_l'], l6n.parent.parent.auto_strategy_response['agent_1'][1][i,j]['traj_l'])
                                elif m_type1 == 'ql1_resp':
                                    resp_range_ag1 = (l6n.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'], l6n.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'])
                                elif m_type1 == 'robust_resp':
                                    resp_range_ag1 = (l6n.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[0], l6n.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[1])
                                if m_type2 == 'auto_resp':
                                    resp_range_ag2 = (l6n.parent.parent.auto_strategy_response['agent_2'][0][i,j]['traj_l'], l6n.parent.parent.auto_strategy_response['agent_2'][1][i,j]['traj_l'])
                                elif m_type2 == 'ql1_resp':
                                    resp_range_ag2 = (l6n.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'], l6n.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'])
                                elif m_type2 == 'robust_resp':
                                    resp_range_ag2 = (l6n.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[0], l6n.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[1])
                                if min(resp_range_ag1) <= ag_1l_p <= max(resp_range_ag1) and min(resp_range_ag2) <= ag_2l_p <= max(resp_range_ag2):
                                    ag_1l_pp = l6n.parent.parent.path_from_root['agent_1'].get_last().length
                                    ag_2l_pp = l6n.parent.parent.path_from_root['agent_2'].get_last().length
                                    if m_type1 == 'auto_resp':
                                        resp_range_ag1 = (l6n.parent.parent.parent.auto_strategy_response['agent_1'][0][i,j]['traj_l'], l6n.parent.parent.parent.auto_strategy_response['agent_1'][1][i,j]['traj_l'])
                                    elif m_type1 == 'ql1_resp':
                                        resp_range_ag1 = (l6n.parent.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'], l6n.parent.parent.parent.ql1_response['response']['agent_1'][i,j]['traj_l'])
                                    elif m_type1 == 'robust_resp':
                                        resp_range_ag1 = (l6n.parent.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[0], l6n.parent.parent.parent.robust_response['agent_1'][i,0].veh_eq_acts[1])
                                    if m_type2 == 'auto_resp':
                                        resp_range_ag2 = (l6n.parent.parent.parent.auto_strategy_response['agent_2'][0][i,j]['traj_l'], l6n.parent.parent.parent.auto_strategy_response['agent_2'][1][i,j]['traj_l'])
                                    elif m_type2 == 'ql1_resp':
                                        resp_range_ag2 = (l6n.parent.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'], l6n.parent.parent.parent.ql1_response['response']['agent_2'][i,j]['traj_l'])
                                    elif m_type2 == 'robust_resp':
                                        resp_range_ag2 = (l6n.parent.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[0], l6n.parent.parent.parent.robust_response['agent_2'][i,0].peds_eq_acts[1])
                                    
                                    if min(resp_range_ag1) <= ag_1l_pp <= max(resp_range_ag1) and min(resp_range_ag2) <= ag_2l_pp <= max(resp_range_ag2):
                                        if resfile_name not in results_map:
                                            results_map[resfile_name] = {k:dict() for k in itertools.product(model_types,model_types)}
                                        results_map[resfile_name][(m_type1,m_type2)][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
                        if resfile_name not in results_map:
                            results_map[resfile_name] = {k:dict() for k in itertools.product(model_types,model_types)}
                        if hasattr(l6n, 'on_mspe') and l6n.on_mspe[i,j]:
                            if 'mspe' not in results_map[resfile_name]:
                                results_map[resfile_name]['mspe'] = dict() 
                            results_map[resfile_name]['mspe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
                        if hasattr(l6n, 'on_uspe') and l6n.on_uspe[i,j]:
                            if 'uspe' not in results_map[resfile_name]:
                                results_map[resfile_name]['uspe'] = dict() 
                            results_map[resfile_name]['uspe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
        return results_map     
    
    def process_intersection_clearance(self):
        rg_constants.SCENE_TYPE = ('synthetic','intersection_clearance')
        rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
        rg_constants.RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'results'
        results_map = self.generate_results_map()
        print('****** RESULTS CSV START ********')
        u = Utilities()
        with open(os.path.join(rg_constants.RESULTS_FILES,'all_results_1'+'.csv'), mode='w') as resfile:
            res_writer = csv.writer(resfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            res_writer.writerow(['agent1_model_type','agent2_model_type','lt_type', 'st1_type', 'st2_type','lt_speed','st1_speed','st2_speed','lt_trajl','st1_trajl','st2_trajl','dist_gap12','dist_gap13','success'])                      
            succ_ct = None
            succ_map_speed,succ_map_type,strat_types = OrderedDict(), OrderedDict(), OrderedDict()
            for k12 in results_map.keys():
                for k13 in results_map.keys():
                    
                    if k12 == k13 or k12.split('_')[1].split('-')[1] != '2' or k13.split('_')[1].split('-')[1] != '3':
                        continue
                    lt_vel_1, lt_vel_2 = k12.split('_')[2], k13.split('_')[2]
                    if lt_vel_1 == lt_vel_2:
                        for m_type in results_map[k12].keys():
                            if succ_ct is None:
                                succ_ct = {_k:0 for _k in results_map[k12].keys()}
                            for type_comb1 in results_map[k12][m_type].keys():
                                for type_comb2 in results_map[k13][m_type].keys():
                                    if type_comb1[0] == type_comb2[0]:
                                        ''' agent 1's velocity and type matches, so we can check this scenario now '''
                                        ag1_l, ag2_l, ag1_traj_frag, ag2_traj_frag = results_map[k12][m_type][type_comb1]
                                        ag1_l2, ag3_l, ag1_traj_frag, ag3_traj_frag = results_map[k13][m_type][type_comb2]
                                        ag1_traj = ag1_traj_frag.loaded_traj
                                        ag2_traj = ag2_traj_frag.loaded_traj
                                        ag3_traj = ag3_traj_frag.loaded_traj
                                        dist_gap12,dist_gap13 = None, None
                                        if max([ag1_l,ag1_l2]) >= 20 and (ag2_l <= IntersectionClearanceMapInfo.st1_on_intersection_distance[0] or ag2_l >= IntersectionClearanceMapInfo.st1_on_intersection_distance[1]) \
                                            and (ag3_l <= 50 or ag3_l >= IntersectionClearanceMapInfo.st2_on_intersection_distance[1]):
                                            dist_gap12 = u.calc_dist_gap(veh_traj = ag1_traj, ped_traj = ag2_traj)
                                            dist_gap13 = u.calc_dist_gap(veh_traj = ag1_traj, ped_traj = ag3_traj)
                                            #print('dist_gap',m_type,type_comb1[0],type_comb1[1],type_comb2[1],min(dist_gap12,dist_gap13))
                                            if min(dist_gap12,dist_gap13) >= 1:
                                                succ_ct[m_type] += 1
                                                success = True
                                            else:
                                                success = False
                                        else:
                                            success = False
                                        if success or (dist_gap12 is not None and dist_gap13 is not None and min(dist_gap12,dist_gap13) < 1):
                                            lt_speed = lt_vel_1.replace(',','.')
                                            st1_speed = k12.split('_')[3].split('.')[0].replace(',','.')
                                            st2_speed = k13.split('_')[3].split('.')[0].replace(',','.')
                                            lt_type, st1_type, st2_type = type_comb1[0],type_comb1[1],type_comb2[1]
                                            if (lt_speed,st1_speed,st2_speed) not in succ_map_speed:
                                                succ_map_speed[(lt_speed,st1_speed,st2_speed)] = dict()
                                            if m_type not in succ_map_speed[(lt_speed,st1_speed,st2_speed)]:
                                                succ_map_speed[(lt_speed,st1_speed,st2_speed)][m_type] = 0
                                            succ_map_speed[(lt_speed,st1_speed,st2_speed)][m_type] += 1
                                            
                                            if (lt_type, st1_type, st2_type) not in succ_map_type:
                                                succ_map_type[(lt_type, st1_type, st2_type)] = dict()
                                            if m_type not in succ_map_type[(lt_type, st1_type, st2_type)]:
                                                succ_map_type[(lt_type, st1_type, st2_type)][m_type] = 0
                                            succ_map_type[(lt_type, st1_type, st2_type)][m_type] += 1
                                            
                                            if m_type not in strat_types:
                                                strat_types[m_type] = dict()
                                            ag1_manvs = tuple(ag1_traj_frag.manv_from_root)
                                            ag2_manvs = tuple(ag2_traj_frag.manv_from_root)
                                            ag3_manvs = tuple(ag3_traj_frag.manv_from_root)
                                            ag1_trajl = round(ag1_traj_frag.total_length * 2) / 2
                                            ag2_trajl = round(ag2_traj_frag.total_length * 2) / 2
                                            ag3_trajl = round(ag3_traj_frag.total_length * 2) / 2
                                            
                                            if tuple([ag1_trajl,ag2_trajl,ag3_trajl]) not in strat_types[m_type]:
                                                strat_types[m_type][tuple([ag1_trajl,ag2_trajl,ag3_trajl])] = 1
                                            else:
                                                strat_types[m_type][tuple([ag1_trajl,ag2_trajl,ag3_trajl])] += 1
                                            if isinstance(m_type, tuple): 
                                                print(m_type[0],m_type[1],lt_type, st1_type, st2_type,lt_speed,st1_speed,st2_speed,ag1_trajl,ag2_trajl,ag3_trajl,dist_gap12,dist_gap13,success, sep=",")
                                                res_writer.writerow([m_type[0],m_type[1],lt_type, st1_type, st2_type,lt_speed,st1_speed,st2_speed,ag1_trajl,ag2_trajl,ag3_trajl,dist_gap12,dist_gap13,success])
                                            else:
                                                print(m_type,m_type,lt_type, st1_type, st2_type,lt_speed,st1_speed,st2_speed,ag1_trajl,ag2_trajl,ag3_trajl,dist_gap12,dist_gap13,success, sep=",")
                                                res_writer.writerow([m_type,m_type,lt_type, st1_type, st2_type,lt_speed,st1_speed,st2_speed,ag1_trajl,ag2_trajl,ag3_trajl,dist_gap12,dist_gap13,success])
                                        else:    
                                            if st1_speed > 15 :
                                                '''
                                                plt.figure()
                                                plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],color='blue')
                                                plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],color='red',marker='x')
                                                plt.plot([x[1] for x in ag3_traj],[x[2] for x in ag3_traj],color='pink',marker='o')
                                                plt.show()
                                               '''
                                                if ag1_l > ag1_l2:
                                                    ag1_traj = results_map[k12][m_type][type_comb1][2].loaded_traj
                                                else:
                                                    ag1_traj = results_map[k13][m_type][type_comb1][2].loaded_traj
                                                analytics_obj = UniWeberAnalytics('769')
                                                analytics_obj.animate_scene([[(x[1],x[2]) for x in ag1_traj],[(x[1],x[2]) for x in ag2_traj],[(x[1],x[2]) for x in ag3_traj]],None)
                                                
                                            
                                                
                                               
                                            
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'strat_types'+'.result'), strat_types)                              
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_type'+'.result'), succ_map_type)  
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_speed'+'.result'), succ_map_speed)  
        print('****** RESULTS CSV END ********')
        print('Total success counts:')
        for k,v in succ_ct.items():
            print(k,':',v)                        
        
    
    def process_merge_before_intersection(self):
        rg_constants.SCENE_TYPE = ('synthetic','merge_before_intersection')
        rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
        rg_constants.RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'results'
        results_map = self.generate_results_map()
        model_types = ['auto_resp','ql1_resp','robust_resp']
        u = Utilities()
        print('****** RESULTS CSV START ********')
        with open(os.path.join(rg_constants.RESULTS_FILES,'all_results'+'.csv'), mode='w') as resfile:
            res_writer = csv.writer(resfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            res_writer.writerow(['agent1_model_type','agent2_model_type','agent1_type','agent2_type','agent1_speed','agent2_speed','agent1_traj_length','agent1_traj_length2','agent2_traj_length','agent2_traj_length2','dist_gap'])
            succ_ct = None
            succ_map_speed,succ_map_type,strat_types = OrderedDict(), OrderedDict(), OrderedDict()
            for k12 in results_map.keys():
                for m_type in results_map[k12].keys():
                    if succ_ct is None:
                        succ_ct = {_k:0 for _k in results_map[k12].keys()}
                    for type_comb1 in results_map[k12][m_type].keys():
                        ag1_l, ag2_l, ag1_traj_frag, ag2_traj_frag = results_map[k12][m_type][type_comb1]
                        
                        ag1_traj = ag1_traj_frag.loaded_traj
                        ag2_traj = ag2_traj_frag.loaded_traj
                        success = False
                        '''
                        plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],color='blue')
                        plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],color='red')
                        plt.title(str(type_comb1)+'-'+str(m_type))
                        plt.show()
                        '''
                        stopline = LineString(MergeBeforeIntersection.stopline)
                        ag1_path = LineString([(x[1],x[2]) for x in ag1_traj])
                        crosspt = ag1_path.intersection(stopline)
                        if isinstance(crosspt, multipoint.MultiPoint):
                            crosspt = crosspt[0]
                        if len(crosspt.coords) > 0:
                            '''
                            crosspt = nearest_points(ag1_path, crosspt)[0]
                            dist_2_crosspt = ag1_path.project(crosspt)
                            _p = [(x[1],x[2]) for x in ag1_traj]
                            path_dist_cumsum = [0] + np.cumsum([math.hypot(x1[0]-x2[0], x1[1]-x2[1]) for x1,x2 in zip(_p[:-1],_p[1:])]).tolist()
                            crosspt_idx = next(x[0] for x in enumerate(path_dist_cumsum) if x[1] > dist_2_crosspt)
                            ag1_time_to_cross = ag1_traj[crosspt_idx][0]
                            '''
                            for _idx in np.arange(1,len(ag1_traj)):
                                if rg_utils.has_crossed((ag1_traj[_idx][1],ag1_traj[_idx][2]), (ag1_traj[_idx-1][1],ag1_traj[_idx-1][2]), stopline):
                                    ag1_time_to_cross = _idx
                                    
                        else:
                            ag1_time_to_cross = np.inf
                        ag2_path = LineString([(x[1],x[2]) for x in ag2_traj])
                        crosspt = ag2_path.intersection(stopline)
                        if isinstance(crosspt, multipoint.MultiPoint):
                            crosspt = crosspt[0]
                        if len(crosspt.coords) > 0:
                            '''
                            crosspt = nearest_points(ag2_path, crosspt)[0]
                            dist_2_crosspt = ag2_path.project(crosspt)
                            _p = [(x[1],x[2]) for x in ag2_traj]
                            path_dist_cumsum = [0] + np.cumsum([math.hypot(x1[0]-x2[0], x1[1]-x2[1]) for x1,x2 in zip(_p[:-1],_p[1:])]).tolist()
                            crosspt_idx = next(x[0] for x in enumerate(path_dist_cumsum) if x[1] > dist_2_crosspt)
                            ag2_time_to_cross = ag2_traj[crosspt_idx][0]
                            '''
                            for _idx in np.arange(1,len(ag2_traj)):
                                if rg_utils.has_crossed((ag2_traj[_idx][1],ag2_traj[_idx][2]), (ag2_traj[_idx-1][1],ag2_traj[_idx-1][2]), stopline):
                                    ag2_time_to_cross = _idx
                        else:
                            ag2_time_to_cross = np.inf
                        ag2_speed = k12.split('_')[3].split('.')[0].replace(',','.')
                        ag1_speed = k12.split('_')[2].replace(',','.')
                            
                        if ag2_time_to_cross < ag1_time_to_cross:
                            success = True      
                            succ_ct[m_type] += 1
                        
                        if m_type[0] == 'robust_resp' and m_type[1] == 'robust_resp' and type_comb1 == (4,0) and not success:
                            f=1
                        if success:
                            
                            if (ag1_speed,ag2_speed) not in succ_map_speed:
                                succ_map_speed[(ag1_speed,ag2_speed)] = dict()
                            if m_type not in succ_map_speed[(ag1_speed,ag2_speed)]:
                                succ_map_speed[(ag1_speed,ag2_speed)][m_type] = 0
                            succ_map_speed[(ag1_speed,ag2_speed)][m_type] += 1
                            
                            if type_comb1 not in succ_map_type:
                                succ_map_type[type_comb1] = dict()
                            if m_type not in succ_map_type[type_comb1]:
                                succ_map_type[type_comb1][m_type] = 0
                            succ_map_type[type_comb1][m_type] += 1
                            
                            dist_gap = u.calc_dist_gap(ag1_traj,ag2_traj)
                            dist_gap = round(dist_gap * 2) / 2
                            if isinstance(m_type, tuple):
                                res_writer.writerow([m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag1_traj_frag.next_fragment.length,ag2_traj_frag.length,ag2_traj_frag.next_fragment.length,dist_gap])
                                print(m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap, sep=",")
                            else:
                                res_writer.writerow([m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag1_traj_frag.next_fragment.length,ag2_traj_frag.length,ag2_traj_frag.next_fragment.length,dist_gap])
                                print(m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap, sep=",")
                            
                            if dist_gap not in strat_types:
                                strat_types[dist_gap] = dict()
                            if m_type not in strat_types[dist_gap]:
                                strat_types[dist_gap][m_type] = 0
                            else:
                                strat_types[dist_gap][m_type] += 1
                            '''
                            if dist_gap > 2.5:
                                
                                plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],'o')
                                plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],'x')
                                plt.plot([x[0] for x in MergeBeforeIntersection.stopline],[x[1] for x in MergeBeforeIntersection.stopline])
                                img = plt.imread("D:\\behavior modeling\\background.jpg")
                                ax = plt.gca()
                                ax.imshow(img, extent=[538780, 538890, 4813970, 4814055])
                                print(ag1_time_to_cross,ag2_time_to_cross)
                                plt.show()
                                
                            
                                analytics_obj = UniWeberAnalytics('769')
                                analytics_obj.animate_scene([[(x[1],x[2]) for x in ag1_traj],[(x[1],x[2]) for x in ag2_traj]],None)
                            '''
                        '''
                        else:
                            if isinstance(m_type, tuple):
                                print(m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,success, sep=",")
                            else:
                                print(m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,success, sep=",")
                        ''' 
                            
                            
                        
                            
                            
        #self.show_results(succ_ct, strat_types, model_types, succ_map_type)            
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'strat_types'+'.result'), strat_types)                              
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_type'+'.result'), succ_map_type)  
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_speed'+'.result'), succ_map_speed)  
        print('****** RESULTS CSV END ********')
        print('Total success counts:')
        for k,v in succ_ct.items():
            print(k,':',v)                              
    
    
        
    
    def process_parking_pullout(self):
        rg_constants.SCENE_TYPE = ('synthetic','parking_pullout')
        rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
        rg_constants.RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'results'
        results_map = self.generate_results_map()
        model_types = ['auto_resp','ql1_resp','robust_resp']
        u = Utilities()
        print('****** RESULTS CSV START ********')
        with open(os.path.join(rg_constants.RESULTS_FILES,'all_results'+'.csv'), mode='w') as resfile:
            res_writer = csv.writer(resfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            res_writer.writerow(['agent1_model_type','agent2_model_type','agent1_type','agent2_type','agent1_speed','agent2_speed','agent1_traj_length','agent2_traj_length','dist_gap'])
            succ_ct = None
            succ_map_speed,succ_map_type,strat_types = OrderedDict(), OrderedDict(), OrderedDict()
            for k12 in results_map.keys():
                for m_type in results_map[k12].keys():
                    if succ_ct is None:
                        succ_ct = {_k:0 for _k in results_map[k12].keys()}
                    for type_comb1 in results_map[k12][m_type].keys():
                        ag1_l, ag2_l, ag1_traj_frag, ag2_traj_frag = results_map[k12][m_type][type_comb1]
                        
                        ag1_traj = ag1_traj_frag.loaded_traj
                        ag2_traj = ag2_traj_frag.loaded_traj
                        success = False
                        '''
                        plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],color='blue')
                        plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],color='red')
                        plt.title(str(type_comb1)+'-'+str(m_type))
                        plt.show()
                        '''
                        stopline = LineString(ParkingPullout.exitline)
                        ag1_path = LineString([(x[1],x[2]) for x in ag1_traj])
                        crosspt = ag1_path.intersection(stopline)
                        if isinstance(crosspt, multipoint.MultiPoint):
                            crosspt = crosspt[0]
                        if len(crosspt.coords) > 0:
                            '''
                            crosspt = nearest_points(ag1_path, crosspt)[0]
                            dist_2_crosspt = ag1_path.project(crosspt)
                            _p = [(x[1],x[2]) for x in ag1_traj]
                            path_dist_cumsum = [0] + np.cumsum([math.hypot(x1[0]-x2[0], x1[1]-x2[1]) for x1,x2 in zip(_p[:-1],_p[1:])]).tolist()
                            crosspt_idx = next(x[0] for x in enumerate(path_dist_cumsum) if x[1] > dist_2_crosspt)
                            ag1_time_to_cross = ag1_traj[crosspt_idx][0]
                            '''
                            for _idx in np.arange(1,len(ag1_traj)):
                                if rg_utils.has_crossed((ag1_traj[_idx][1],ag1_traj[_idx][2]), (ag1_traj[_idx-1][1],ag1_traj[_idx-1][2]), stopline):
                                    ag1_time_to_cross = _idx
                        else:
                            ag1_time_to_cross = np.inf
                        ag2_path = LineString([(x[1],x[2]) for x in ag2_traj])
                        crosspt = ag2_path.intersection(stopline)
                        if isinstance(crosspt, multipoint.MultiPoint):
                            crosspt = crosspt[0]
                        if len(crosspt.coords) > 0:
                            '''
                            crosspt = nearest_points(ag2_path, crosspt)[0]
                            dist_2_crosspt = ag2_path.project(crosspt)
                            _p = [(x[1],x[2]) for x in ag2_traj]
                            path_dist_cumsum = [0] + np.cumsum([math.hypot(x1[0]-x2[0], x1[1]-x2[1]) for x1,x2 in zip(_p[:-1],_p[1:])]).tolist()
                            crosspt_idx = next(x[0] for x in enumerate(path_dist_cumsum) if x[1] > dist_2_crosspt)
                            ag2_time_to_cross = ag2_traj[crosspt_idx][0]
                            '''
                            for _idx in np.arange(1,len(ag2_traj)):
                                if rg_utils.has_crossed((ag2_traj[_idx][1],ag2_traj[_idx][2]), (ag2_traj[_idx-1][1],ag2_traj[_idx-1][2]), stopline):
                                    ag2_time_to_cross = _idx
                        else:
                            ag2_time_to_cross = np.inf
                        ag2_speed = k12.split('_')[3].split('.')[0].replace(',','.')
                        ag1_speed = k12.split('_')[2].replace(',','.')
                            
                        if ag2_time_to_cross < ag1_time_to_cross:
                            success = True      
                            succ_ct[m_type] += 1
                        
                        if success:
                            if (ag1_speed,ag2_speed) not in succ_map_speed:
                                succ_map_speed[(ag1_speed,ag2_speed)] = dict()
                            if m_type not in succ_map_speed[(ag1_speed,ag2_speed)]:
                                succ_map_speed[(ag1_speed,ag2_speed)][m_type] = 0
                            succ_map_speed[(ag1_speed,ag2_speed)][m_type] += 1
                            
                            if type_comb1 not in succ_map_type:
                                succ_map_type[type_comb1] = dict()
                            if m_type not in succ_map_type[type_comb1]:
                                succ_map_type[type_comb1][m_type] = 0
                            succ_map_type[type_comb1][m_type] += 1
                            
                            dist_gap = u.calc_dist_gap(ag1_traj,ag2_traj)
                            #dist_gap = round(dist_gap * 2) / 2
                            
                            if isinstance(m_type, tuple):
                                res_writer.writerow([m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag2_traj_frag.length,dist_gap])
                                print(m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,success, sep=",")
                            else:
                                res_writer.writerow([m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag2_traj_frag.length,dist_gap])
                                print(m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,success, sep=",")
                            
                            if dist_gap not in strat_types:
                                strat_types[dist_gap] = dict()
                            if m_type not in strat_types[dist_gap]:
                                strat_types[dist_gap][m_type] = 0
                            else:
                                strat_types[dist_gap][m_type] += 1
                            '''
                            if float(ag1_speed) > 0.8:
                                analytics_obj = UniWeberAnalytics('769')
                                analytics_obj.animate_scene([[(x[1],x[2]) for x in ag1_traj],[(x[1],x[2]) for x in ag2_traj]],'parking_pullout')
                            '''
                            '''
                            plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],'o')
                            plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],'x')
                            plt.show()
                            '''
                            
                        else:
                            dist_gap = u.calc_dist_gap(ag1_traj,ag2_traj)
                            #dist_gap = round(dist_gap * 2) / 2
                            if isinstance(m_type, tuple):
                                print(m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,success, sep=",")
                            else:
                                print(m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,success, sep=",")
                            
                            #res_writer.writerow([m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,success])
                            
        #self.show_results(succ_ct, strat_types, model_types, succ_map_type)            
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'strat_types'+'.result'), strat_types)                              
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_type'+'.result'), succ_map_type)  
        pickle_dump_to_dir(os.path.join(rg_constants.RESULTS_FILES,'succ_map_speed'+'.result'), succ_map_speed)  
        print('****** RESULTS CSV END ********')
        print('Total success counts:')
        for k,v in succ_ct.items():
            print(k,':',v)           
    
    def show_results(self,succ_ct,strat_types,model_types,succ_map_type):
        for k,v in succ_ct.items():
            print(k,':',v)         
        type_arr = list(itertools.product([0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]))
        #type_arr = [tuple([str(y) for y in list(x)]) for x in list(itertools.product([0,0.5,1,1.5,2],[10,12,15,17,20],[10,12,15,17,20]))]
        for k,v in strat_types.items():
            plt.figure()
            plt.title(str(k)+' - strat type')
            plt.gcf().subplots_adjust(bottom=.5)
            plt.bar(np.arange(len(v)), list(v.values()), align='center', alpha=0.5)
            plt.xticks(np.arange(len(v)), list(v.keys()),fontsize=6)
            plt.xticks(rotation=90)
            
            
            
        all_models = model_types + ['mspe','uspe']
        N = len(type_arr)
        all_data = []
        for midx,m in enumerate(all_models):
            data = []
            for t in type_arr:
                if t in succ_map_type:
                    if m in succ_map_type[t]:
                        data.append(succ_map_type[t][m])
                    else:
                        data.append(0)
                else:
                    data.append(0)
            all_data.append(data)
        fig, axs = plt.subplots(2)
        #fig.suptitle('pooling map-'+k,y=1.12)
        mdl_colors = ['r','g','b','c','black']
        barplots = [None] * len(all_models)
        for midx,m in enumerate(all_models):
            print(midx, all_data[midx])
            if midx == 0:
                barplots[midx] = axs[0].bar(np.arange(N), all_data[midx], color=mdl_colors[midx])
                print(midx, all_data[midx])
            else:
                if midx > 1:
                    _arr = np.asarray(all_data[:midx-1])
                    _bottom_sum = np.sum(_arr,axis=0)
                else:
                    _bottom_sum = all_data[0]
                    
                barplots[midx] = axs[0].bar(np.arange(N), all_data[midx], color=mdl_colors[midx], bottom = _bottom_sum)
                print(midx,_bottom_sum)
        axs[0].legend (tuple(barplots), tuple(all_models))
        xax_legend = np.asarray(type_arr).T
        rg_utils.plot_heatmap(plt, fig, axs[1], xax_legend)
        plt.show()
        f=1    

    
    
class ScenarioRunner():
    
    def __init__(self,scenario_type):
        rg_constants.SCENE_TYPE = scenario_type#('synthetic','intersection_clearance')
        rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    
    def run_scene(self):
        if rg_constants.SCENE_TYPE == ('synthetic','merge_before_intersection'):
            agent1_id, agent2_id = 1, 2
            run_id = 0
            rerun = True
            for ag_vels in itertools.product(np.linspace(0,10,20).tolist(),np.linspace(0,3,5).tolist()):
                run_id += 1
                file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(ag_vels[0]).replace('.',',')+'_'+str(ag_vels[1]).replace('.',',')
                if not rerun and os.path.isfile(os.path.join(rg_constants.TREE_FILES,file_id+'.gt')):
                    print('file',file_id,'processed....continuing')
                    continue
                try:
                    run_merge_before_intersection(run_id,agent1_id, agent2_id,ag_vels[0], ag_vels[1])
                except UnsupportedLatticeException:
                    print('file',file_id,'raised UnsupportedLatticeException....continuing')
        elif rg_constants.SCENE_TYPE == ('synthetic','parking_pullout'):
            agent1_id, agent2_id = 1, 2
            run_id = 0
            for ag_vels in itertools.product(np.linspace(0,1,5).tolist(),np.linspace(10,20,20).tolist()):
                run_id += 1
                file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(ag_vels[0]).replace('.',',')+'_'+str(ag_vels[1]).replace('.',',')
                if os.path.isfile(os.path.join(rg_constants.TREE_FILES,file_id+'.gt')):
                    print('file',file_id,'processed....continuing')
                    continue
                try:
                    run_parking_pullout(run_id,agent1_id, agent2_id,ag_vels[0], ag_vels[1])
                except UnsupportedLatticeException:
                    print('file',file_id,'raised UnsupportedLatticeException....continuing')
        elif rg_constants.SCENE_TYPE == ('synthetic','intersection_clearance'):
            agent1_id = 1
            run_id = 0
            for ag_vels in itertools.product([0,0.5,1,1.5,2],[10,12,15,17,20]):
                run_id += 1
                for ag2id in [2,3]:
                    file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(ag2id)+'_'+str(ag_vels[0]).replace('.',',')+'_'+str(ag_vels[1]).replace('.',',')
                    if os.path.isfile(os.path.join(rg_constants.TREE_FILES,file_id+'.gt')):
                        print('file',file_id,'processed....continuing')
                        continue
                    try:
                        run_intersection_clearance(run_id,agent1_id, ag2id,ag_vels[0], ag_vels[1])
                    except UnsupportedLatticeException:
                        print('file',file_id,'raised UnsupportedLatticeException....continuing')
        else:
            raise UnsupportedScenarioException(rg_constants.SCENE_TYPE)
            
                    
if __name__ == '__main__':
    rg_constants.DATASET = 'intersection_dataset'
    runner = ScenarioRunner(('synthetic','intersection_clearance'))
    runner.run_scene()
    
    
    '''
    res = ProcessResults()
    #res.visualization_only = ['1-2_1_20','1-3_1_10']
    #res.visualization_only = ['1-2_1,0_20,0']
    res.process_parking_pullout()
    '''
    
    
    #run_intersection_clearance(11,1, 2,1, 10)
    
    