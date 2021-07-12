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
from os.path import isfile, join
from equilibrium import game_tree
from maps.map_info import IntersectionClearanceMapInfo
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
    scene_def.add_agent(agent_tag='agent_1',agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, agent_waypoints=IntersectionClearanceMapInfo.lt_waypoints,agent_waypoint_segments=IntersectionClearanceMapInfo.lt_waypoint_segments, direction='L_S_W', file_id=file_id,initialize_db=True,start_ts=0,freq=0.5)
    scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=IntersectionClearanceMapInfo.st2_waypoints,agent_waypoint_segments=IntersectionClearanceMapInfo.st2_waypoint_segments, direction='L_N_S', file_id=file_id,initialize_db=True,start_ts=0,freq=0.5)
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

def process_results_intersection_clearance():
    rg_constants.SCENE_TYPE = ('synthetic','intersection_clearance')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    treefiles = [f for f in listdir(rg_constants.TREE_FILES) if isfile(join(rg_constants.TREE_FILES, f))]
    model_types = ['auto_resp','ql1_resp','robust_resp']
    results_map = dict()
    u = Utilities()
    for ctidx,resfile_name in enumerate(treefiles):
        #if ctidx > 10:
        #    break
        print('processing',ctidx+1,resfile_name)
        gt = all_utils.utils.pickle_load(os.path.join(rg_constants.TREE_FILES,resfile_name))
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
                                    results_map[resfile_name][m_type][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'].loaded_traj,l6n.path_from_root['agent_2'].loaded_traj)
                    if resfile_name not in results_map:
                        results_map[resfile_name] = {k:dict() for k in model_types}
                    if hasattr(l6n, 'on_mspe') and l6n.on_mspe[i,j]:
                        if 'mspe' not in results_map[resfile_name]:
                            results_map[resfile_name]['mspe'] = dict() 
                        results_map[resfile_name]['mspe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'].loaded_traj,l6n.path_from_root['agent_2'].loaded_traj)
                    if hasattr(l6n, 'on_uspe') and l6n.on_uspe[i,j]:
                        if 'uspe' not in results_map[resfile_name]:
                            results_map[resfile_name]['uspe'] = dict() 
                        results_map[resfile_name]['uspe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'].loaded_traj,l6n.path_from_root['agent_2'].loaded_traj)
                        
                        
    succ_ct = None
    succ_map_speed,succ_map_type = OrderedDict(), OrderedDict()
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
                                ag1_l, ag2_l, ag1_traj, ag2_traj = results_map[k12][m_type][type_comb1]
                                ag1_l2, ag3_l, ag1_traj, ag3_traj = results_map[k13][m_type][type_comb2]
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
                                    print('auto_resp',m_type,lt_speed,st1_speed,st2_speed,lt_type, st1_type, st2_type,max([ag1_l,ag1_l2]),ag2_l,ag3_l,'success' if success else 'fail')
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
                                        

    for k,v in succ_ct.items():
        print(k,':',v)         
    type_arr = list(itertools.product([0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]))
    #type_arr = [tuple([str(y) for y in list(x)]) for x in list(itertools.product([0,0.5,1,1.5,2],[10,12,15,17,20],[10,12,15,17,20]))]
    
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
                    
if __name__ == '__main__':
    rg_constants.SCENE_TYPE = ('synthetic','intersection_clearance')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    '''
    agent1_id = 1
    run_id = 0
    for ag_vels in itertools.product([0,0.5,1,1.5,2],[10,12,15,17,20]):
        run_id += 1
        for ag2id in [2,3]:
            file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(ag2id)+'_'+str(ag_vels[0]).replace('.',',')+'_'+str(ag_vels[1]).replace('.',',')
            if os.path.isfile(os.path.join(rg_constants.TREE_FILES,file_id+'.gt')):
                print('file',file_id,'processed....continuing')
                continue
            run_intersection_clearance(run_id,agent1_id, ag2id,ag_vels[0], ag_vels[1])
    '''
    '''
    fig, axs = plt.subplots(2)
    type_arr = [tuple([str(y) for y in list(x)]) for x in list(itertools.product([0,0.5,1,1.5,2],[10,12,15,17,20],[10,12,15,17,20]))]
    xax_legend = np.asarray(type_arr).T
    rg_utils.plot_heatmap(plt, fig, axs[1], xax_legend)
    plt.show()
    '''
    process_results_intersection_clearance()
    