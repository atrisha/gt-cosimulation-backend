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
from maps.map_info import NYCPedVeh
from equilibrium.game_tree import *
import rg_constants
log = constants.common_logger
from equilibrium.automata_strategies import *



def run_nyc_ped_veh(run_id,agent1_id, agent2_id,agent1_vel, agent2_vel):
    initialize_db=True
    freq=0.75
    file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(agent1_vel).replace('.',',')+'_'+str(agent2_vel).replace('.',',')
    rg_constants.CURRENT_RG_FILE_ID = file_id
    rg_constants.SCENE_TYPE = ('synthetic','nyc_ped_veh')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    scene_def = TwoAgentSyntheticScenarioDef(initialize_db=initialize_db,file_id=file_id)
    ''' agent-1 is vehicle, agent-2 is pedestrian'''
    scene_def.add_agent(agent_tag='agent_1',agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, agent_waypoints=NYCPedVeh.veh_nw_waypoints,agent_waypoint_segments=NYCPedVeh.veh_nw_waypoint_segments, direction='L_N_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq,agent_type='vehicle')
    scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=NYCPedVeh.ped_sn_waypoints,agent_waypoint_segments=None, direction='L_S_N', file_id=file_id,initialize_db=True,start_ts=0,freq=freq,agent_type='pedestrian')
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



def run_nyc_veh_veh(run_id,agent1_id, agent2_id,agent1_vel, agent2_vel):
    initialize_db=True
    freq=0.75
    file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(agent1_vel).replace('.',',')+'_'+str(agent2_vel).replace('.',',')
    rg_constants.CURRENT_RG_FILE_ID = file_id
    rg_constants.SCENE_TYPE = ('synthetic','nyc_veh_veh')
    rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
    scene_def = TwoAgentSyntheticScenarioDef(initialize_db=initialize_db,file_id=file_id)
    scene_def.add_agent(agent_tag='agent_1',agent_id=agent1_id, agent_init_velocity_mps=agent1_vel, agent_waypoints=NYCPedVeh.veh_nw_waypoints,agent_waypoint_segments=NYCPedVeh.veh_nw_waypoint_segments, direction='L_N_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq,agent_type='vehicle')
    scene_def.add_agent(agent_tag='agent_2',agent_id=agent2_id, agent_init_velocity_mps=agent2_vel, agent_waypoints=NYCPedVeh.veh_sw_waypoints,agent_waypoint_segments=NYCPedVeh.veh_sw_waypoint_segments, direction='L_S_W', file_id=file_id,initialize_db=True,start_ts=0,freq=freq,agent_type='vehicle')
    manv_map = {'agent_1':{'maneuvers':{'wait':None,'turn':None}},'agent_2':{'maneuvers':{'wait':None,'turn':None}}}
    maneuver_constraints = scene_def.setup_trajectory_constraints(manv_map)
    for k,v in maneuver_constraints['agent_1']['maneuvers'].items():
        setattr(v, 'path_degree', 3)
    for k,v in maneuver_constraints['agent_2']['maneuvers'].items():
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
                        if hasattr(l6n, 'on_espe') and l6n.on_espe[i,j]:
                            if 'espe' not in results_map[resfile_name]:
                                results_map[resfile_name]['espe'] = dict() 
                            results_map[resfile_name]['espe'][(i,j)] = (l6n.path_from_root['agent_1'].total_length,l6n.path_from_root['agent_2'].total_length,l6n.path_from_root['agent_1'],l6n.path_from_root['agent_2'])
        return results_map     
        
    
    def process_nyc_veh_veh(self):
        rg_constants.SCENE_TYPE = ('synthetic','nyc_veh_veh')
        rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
        rg_constants.RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'results'
        rg_constants.MAP_FILE = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\taxonomy_map.csv'
        results_map = self.generate_results_map()
        manv_codes = {'ped_walk':'P','ped_wait':'W','wait':'W','turn':'P'}
        model_types = ['auto_resp','ql1_resp','robust_resp']
        u = Utilities()
        tax_map = dict()
        with open(rg_constants.MAP_FILE, mode='r',newline='\n') as tax_map_file:
            sc_reader = csv.reader(tax_map_file, delimiter='|')
            for row in sc_reader:
                _k = ast.literal_eval(row[0])
                tax_map[(_k[0].upper(),_k[1].upper())] = (row[1],row[2])
        print('****** RESULTS CSV START ********')
        with open(os.path.join(rg_constants.RESULTS_FILES,'all_results'+'.csv'), mode='w') as resfile:
            res_writer = csv.writer(resfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            res_writer.writerow(['agent1_model_type','agent2_model_type','agent1_type','agent2_type','agent1_speed','agent2_speed','agent1_traj_length','agent2_traj_length','dist_gap','ag1_manv','ag2_manv'])
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
                        ag1_manv = ''.join([manv_codes[x] for x in ag1_traj_frag.manv_from_root])
                        ag2_manv = ''.join([manv_codes[x] for x in ag2_traj_frag.manv_from_root])
                        if (ag1_manv,ag2_manv) in tax_map:
                            ag1_manv_tax = tax_map[(ag1_manv,ag2_manv)][0]
                            ag2_manv_tax = tax_map[(ag1_manv,ag2_manv)][1]
                        else:
                            ag1_manv_tax, ag2_manv_tax =  ag1_manv, ag2_manv
                        success = False
                        '''
                        plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],color='blue')
                        plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],color='red')
                        plt.title(str(type_comb1)+'-'+str(m_type))
                        plt.show()
                        '''
                        ag2_speed = k12.split('_')[3].split('.')[0].replace(',','.')
                        ag1_speed = k12.split('_')[2].replace(',','.')
                        success = True
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
                                res_writer.writerow([m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag2_traj_frag.length,dist_gap,ag1_manv_tax,ag2_manv_tax])
                                print(m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,ag1_manv_tax,ag2_manv_tax, sep=",")
                            else:
                                res_writer.writerow([m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag2_traj_frag.length,dist_gap,ag1_manv_tax,ag2_manv_tax])
                                print(m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,ag1_manv_tax,ag2_manv_tax, sep=",")
                            
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
                            print(ag1_manv,ag2_manv)
                            plt.figure()
                            plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],'o')
                            plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],'x')
                            plt.figure()
                            plt.title('ag1-vel')
                            plt.plot(np.arange(len(ag1_traj)).tolist(),[x[3] for x in ag1_traj])
                            plt.figure()
                            plt.plot(np.arange(len(ag2_traj)).tolist(),[x[3] for x in ag2_traj])
                            plt.title('ag2-vel')
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
 
    
    def process_nyc_ped_veh(self):
        rg_constants.SCENE_TYPE = ('synthetic','nyc_ped_veh')
        rg_constants.TREE_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'game_trees'
        rg_constants.RESULTS_FILES = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\'+'results'
        rg_constants.MAP_FILE = 'D:\\repeated_games_data\\intersection_dataset\\'+rg_constants.SCENE_TYPE[0]+'\\'+rg_constants.SCENE_TYPE[1]+'\\taxonomy_map.csv'
        results_map = self.generate_results_map()
        manv_codes = {'ped_walk':'P','ped_wait':'W','wait':'W','turn':'P'}
        model_types = ['auto_resp','ql1_resp','robust_resp']
        u = Utilities()
        tax_map = dict()
        with open(rg_constants.MAP_FILE, mode='r',newline='\n') as tax_map_file:
            sc_reader = csv.reader(tax_map_file, delimiter='|')
            for row in sc_reader:
                _k = ast.literal_eval(row[0])
                tax_map[(_k[0].upper(),_k[1].upper())] = (row[1],row[2])
        print('****** RESULTS CSV START ********')
        with open(os.path.join(rg_constants.RESULTS_FILES,'all_results'+'.csv'), mode='w') as resfile:
            res_writer = csv.writer(resfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            res_writer.writerow(['agent1_model_type','agent2_model_type','agent1_type','agent2_type','agent1_speed','agent2_speed','agent1_traj_length','agent2_traj_length','dist_gap','ag1_manv','ag2_manv'])
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
                        ag1_manv = ''.join([manv_codes[x] for x in ag1_traj_frag.manv_from_root])
                        ag2_manv = ''.join([manv_codes[x] for x in ag2_traj_frag.manv_from_root])
                        if (ag2_manv,ag1_manv) in tax_map:
                            ''' needs to be flipped since pedestrian is agent2 and has row'''
                            ag1_manv_tax = tax_map[(ag2_manv,ag1_manv)][1]
                            ag2_manv_tax = tax_map[(ag2_manv,ag1_manv)][0]
                        else:
                            ag1_manv_tax, ag2_manv_tax =  ag1_manv, ag2_manv
                        success = False
                        '''
                        plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],color='blue')
                        plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],color='red')
                        plt.title(str(type_comb1)+'-'+str(m_type))
                        plt.show()
                        '''
                        ag2_speed = k12.split('_')[3].split('.')[0].replace(',','.')
                        ag1_speed = k12.split('_')[2].replace(',','.')
                        success = True
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
                                res_writer.writerow([m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag2_traj_frag.length,dist_gap,ag1_manv_tax,ag2_manv_tax])
                                print(m_type[0],m_type[1],type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,ag1_manv_tax,ag2_manv_tax, sep=",")
                            else:
                                res_writer.writerow([m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,ag1_traj_frag.length,ag2_traj_frag.length,dist_gap,ag1_manv_tax,ag2_manv_tax])
                                print(m_type,m_type,type_comb1[0],type_comb1[1],ag1_speed,ag2_speed,dist_gap,ag1_manv_tax,ag2_manv_tax, sep=",")
                            
                            if dist_gap not in strat_types:
                                strat_types[dist_gap] = dict()
                            if m_type not in strat_types[dist_gap]:
                                strat_types[dist_gap][m_type] = 0
                            else:
                                strat_types[dist_gap][m_type] += 1
                            '''
                            #if float(ag1_speed) > 0.8:
                            #analytics_obj = UniWeberAnalytics('769')
                            #analytics_obj.animate_scene([[(x[1],x[2]) for x in ag1_traj],[(x[1],x[2]) for x in ag2_traj]],'nyc_ped_veh')
                            
                            #fig, ax = plt.subplots()
                            print(ag1_manv,ag2_manv)
                            plt.figure()
                            plt.plot([x[1] for x in ag1_traj],[x[2] for x in ag1_traj],'o')
                            plt.plot([x[1] for x in ag2_traj],[x[2] for x in ag2_traj],'x')
                            plt.figure()
                            plt.title('ag1-vel')
                            plt.plot(np.arange(len(ag1_traj)).tolist(),[x[3] for x in ag1_traj])
                            plt.figure()
                            plt.plot(np.arange(len(ag2_traj)).tolist(),[x[3] for x in ag2_traj])
                            plt.title('ag2-vel')
                            plt.show()
                            #imextent = [598733, 598817, 4512594, 4512543]
                            #img = plt.imread("D:\\behavior modeling\\assorted_figures\\nyc_background.png")
                            #plt.xlim(598763, 598812)
                            #plt.xlim(4512579, 4512552)
                            #plt.axis('equal')
                            #ax.imshow(img, extent = imextent)
                            #plt.show()
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
        regenerate = True
        if rg_constants.SCENE_TYPE == ('synthetic','nyc_ped_veh'):
            agent1_id, agent2_id = 1, 2
            run_id = 0
            ''' agent-1 is vehicle, agent-2 is pedestrian'''
            for ag_vels in itertools.product(np.linspace(1,12,10).tolist(),np.linspace(1.3,1.8,10).tolist()):
                run_id += 1
                file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(ag_vels[0]).replace('.',',')+'_'+str(ag_vels[1]).replace('.',',')
                if not regenerate and os.path.isfile(os.path.join(rg_constants.TREE_FILES,file_id+'.gt')):
                    print('file',file_id,'processed....continuing')
                    continue
                try:
                    run_nyc_ped_veh(run_id,agent1_id, agent2_id,ag_vels[0], ag_vels[1])
                except UnsupportedLatticeException:
                    print('file',file_id,'raised UnsupportedLatticeException....continuing')
        elif rg_constants.SCENE_TYPE == ('synthetic','nyc_veh_veh'):
            agent1_id, agent2_id = 1, 2
            run_id = 0
            ''' agent-1 is vehicle, agent-2 is pedestrian'''
            for ag_vels in itertools.product(np.linspace(1,12,10).tolist(),np.linspace(1,12,10).tolist()):
                run_id += 1
                file_id = str(run_id)+'_'+str(agent1_id)+'-'+str(agent2_id)+'_'+str(ag_vels[0]).replace('.',',')+'_'+str(ag_vels[1]).replace('.',',')
                if os.path.isfile(os.path.join(rg_constants.TREE_FILES,file_id+'.gt')):
                    print('file',file_id,'processed....continuing')
                    continue
                try:
                    run_nyc_veh_veh(run_id,agent1_id, agent2_id,ag_vels[0], ag_vels[1])
                except UnsupportedLatticeException:
                    print('file',file_id,'raised UnsupportedLatticeException....continuing')
        else:
            raise UnsupportedScenarioException(rg_constants.SCENE_TYPE)
            
                    
if __name__ == '__main__':
    rg_constants.DATASET = 'intersection_dataset'
    '''
    runner = ScenarioRunner(('synthetic','nyc_veh_veh'))
    runner.run_scene()
    '''
    
    
    res = ProcessResults()
    #res.visualization_only = ['1-2_1_20','1-3_1_10']
    #res.visualization_only = ['1-2_8_5']
    res.process_nyc_ped_veh()
    
    
    
    
    #run_intersection_clearance(11,1, 2,1, 10)
    
    