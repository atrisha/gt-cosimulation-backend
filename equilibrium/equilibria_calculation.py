'''
Created on Mar 22, 2021

@author: Atrisha
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sqlite3
from shapely.geometry import Polygon, LineString
from equilibrium.utilities import Utilities
from code_utils.utils import lighten_color
import itertools
from equilibrium.gametree_objects import TrajectoryFragment, UnsupportedScenarioException
from typing import List
from code_utils import utils
from shapely.geometry import multipoint, point, linestring, multilinestring, GeometryCollection
from statistics import mean
from collections import OrderedDict
from code_utils.code_util_objects import RunContext
from code_utils.utils import get_all_level_nodes, get_nearest_node
from code_utils.utils import *
import copy
from maps.map_info import IntersectionClearanceMapInfo


#from figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line_issimple

COLOR = {
    True:  '#6699cc',
    False: '#ffcc33'
    }

def v_color(ob):
    return COLOR[ob.is_simple]

def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)

def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)

def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color=v_color(ob), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


show_plots = False


class EquilibriaSolution:
    
    def __init__(self, veh_br_map, peds_br_map):
        self.veh_br_map = veh_br_map
        self.peds_br_map = peds_br_map
        
    def set_veh_acts(self,veh_acts):
        self.veh_eq_acts = veh_acts
        
    def set_peds_acts(self,peds_acts):
        self.peds_eq_acts = peds_acts
        
    def set_veh_utils(self,veh_utils):
        self.veh_eq_utils = veh_utils
        
    def set_peds_utils(self,peds_utils):
        self.peds_eq_utils = peds_utils
        

class Equilibria:
        
    
    '''
        This takes in a key value pair of (m,t_idx) : u_i for a give -i action.
        Generates the best response set modulo manuver.
        Also generates the utility interval for the best response set.
    '''
    def solve(self,node,last_decision_level):
        if node.level == last_decision_level:
            try:
                peds_actions = node.actions['agent_2']
            except TypeError:
                f=1
                raise
            veh_actions = node.actions['agent_1']
            all_strategies = itertools.product(veh_actions,peds_actions)
            if hasattr(self, 'calc_equilibria'):
                self.calc_equilibria(veh_actions,peds_actions,node,last_decision_level)
            else:
                self.calc_response(veh_actions,peds_actions,node,last_decision_level)
        else:
            if not node.is_leaf: 
                for c in node.children:
                    self.solve(c,last_decision_level)
                peds_actions = node.actions['agent_2']
                veh_actions = node.actions['agent_1']
                all_strategies = itertools.product(veh_actions,peds_actions)
                if hasattr(self, 'calc_equilibria'):
                    self.calc_equilibria(veh_actions,peds_actions,node,last_decision_level)
                else:
                    self.calc_response(veh_actions,peds_actions,node,last_decision_level)
                
    def _max_util(self,x,ag_idx):
        if x is not None:
            return max([y.peds_eq_utils[0] for y in x]) if ag_idx == 1 else max([y.veh_eq_utils[0] for y in x])
        else:
            return np.nan
                
    def _process_LineString(self, eq_obj,belief_index):
        equil = []
        for eq_pt in eq_obj.coords:
            eq_strat = self.expand_equil(eq_strat = point.Point(eq_pt), veh_br_map = self.veh_best_response, peds_br_map = self.peds_best_response, belief_index=belief_index)
            equil.append(eq_strat)
        return equil
    
    def _process_Point(self, eq_obj, belief_index):
        equil = []
        eq_strat = self.expand_equil(eq_strat = eq_obj, veh_br_map = self.veh_best_response, peds_br_map = self.peds_best_response, belief_index=belief_index)
        equil.append(eq_strat)
        return equil
    
    def _process_MultiPoint(self, eq_obj, belief_index):
        equil = []
        for eq_pt in eq_obj:
            eq_strat = self.expand_equil(eq_strat = eq_pt, veh_br_map = self.veh_best_response, peds_br_map = self.peds_best_response, belief_index=belief_index)
            equil.append(eq_strat)
        return equil
    
    '''
    This is the key idea. It takes in the equilibrium and expands it based on mspe and uspe logic, and returns the expanded range.
    '''
    def expand_equil(self,eq_strat,veh_br_map, peds_br_map, belief_index):
        veh_eq_act = eq_strat.x
        peds_eq_act = eq_strat.y
        i,j = belief_index[0], belief_index[1]
        veh_br_key = min(veh_br_map.keys(), key=lambda x:abs(x-peds_eq_act))
        veh_br_range = (veh_br_map[veh_br_key][0][i,j], veh_br_map[veh_br_key][1][i,j], veh_br_map[veh_br_key][2][i,j])
        peds_br_key = min(peds_br_map.keys(), key=lambda x:abs(x-veh_eq_act))
        ''' for agent_2, the indexes should be flipped since i is always agent_1, and agent_2 br matrix had i as agent_2 threshold'''
        ''' on second thought, they need not be flipped, because peds_br_map has the entries based on the gamma matrix[1]'''
        peds_br_range = (peds_br_map[peds_br_key][0][i,j], peds_br_map[peds_br_key][1][i,j], peds_br_map[peds_br_key][2][i,j])
        return (veh_br_range,peds_br_range)
    
    def __init__(self,run_context = None):
        if run_context is None:
            self.run_context = RunContext()
        else:
            self.run_context = run_context
        l6_nodes = get_all_level_nodes(run_context.gt_obj.root, [], run_context.gt_obj.horizon)
        print('adding contd utils....')
        N = len(l6_nodes)
        for ctr,n in enumerate(l6_nodes):
            if hasattr(n, 'extd_util_calculated'):
                continue
            if rg_constants.SCENE_TYPE[0] == 'REAL' or (rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['merge_before_intersection']) \
                or (rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['parking_pullout']):
                print('adding contd utils....',ctr,'/',N)
            all_agents_ext_utils = self.calc_extended_util(n.path_from_root['agent_1'], n.path_from_root['agent_2'],n)
            
            #n.path_from_root['agent_1'].mean_safe_util_contd = mean_safe_util_contd
            #n.path_from_root['agent_2'].mean_safe_util_contd = mean_safe_util_contd
            n.mean_safe_util_contd_ag1 = all_agents_ext_utils[0][0]
            n.mean_progress_util_contd_ag1 = all_agents_ext_utils[0][1]
            n.mean_safe_util_contd_ag2 = all_agents_ext_utils[1][0]
            n.mean_progress_util_contd_ag2 = all_agents_ext_utils[1][1]
            n.extd_util_calculated = True
            
        print('adding contd utils....DONE')
    
           
    def calc_extended_util(self,ag1_traj_frag,ag2_traj_frag,n):
        if rg_constants.SCENE_TYPE[0] == 'REAL' or (rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['merge_before_intersection']) \
         or rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['parking_pullout','nyc_ped_veh']:
            ag1_motion_obj = self.run_context.maneuver_constraints['motion_info']['agent_1'][ag1_traj_frag.get_last().manv]
            try:
                ag2_motion_obj = self.run_context.maneuver_constraints['motion_info']['agent_2'][ag2_traj_frag.get_last().manv]
            except KeyError:
                f=1
                raise
            ag1_trajs = ag1_motion_obj.generate_extended_trajectory(ag1_traj_frag,{manv:manv for manv in self.run_context.maneuver_constraints['agent_1']['maneuvers'].keys()})
            ag2_trajs = ag2_motion_obj.generate_extended_trajectory(ag2_traj_frag,{manv:manv for manv in self.run_context.maneuver_constraints['agent_2']['maneuvers'].keys()})
            
            
            '''
            for ag1_t,ag2_t in itertools.product(ag1_trajs,ag2_trajs):
                plt.plot([x[1] for x in ag1_t],[x[2] for x in ag1_t])
                plt.plot([x[1] for x in ag2_t],[x[2] for x in ag2_t])
            plt.xlim(538780, 538890)
            plt.ylim(4813970, 4814055)
            plt.show()
            '''
            u = Utilities()
            dist_gaps = []
            for ag1_t,ag2_t in itertools.product(ag1_trajs,ag2_trajs):
                _dg = u.calc_dist_gap(ag1_t, ag2_t, (1,2))
                dist_gaps.append(_dg)
            avg_dg = np.mean(dist_gaps)
            mean_safe_util = u.calc_safe_payoff(avg_dg)
            if rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['parking_pullout']:
                ag1_trajl = n.path_from_root['agent_1'].total_length
                ag2_trajl = n.path_from_root['agent_2'].total_length
                if ag2_trajl < 40 and ag1_trajl > 6:
                    ag1_ext_utils = (-1,0.5)
                    ag2_ext_utils = (0.5,0)
                    return [ag1_ext_utils,ag2_ext_utils]
                else:
                    return [(mean_safe_util,None), (mean_safe_util,None)]
            else:
                return [(mean_safe_util,None), (mean_safe_util,None)]
        elif rg_constants.SCENE_TYPE[0] == 'synthetic' and rg_constants.SCENE_TYPE[1] in ['test','intersection_clearance']:
            ext_utils = None
            agent_2_id = int(n._tree_link.file_id.split('_')[1].split('-')[1])
            ag1_trajl = n.path_from_root['agent_1'].total_length
            ag2_trajl = n.path_from_root['agent_2'].total_length
            if ag1_trajl < 28:
                ag1_ext_utils = (-1,-1)
            else:
                ag1_ext_utils = (1,1)
            dist_2_intersection_start, dist_2_intersection_end = IntersectionClearanceMapInfo.st1_on_intersection_distance if agent_2_id==2 else IntersectionClearanceMapInfo.st2_on_intersection_distance
            if ag2_trajl < dist_2_intersection_start:
                ag2_ext_utils = (1,0.5)
            elif dist_2_intersection_start <= ag2_trajl < dist_2_intersection_end:
                ag2_ext_utils = (-1,-1)
            else:
                ag2_ext_utils = (0.5,1)
            return [ag1_ext_utils,ag2_ext_utils]
        else:
            raise UnsupportedScenarioException(str(rg_constants.SCENE_TYPE)) 
                
                
                
class AutoStrategyResponse(Equilibria):
    
    def calc_response(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        gamma_matrix.reverse()
        self.gamma_matrix = gamma_matrix
        ''' agent_1=0 agent_2 = 1'''
        #gamma_matrix = [np.linspace(start=-1, stop=1, num=20), np.linspace(start=-1, stop=1, num=20)]
        u = Utilities()
        veh_acts.sort(key=lambda x: x.length)
        ped_acts.sort(key=lambda x: x.length)
        ''' agent_2 best response to agent_1's trajectory length'''
        node.auto_strategy_response = dict()
        agent1_br_map, agent2_br_map = OrderedDict(), OrderedDict()
        
        ag_2_resp = []
        for ag1_tf,ag2_tf in zip(veh_acts,ped_acts):
            dist_gap = u.calc_dist_gap(veh_traj = ag1_tf.loaded_traj_frag, ped_traj = ag2_tf.loaded_traj_frag)
            safety_payoff = u.calc_safe_payoff(dist_gap)
            if not node.is_root:
                ag1_ac_gamma = node.automata_strategy_info['agent_1']['ac_auto_gamma']
                ag1_nac_gamma = node.automata_strategy_info['agent_1']['nac_auto_gamma']
            else:
                ag1_ac_gamma = (-1,1)
                ag1_nac_gamma = (-1,1)
            ag1_tf.is_ac_likely, ag2_tf.is_ac_likely = True,True
            ag1_tf.is_nac_likely, ag2_tf.is_nac_likely = True,True
            if ag1_tf.manv == self.run_context.manv_map['agent_1']['proceed'] and not type(ag1_ac_gamma) is bool and(safety_payoff < ag1_ac_gamma[0] or safety_payoff < ag1_ac_gamma[1]):
                ag1_tf.is_ac_likely = False
            if ag1_tf.manv == self.run_context.manv_map['agent_1']['wait'] and not type(ag1_nac_gamma) is bool and (safety_payoff > ag1_nac_gamma[0] or safety_payoff > ag1_nac_gamma[1]):
                ag1_tf.is_nac_likely = False
            '''
            check the running dynamics and if this action of agent 1 is unlikely based on the running
            dynamics, then there is no need to respond, since this action will never be taken.
            '''
            if (self.run_context.acc_dynamic and not ag1_tf.is_ac_likely) \
                and (self.run_context.non_acc_dynamic and not ag1_tf.is_nac_likely):
                continue
            safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
            step_util = u.combine_utils(u.progress_payoff_dist(ag2_tf.length, 'agent_2'), safe_payoff_for_dist, gamma_matrix[1])
            if node.level == last_decision_level:
                ext_safe_utils = ag2_tf._next_node.mean_safe_util_contd_ag2
                ext_prog_utils = ag2_tf._next_node.mean_progress_util_contd_ag2 if ag2_tf._next_node.mean_progress_util_contd_ag2 is not None else u.progress_payoff_dist(ag2_tf.length, 'agent_2')
                cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[1])
                _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
            else:
                if len(ag2_tf._next_node.children) == 0:
                    continue
                else:
                    cont_util = ag2_tf._next_node.auto_strategy_response['agent_2'][0]['utils']
                    safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
                #cont_util = cont_util.T
                #cont_util = max([max(x.peds_eq_utils) for x in peds_frag._next_node.equilibrium_solutions])
            _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
            #if np.isnan(cont_util).any():
            #    continue
            _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
            if ag2_tf._next_node is not None:
                if hasattr(ag2_tf._next_node, 'util_info'):
                    if 'auto_resp' not in ag2_tf._next_node.util_info:
                        ag2_tf._next_node.util_info['auto_resp'] = dict()
                    ag2_tf._next_node.util_info['auto_resp']['agent_2'] = _util_entry_matrix
                else:
                    ag2_tf._next_node.util_info = dict()
                    ag2_tf._next_node.util_info['auto_resp'] = dict()
                    ag2_tf._next_node.util_info['auto_resp']['agent_2'] = _util_entry_matrix
                    
            manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=ag2_tf.manv)
            traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ag2_tf.length)
            _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
            ag_2_resp.append(_resp_entry)
            
        if len(ag_2_resp) == 0:
            self.agent_2_auto_strategy_response = None
        else:
            ag_2_resp = np.array(ag_2_resp)
            ag_2_resp_sorted = np.sort(ag_2_resp,axis=0,order='utils')[::-1]
            upper_bound_matrix = np.copy(ag_2_resp_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], ag_2_resp_sorted.shape[0], axis=0)
            lower_bound_safety_matrix = np.copy(ag_2_resp_sorted)
            lower_safety_bound_matrix_check = ag_2_resp_sorted['safe_utils'] >= gamma_matrix[1]
            lower_bound_matrix = np.copy(ag_2_resp_sorted)
            _x1 = ag_2_resp_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = ag_2_resp_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            if np.all(_x3):
                ag2_lower_bound_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(_x3)):
                ag2_lower_bound_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(_x3, axis=0) - 1
                ag2_lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
                
            if np.all(lower_safety_bound_matrix_check):
                ag2_lower_bound_safety_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(lower_safety_bound_matrix_check)):
                ag2_lower_bound_safety_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(lower_safety_bound_matrix_check, axis=0) - 1
                ag2_lower_bound_safety_matrix = np.take_along_axis(lower_bound_safety_matrix,selected_indices[np.newaxis],axis=0)[0]
            
            ag2_upper_bound_matrix = upper_bound_matrix[0,:,:]
            
            node.auto_strategy_response['agent_2'] = (ag2_upper_bound_matrix,ag2_lower_bound_matrix,ag2_lower_bound_safety_matrix)
            node.auto_strategy_response['agent_2_all_responses'] = ag_2_resp
        
        ag_1_resp = []
        for ag2_tf,ag1_tf in zip(ped_acts,veh_acts):
            dist_gap = u.calc_dist_gap(veh_traj = ag1_tf.loaded_traj_frag, ped_traj = ag2_tf.loaded_traj_frag)
            safety_payoff = u.calc_safe_payoff(dist_gap)
            if node.is_root:
                ag2_ac_gamma = (-1,1)
                ag2_nac_gamma = (-1,1)
            else:
                ag2_ac_gamma = node.automata_strategy_info['agent_2']['ac_auto_gamma']
                ag2_nac_gamma = node.automata_strategy_info['agent_2']['nac_auto_gamma']
            ag2_tf.is_ac_likely, ag1_tf.is_ac_likely = True,True
            ag2_tf.is_nac_likely, ag1_tf.is_nac_likely = True,True
            if ag2_tf.manv == self.run_context.manv_map['agent_2']['proceed'] and not type(ag2_ac_gamma) is bool and(safety_payoff < ag2_ac_gamma[0] or safety_payoff < ag2_ac_gamma[1]):
                ag2_tf.is_ac_likely = False
            if ag2_tf.manv == self.run_context.manv_map['agent_2']['wait'] and not type(ag2_nac_gamma) is bool and (safety_payoff > ag2_nac_gamma[0] or safety_payoff > ag2_nac_gamma[1]):
                ag2_tf.is_nac_likely = False
            '''
            check the running dynamics and if this action of agent 1 is unlikely based on the running
            dynamics, then there is no need to respond, since this action will never be taken.
            '''
            if (self.run_context.acc_dynamic and not ag2_tf.is_ac_likely) \
                and (self.run_context.non_acc_dynamic and not ag2_tf.is_nac_likely):
                continue
            safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
            step_util = u.combine_utils(u.progress_payoff_dist(ag1_tf.length, 'agent_1'), safe_payoff_for_dist, gamma_matrix[0])
            if node.level == last_decision_level:
                ext_safe_utils = ag1_tf._next_node.mean_safe_util_contd_ag1
                ext_prog_utils = ag1_tf._next_node.mean_progress_util_contd_ag1 if ag1_tf._next_node.mean_progress_util_contd_ag1 is not None else u.progress_payoff_dist(ag1_tf.length, 'agent_1')
                cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[0])
                _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
            else:
                if len(ag1_tf._next_node.children) == 0:
                    continue
                else:
                    cont_util = ag1_tf._next_node.auto_strategy_response['agent_1'][0]['utils']
                    safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
            _resp_entry = np.empty(shape= gamma_matrix[0].shape, dtype=np.record)
            #if np.isnan(cont_util).any():
            #    continue
            _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
            if ag1_tf._next_node is not None:
                if hasattr(ag1_tf._next_node, 'util_info'):
                    if 'auto_resp' not in ag1_tf._next_node.util_info:
                        ag1_tf._next_node.util_info['auto_resp'] = dict()
                    ag1_tf._next_node.util_info['auto_resp']['agent_1'] = _util_entry_matrix
                else:
                    ag1_tf._next_node.util_info = dict()
                    ag1_tf._next_node.util_info['auto_resp'] = dict()
                    ag1_tf._next_node.util_info['auto_resp']['agent_1'] = _util_entry_matrix
        
            manv_str_arr = np.full(shape = gamma_matrix[0].shape, fill_value=ag1_tf.manv)
            traj_l_arr = np.full(shape = gamma_matrix[0].shape, fill_value = ag1_tf.length)
            _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
            ag_1_resp.append(_resp_entry)
        if len(ag_1_resp) == 0:
            self.agent_1_auto_strategy_response = None
        else:
            ag_1_resp = np.array(ag_1_resp)
            ag_1_resp_sorted = np.sort(ag_1_resp,axis=0,order='utils')[::-1]               
            upper_bound_matrix = np.copy(ag_1_resp_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], ag_1_resp_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(ag_1_resp_sorted)
            lower_bound_safety_matrix = np.copy(ag_1_resp_sorted)
            lower_safety_bound_matrix_check = ag_1_resp_sorted['safe_utils'] >= gamma_matrix[0]
            _x1 = ag_1_resp_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = ag_1_resp_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            if np.all(_x3):
                ag1_lower_bound_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(_x3)):
                ag1_lower_bound_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(_x3, axis=0) - 1
                ag1_lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
                
            if np.all(lower_safety_bound_matrix_check):
                ag1_lower_bound_safety_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(lower_safety_bound_matrix_check)):
                ag1_lower_bound_safety_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(lower_safety_bound_matrix_check, axis=0) - 1
                ag1_lower_bound_safety_matrix = np.take_along_axis(lower_bound_safety_matrix,selected_indices[np.newaxis],axis=0)[0]
            
            ag1_upper_bound_matrix = upper_bound_matrix[0,:,:]
                         
            node.auto_strategy_response['agent_1'] = (ag1_upper_bound_matrix,ag1_lower_bound_matrix, ag1_lower_bound_safety_matrix)
            node.auto_strategy_response['agent_1_all_responses'] = ag_1_resp
        
    
class RobustResponse(Equilibria):

    def calc_response(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        gamma_matrix.reverse()
        ''' agent_1=0 agent_2 = 1'''
        node.robust_response = {'agent_1' : np.empty(shape= (gamma_matrix[0].shape[0],1), dtype=object),
                                'agent_2' : np.empty(shape= (gamma_matrix[1].shape[1],1), dtype=object)}
        node.robust_response_type = dict()
        for i in np.arange(node.robust_response['agent_1'].shape[0]):
            ''' agent_1 private tolerance type is i '''
            
            if (not hasattr(node, 'on_mspe') or  (hasattr(node, 'on_mspe') and np.all(node.on_mspe[i,:] == False))) and (not hasattr(node, 'on_uspe') or  (hasattr(node, 'on_uspe') and np.all(node.on_uspe[i,:] == False))):
                no_spe = True
            else:
                no_spe = False
                all_eq_list = node.equilibrium_solutions[i,:]
                ''' this is just a response, so te best response function can be None'''
                            
                min_util,min_idx = np.inf,(0,0)
                for bl_idx,all_eq in enumerate(all_eq_list):
                    if all_eq is not None and hasattr(node, 'on_mspe') and node.on_mspe[i,bl_idx]:
                        for idx,eq in enumerate(all_eq):
                            if eq.veh_eq_utils[0] < min_util:
                                min_util = eq.veh_eq_utils[0]
                                min_idx = (bl_idx,idx)
                for bl_idx,all_eq in enumerate(all_eq_list):
                    if all_eq is not None and hasattr(node, 'on_uspe') and node.on_uspe[i,bl_idx]:
                        for idx,eq in enumerate(all_eq):
                            if eq.veh_eq_utils[0] < min_util:
                                min_util = eq.veh_eq_utils[0]
                                min_idx = (bl_idx,idx)
                
            soln = EquilibriaSolution(None,None)
            
            ''' just take the first one because for automata strategy response, all columns are same '''
            if node.auto_strategy_response is not None and len(node.auto_strategy_response)>1:
                all_auto_resps = (node.auto_strategy_response['agent_1'][0][i,0], node.auto_strategy_response['agent_1'][1][i,0])
                soln.veh_eq_acts = (all_auto_resps[0]['traj_l'],all_auto_resps[1]['traj_l'])
                soln.veh_eq_utils = (all_auto_resps[0]['utils'],all_auto_resps[1]['utils'])
                if no_spe:
                    node.robust_response['agent_1'][i] = soln
                    node.robust_response_type['agent_1'] = 'auto'
                else:
                    robust_resp_to_spe = all_eq_list[min_idx[0]][min_idx[1]]
                    node.robust_response['agent_1'][i] = soln if soln.veh_eq_utils[0] < robust_resp_to_spe.veh_eq_utils[0] else copy.copy(robust_resp_to_spe)
                    node.robust_response['agent_1'][i,0].veh_eq_utils = (node.robust_response['agent_1'][i,0].veh_eq_utils[0],soln.veh_eq_utils[1] if soln.veh_eq_utils[1] > robust_resp_to_spe.veh_eq_utils[1] else robust_resp_to_spe.veh_eq_utils[1])
                    node.robust_response['agent_1'][i,0].veh_eq_acts = (node.robust_response['agent_1'][i,0].veh_eq_acts[0],soln.veh_eq_acts[1] if soln.veh_eq_utils[1] > robust_resp_to_spe.veh_eq_utils[1] else robust_resp_to_spe.veh_eq_acts[1])
                    node.robust_response_type['agent_1'] = 'auto' if soln.veh_eq_utils[0] < robust_resp_to_spe.veh_eq_utils[0] else 'spe'
                    node.robust_response['agent_1'][i,0].veh_eq_utils = (node.robust_response['agent_1'][i,0].veh_eq_utils[0],node.robust_response['agent_1'][i,0].veh_eq_utils[1] if node.robust_response['agent_1'][i,0].veh_eq_utils[1] > robust_resp_to_spe.veh_eq_utils[2] else robust_resp_to_spe.veh_eq_utils[2])
                    node.robust_response['agent_1'][i,0].veh_eq_acts = (node.robust_response['agent_1'][i,0].veh_eq_acts[0],node.robust_response['agent_1'][i,0].veh_eq_acts[1] if node.robust_response['agent_1'][i,0].veh_eq_utils[1] > robust_resp_to_spe.veh_eq_utils[2] else robust_resp_to_spe.veh_eq_acts[2]) 
            else:
                if not no_spe and all_eq_list[min_idx[0]] is not None:
                    robust_resp_to_spe = all_eq_list[min_idx[0]][min_idx[1]]
                    node.robust_response['agent_1'][i] = robust_resp_to_spe
                    node.robust_response_type['agent_1'] = 'spe'
                else:
                    node.robust_response['agent_1'][i] = None 
                    node.robust_response_type['agent_1'] = None
        
        for j in np.arange(node.robust_response['agent_2'].shape[0]):
            ''' agent_2 private tolerance type is j '''
            if (not hasattr(node, 'on_mspe') or  (hasattr(node, 'on_mspe') and np.all(node.on_mspe[:,j] == False))) and (not hasattr(node, 'on_uspe') or  (hasattr(node, 'on_uspe') and np.all(node.on_uspe[:,j] == False))):
                no_spe = True
            else:
                no_spe = False
                all_eq_list = node.equilibrium_solutions[:,j]
                ''' this is just a response, so te best response function can be None'''
                
                min_util,min_idx = np.inf,(0,0)
                for bl_idx,all_eq in enumerate(all_eq_list):
                    if all_eq is not None and hasattr(node, 'on_mspe') and node.on_mspe[bl_idx,j]:
                        for idx,eq in enumerate(all_eq):
                            if eq.peds_eq_utils[0] < min_util:
                                min_util = eq.peds_eq_utils[0]
                                min_idx = (bl_idx,idx)
                for bl_idx,all_eq in enumerate(all_eq_list):
                    if all_eq is not None and hasattr(node, 'on_uspe') and node.on_uspe[bl_idx,j]:
                        for idx,eq in enumerate(all_eq):
                            if eq.peds_eq_utils[0] < min_util:
                                min_util = eq.peds_eq_utils[0]
                                min_idx = (bl_idx,idx)
                
            soln = EquilibriaSolution(None,None)
            if node.auto_strategy_response is not None and len(node.auto_strategy_response)>1:
                ''' just take the first one because for automata strategy response, all columns are same '''
                all_auto_resps = (node.auto_strategy_response['agent_2'][0][0,j], node.auto_strategy_response['agent_2'][1][0,j])
                soln.peds_eq_acts = (all_auto_resps[0]['traj_l'],all_auto_resps[1]['traj_l'])
                soln.peds_eq_utils = (all_auto_resps[0]['utils'],all_auto_resps[1]['utils'])
                if no_spe:
                    node.robust_response['agent_2'][j] = soln
                    node.robust_response_type['agent_2'] = 'auto'
                else:
                    robust_resp_to_spe = all_eq_list[min_idx[0]][min_idx[1]]
                    node.robust_response['agent_2'][j] = soln if soln.peds_eq_utils[0] < robust_resp_to_spe.peds_eq_utils[0] else copy.copy(robust_resp_to_spe)
                    node.robust_response['agent_2'][j,0].peds_eq_utils = (node.robust_response['agent_2'][j,0].peds_eq_utils[0],soln.peds_eq_utils[1] if soln.peds_eq_utils[1] > robust_resp_to_spe.peds_eq_utils[1] else robust_resp_to_spe.peds_eq_utils[1])
                    node.robust_response['agent_2'][j,0].peds_eq_acts = (node.robust_response['agent_2'][j,0].peds_eq_acts[0],soln.peds_eq_acts[1] if soln.peds_eq_utils[1] > robust_resp_to_spe.peds_eq_utils[1] else robust_resp_to_spe.peds_eq_acts[1]) 
                    node.robust_response_type['agent_2'] = 'auto' if soln.peds_eq_utils[0] < robust_resp_to_spe.peds_eq_utils[0] else 'spe'
                    node.robust_response['agent_2'][j,0].peds_eq_utils = (node.robust_response['agent_2'][j,0].peds_eq_utils[0],node.robust_response['agent_2'][j,0].peds_eq_utils[1] if node.robust_response['agent_2'][j,0].peds_eq_utils[1] > robust_resp_to_spe.peds_eq_utils[2] else robust_resp_to_spe.peds_eq_utils[2])
                    node.robust_response['agent_2'][j,0].peds_eq_acts = (node.robust_response['agent_2'][j,0].peds_eq_acts[0],node.robust_response['agent_2'][j,0].peds_eq_acts[1] if node.robust_response['agent_2'][j,0].peds_eq_acts[1] > robust_resp_to_spe.peds_eq_utils[2] else robust_resp_to_spe.peds_eq_acts[2]) 
            else:
                if not no_spe and all_eq_list[min_idx[0]] is not None:
                    robust_resp_to_spe = all_eq_list[min_idx[0]][min_idx[1]]
                    node.robust_response['agent_2'][j] = robust_resp_to_spe
                    node.robust_response_type['agent_2'] = 'spe'
                else:
                    node.robust_response['agent_2'][j] = None
                    node.robust_response_type['agent_2'] = None
        
class Ql0Model(Equilibria):
    
    def calc_response(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        precision_parm = self.run_context.precision_parm
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        gamma_matrix.reverse()
        ''' agent_1=0 agent_2 = 1'''
        node.ql0ql0_response = {'response':{'agent_1' : np.empty(shape= (gamma_matrix[0].shape[0],1), dtype=object),
                                'agent_2' : np.empty(shape= (gamma_matrix[1].shape[1],1), dtype=object)},
                             'distribution':{'agent_1' : None,
                                'agent_2' : None},
                             'all_responses':{'agent_1' : None,
                                'agent_2' : None}}
        agent1_distr,agent2_distr = dict(), dict()
            
        
        #for i in np.arange(node.ql1_response['agent_2'].shape[0]):
        #    ''' agent_1 private tolerance type is i '''
        
        u = Utilities()
        ag_2_resp = []
        for ag2_tf in ped_acts:
            ag_2_resp_per_ag1_act = []
            for ag1_tf in veh_acts:
                dist_gap = u.calc_dist_gap(veh_traj = ag1_tf.loaded_traj_frag, ped_traj = ag2_tf.loaded_traj_frag)
                safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
                step_util = u.combine_utils(u.progress_payoff_dist(ag2_tf.length, 'agent_2'), safe_payoff_for_dist, gamma_matrix[1])
                if node.level == last_decision_level:
                    ext_safe_utils = ag2_tf._next_node.mean_safe_util_contd_ag2
                    ext_prog_utils = ag2_tf._next_node.mean_progress_util_contd_ag2 if ag2_tf._next_node.mean_progress_util_contd_ag2 is not None else u.progress_payoff_dist(ag2_tf.length, 'agent_2')
                    cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[1])
                    _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                    safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
                    cont_util = step_util
                else:
                    if len(ag2_tf._next_node.children) == 0:
                        continue
                    else:
                        cont_util = ag2_tf._next_node.ql0ql0_response['response']['agent_2']['utils']
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
                _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
                _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                if ag1_tf._next_node is not None:
                    if hasattr(ag1_tf._next_node, 'util_info'):
                        if 'ql1' not in ag1_tf._next_node.util_info:
                            ag1_tf._next_node.util_info['ql1'] = _util_entry_matrix
                    else:
                        ag1_tf._next_node.util_info = dict()
                        ag1_tf._next_node.util_info['ql1'] = _util_entry_matrix
                
                manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=ag2_tf.manv)
                traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ag2_tf.length)
                _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
                ag_2_resp_per_ag1_act.append(_resp_entry)
            resp_vect = np.array(ag_2_resp_per_ag1_act)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            ag_2_resp.append(np.copy(resp_vect_sorted[0,:,:]))
            if hasattr(ag2_tf._next_node, 'util_info'):
                if 'qlk' not in ag2_tf._next_node.util_info:
                    ag2_tf._next_node.util_info['qlk'] = dict()
                ag2_tf._next_node.util_info['qlk']['agent_2'] = resp_vect_sorted[0,:,:]['utils']
            else:
                ag2_tf._next_node.util_info = dict()
                ag2_tf._next_node.util_info['qlk'] = dict()
                ag2_tf._next_node.util_info['qlk']['agent_2'] = resp_vect_sorted[0,:,:]['utils']
                 
        resp_vect = np.array(ag_2_resp)
        resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
        _denom = [x['utils'] for x in resp_vect_sorted]
        _denom = [np.exp(precision_parm*x) for x in _denom]
        _denom = sum(_denom)
        for r in resp_vect_sorted:
            _tl = r['traj_l'][0,0]
            _u = r['utils']
            _prob = np.divide(np.exp(precision_parm*_u),_denom)
            agent2_distr[_tl] = np.copy(_prob)[0,:]
        ''' agent 2 is the ql0 agent, therefore this is its response based on maxmax behavior'''
        node.ql0ql0_response['response']['agent_2'] =  np.copy(resp_vect_sorted[0,:,:])
        
        
        ag_1_resp = []
        for ag2_tf in ped_acts:
            ag_1_resp_per_ag2_act = []
            for ag1_tf in veh_acts:
                dist_gap = u.calc_dist_gap(veh_traj = ag1_tf.loaded_traj_frag, ped_traj = ag2_tf.loaded_traj_frag)
                safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
                step_util = u.combine_utils(u.progress_payoff_dist(ag1_tf.length, 'agent_1'), safe_payoff_for_dist, gamma_matrix[0])
                if node.level == last_decision_level:
                    ext_safe_utils = ag1_tf._next_node.mean_safe_util_contd_ag1
                    ext_prog_utils = ag1_tf._next_node.mean_progress_util_contd_ag1 if ag1_tf._next_node.mean_progress_util_contd_ag1 is not None else u.progress_payoff_dist(ag1_tf.length, 'agent_1')
                    cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[0])
                    _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                    safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
                    cont_util = step_util
                else:
                    if len(ag1_tf._next_node.children) == 0:
                        continue
                    else:
                        cont_util = ag1_tf._next_node.ql0ql0_response['response']['agent_1']['utils']
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
                _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
                _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                if ag1_tf._next_node is not None:
                    if hasattr(ag1_tf._next_node, 'util_info'):
                        if 'ql1' not in ag1_tf._next_node.util_info:
                            ag1_tf._next_node.util_info['ql1'] = _util_entry_matrix
                    else:
                        ag1_tf._next_node.util_info = dict()
                        ag1_tf._next_node.util_info['ql1'] = _util_entry_matrix
                
                manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=ag1_tf.manv)
                traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ag1_tf.length)
                _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
                ag_1_resp_per_ag2_act.append(_resp_entry)
            resp_vect = np.array(ag_1_resp_per_ag2_act)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            ag_1_resp.append(np.copy(resp_vect_sorted[0,:,:]))
            if hasattr(ag1_tf._next_node, 'util_info'):
                if 'qlk' not in ag1_tf._next_node.util_info:
                    ag1_tf._next_node.util_info['qlk'] = dict()
                ag1_tf._next_node.util_info['qlk']['agent_1'] = resp_vect_sorted[0,:,:]['utils']
            else:
                ag1_tf._next_node.util_info = dict()
                ag1_tf._next_node.util_info['qlk'] = dict()
                ag1_tf._next_node.util_info['qlk']['agent_1'] = resp_vect_sorted[0,:,:]['utils']
                 
        resp_vect = np.array(ag_1_resp)
        resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
        _denom = [x['utils'] for x in resp_vect_sorted]
        _denom = [np.exp(precision_parm*x) for x in _denom]
        _denom = sum(_denom)
        for r in resp_vect_sorted:
            _tl = r['traj_l'][0,0]
            _u = r['utils']
            _prob = np.divide(np.exp(precision_parm*_u),_denom)
            agent1_distr[_tl] = np.copy(_prob)[0,:]
        
        ''' agent 1 is the ql0 agent too in this, therefore this is its response based on maxmax behavior'''
        node.ql0ql0_response['response']['agent_1'] =  np.copy(resp_vect_sorted[0,:,:])
        node.ql0ql0_response['distribution']['agent_1'] = agent1_distr
        node.ql0ql0_response['distribution']['agent_2'] = agent2_distr
        
        f=1
    
    
class Ql1Model(Equilibria):
    
    
    def calc_response(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        
        precision_parm = self.run_context.precision_parm
        ''' find the optimal response for both agents. 
            Create a map of action(trajectory_length) -> probability, based on precision_parm
            each agent best response to that belief distribution '''
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        gamma_matrix.reverse()
        ''' agent_1=0 agent_2 = 1'''
        node.ql1_response = {'response':{'agent_1' : np.empty(shape= (gamma_matrix[0].shape[0],1), dtype=object),
                                'agent_2' : np.empty(shape= (gamma_matrix[1].shape[1],1), dtype=object)},
                             'distribution':{'agent_1' : None,
                                'agent_2' : None},
                             'all_responses':{'agent_1' : None,
                                'agent_2' : None}}
        agent1_distr,agent2_distr = dict(), dict()
            
        
        #for i in np.arange(node.ql1_response['agent_2'].shape[0]):
        #    ''' agent_1 private tolerance type is i '''
        
        u = Utilities()
        ag_2_resp = []
        for ag2_tf in ped_acts:
            ag_2_resp_per_ag1_act = []
            for ag1_tf in veh_acts:
                dist_gap = u.calc_dist_gap(veh_traj = ag1_tf.loaded_traj_frag, ped_traj = ag2_tf.loaded_traj_frag)
                safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
                step_util = u.combine_utils(u.progress_payoff_dist(ag2_tf.length, 'agent_2'), safe_payoff_for_dist, gamma_matrix[1])
                if node.level == last_decision_level:
                    ext_safe_utils = ag2_tf._next_node.mean_safe_util_contd_ag2
                    ext_prog_utils = ag2_tf._next_node.mean_progress_util_contd_ag2 if ag2_tf._next_node.mean_progress_util_contd_ag2 is not None else u.progress_payoff_dist(ag2_tf.length, 'agent_2')
                    cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[1])
                    _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                    safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
                    cont_util = step_util
                else:
                    if len(ag2_tf._next_node.children) == 0:
                        continue
                    else:
                        cont_util = ag2_tf._next_node.ql1_response['response']['agent_2']['utils']
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
                _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
                _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                if ag1_tf._next_node is not None:
                    if hasattr(ag1_tf._next_node, 'util_info'):
                        if 'ql1' not in ag1_tf._next_node.util_info:
                            ag1_tf._next_node.util_info['ql1'] = _util_entry_matrix
                    else:
                        ag1_tf._next_node.util_info = dict()
                        ag1_tf._next_node.util_info['ql1'] = _util_entry_matrix
                
                manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=ag2_tf.manv)
                traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ag2_tf.length)
                _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
                ag_2_resp_per_ag1_act.append(_resp_entry)
            resp_vect = np.array(ag_2_resp_per_ag1_act)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            ag_2_resp.append(np.copy(resp_vect_sorted[0,:,:]))
            if hasattr(ag2_tf._next_node, 'util_info'):
                if 'qlk' not in ag2_tf._next_node.util_info:
                    ag2_tf._next_node.util_info['qlk'] = dict()
                ag2_tf._next_node.util_info['qlk']['agent_2'] = resp_vect_sorted[0,:,:]['utils']
            else:
                ag2_tf._next_node.util_info = dict()
                ag2_tf._next_node.util_info['qlk'] = dict()
                ag2_tf._next_node.util_info['qlk']['agent_2'] = resp_vect_sorted[0,:,:]['utils']
                 
        resp_vect = np.array(ag_2_resp)
        resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
        ''' agent 2 is the ql0 agent, therefore this is its response based on maxmax behavior'''
        node.ql1_response['response']['agent_2'] =  np.copy(resp_vect_sorted[0,:,:])
        _denom = [x['utils'] for x in resp_vect_sorted]
        _denom = [np.exp(precision_parm*x) for x in _denom]
        _denom = sum(_denom)
        for r in resp_vect_sorted:
            _tl = r['traj_l'][0,0]
            _u = r['utils']
            _prob = np.divide(np.exp(precision_parm*_u),_denom)
            agent2_distr[_tl] = np.copy(_prob)[0,:]
        node.ql1_response['all_responses']['agent_2'] = np.array(ag_2_resp)   
        
        ag_1_resp = []
        for ag1_tf in veh_acts:
            ag_1_resp_per_ag2_act = []
            for ag2_tf in ped_acts:
                dist_gap = u.calc_dist_gap(veh_traj = ag1_tf.loaded_traj_frag, ped_traj = ag2_tf.loaded_traj_frag)
                safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
                step_util = u.combine_utils(u.progress_payoff_dist(ag1_tf.length, 'agent_1'), safe_payoff_for_dist, gamma_matrix[0])
                if node.level == last_decision_level:
                    ext_safe_utils = ag1_tf._next_node.mean_safe_util_contd_ag1
                    ext_prog_utils = ag1_tf._next_node.mean_progress_util_contd_ag1 if ag1_tf._next_node.mean_progress_util_contd_ag1 is not None else u.progress_payoff_dist(ag1_tf.length, 'agent_1')
                    cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[0])
                    _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                    safe_util = np.full(shape=gamma_matrix[0].shape, fill_value=_safe_m)
                    cont_util = step_util
                else:
                    if len(ag1_tf._next_node.children) == 0:
                        continue
                    else:
                        cont_util = ag1_tf._next_node.ql1_response['response']['agent_1']['utils']
                        safe_util = np.full(shape=gamma_matrix[0].shape, fill_value=safe_payoff_for_dist)
                _resp_entry = np.empty(shape= gamma_matrix[0].shape, dtype=np.record)
                _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                manv_str_arr = np.full(shape = gamma_matrix[0].shape, fill_value=ag1_tf.manv)
                traj_l_arr = np.full(shape = gamma_matrix[0].shape, fill_value = ag1_tf.length)
                ag2_util_entry_matrix = None 
                _prob = agent2_distr[ag2_tf.length]
                _exp_util = np.multiply(_prob, _util_entry_matrix)
                _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _exp_util, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
                ag_1_resp_per_ag2_act.append(_resp_entry)
            _sumutil = sum([x['utils'] for x in ag_1_resp_per_ag2_act])
            resp_vect = ag_1_resp_per_ag2_act[0]
            resp_vect['utils'] = _sumutil # This is the expected utility since the values have been weighted by probability of other agent's action earlier
            ag_1_resp.append(np.copy(resp_vect))
            if ag1_tf._next_node is not None:
                if hasattr(ag1_tf._next_node, 'util_info'):
                    if 'qlk' not in ag1_tf._next_node.util_info:
                        ag1_tf._next_node.util_info['qlk'] = dict()
                    ag1_tf._next_node.util_info['qlk']['agent_1'] = _sumutil
                else:
                    ag1_tf._next_node.util_info = dict()
                    ag1_tf._next_node.util_info['qlk'] = dict()
                    ag1_tf._next_node.util_info['qlk']['agent_1'] = _sumutil
        
        resp_vect = np.array(ag_1_resp)
        resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
        node.ql1_response['response']['agent_1'] =  np.copy(resp_vect_sorted[0,:,:])
        _denom = [x['utils'] for x in resp_vect_sorted]
        _denom = [np.exp(precision_parm*x) for x in _denom]
        _denom = sum(_denom)
        for r in resp_vect_sorted:
            _tl = r['traj_l'][0,0]
            _u = r['utils']
            _prob = np.divide(np.exp(precision_parm*_u),_denom)
            agent1_distr[_tl] = np.copy(_prob)[:,0]
        
        node.ql1_response['all_responses']['agent_1'] = np.array(ag_1_resp) 
        
        node.ql1_response['distribution']['agent_1'] = agent1_distr
        node.ql1_response['distribution']['agent_2'] = agent2_distr
        if not hasattr(node, 'all_ql1_resppnse'):
            node.all_ql1_resppnse = dict()
        if precision_parm not in node.all_ql1_resppnse:
            node.all_ql1_resppnse[precision_parm] = copy.deepcopy(node.ql1_response)
        f=1
            
        
class SatisficingEquilibria(Equilibria):
    
    def set_oneq_label(self,gt,eqtype_label):
        s1_time = int(1/gt.freq) if gt.freq == 0.5 else round(1/gt.freq,1)
        l2_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=s1_time)
        for i in np.arange(gt.root.equilibrium_solutions.shape[0]):
            for j in np.arange(gt.root.equilibrium_solutions.shape[1]):
                eq_2l = []
                for eq in gt.root.equilibrium_solutions[i,j]:
                    ag1_tl_range = (eq.veh_eq_acts[0], eq.veh_eq_acts[1]) if eqtype_label == 'on_mspe' else (eq.veh_eq_acts[0], eq.veh_eq_acts[2])
                    ag2_tl_range = (eq.peds_eq_acts[0], eq.peds_eq_acts[1]) if eqtype_label == 'on_mspe' else (eq.peds_eq_acts[0], eq.peds_eq_acts[2])
                    eq_2l += get_within_node(l2_nodes, ag1_tl_range, ag2_tl_range)
                for n2l in l2_nodes:
                    if not hasattr(n2l, eqtype_label):
                        if eqtype_label == 'on_mspe':
                            n2l.on_mspe = np.full(shape = gt.root.equilibrium_solutions.shape, fill_value=False)
                        else:
                            n2l.on_uspe = np.full(shape = gt.root.equilibrium_solutions.shape, fill_value=False)
                    if n2l._ext_id in eq_2l:
                        if eqtype_label == 'on_mspe':
                            n2l.on_mspe[i,j] = True
                        else:
                            n2l.on_uspe[i,j] = True
                        eq_4l = []
                        for eq in n2l.equilibrium_solutions[i,j]:
                            ag1_4ltl_range = (eq.veh_eq_acts[0], eq.veh_eq_acts[1]) if eqtype_label == 'on_mspe' else (eq.veh_eq_acts[0], eq.veh_eq_acts[2])
                            ag2_4ltl_range = (eq.peds_eq_acts[0], eq.peds_eq_acts[1]) if eqtype_label == 'on_mspe' else (eq.peds_eq_acts[0], eq.peds_eq_acts[2])
                            eq_4l += get_within_node(n2l.children, ag1_4ltl_range, ag2_4ltl_range)
                        for n4l in n2l.children:
                            if not hasattr(n4l, eqtype_label):
                                if eqtype_label == 'on_mspe':
                                    n4l.on_mspe = np.full(shape = gt.root.equilibrium_solutions.shape, fill_value=False)
                                else:
                                    n4l.on_uspe = np.full(shape = gt.root.equilibrium_solutions.shape, fill_value=False)
                            if n4l._ext_id in eq_4l:
                                if eqtype_label == 'on_mspe':
                                    n4l.on_mspe[i,j] = True
                                else:
                                    n4l.on_uspe[i,j] = True
                                eq_6l = []
                                for eq in n4l.equilibrium_solutions[i,j]:
                                    ag1_6ltl_range = (eq.veh_eq_acts[0], eq.veh_eq_acts[1]) if eqtype_label == 'on_mspe' else (eq.veh_eq_acts[0], eq.veh_eq_acts[2])
                                    ag2_6ltl_range = (eq.peds_eq_acts[0], eq.peds_eq_acts[1]) if eqtype_label == 'on_mspe' else (eq.peds_eq_acts[0], eq.peds_eq_acts[2])
                                    eq_6l += get_within_node(n4l.children, ag1_6ltl_range, ag2_6ltl_range)
                                for n6l in n4l.children:
                                    if not hasattr(n6l, eqtype_label):
                                        if eqtype_label == 'on_mspe':
                                            n6l.on_mspe = np.full(shape = gt.root.equilibrium_solutions.shape, fill_value=False)
                                        else:
                                            n6l.on_uspe = np.full(shape = gt.root.equilibrium_solutions.shape, fill_value=False)
                                        if n6l._ext_id in eq_6l:
                                            if eqtype_label == 'on_mspe':
                                                n6l.on_mspe[i,j] = True
                                            else:
                                                n6l.on_uspe[i,j] = True
                        
                        
                            
                            
        
    
    def calc_equilibria(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        
        type(node).progress_ctr += 1
        #print('processing node level',node.level,'id:',node._ext_id)
        print('solving node',type(node).progress_ctr,'/',type(node).tree_size)
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        gamma_matrix.reverse()
        ''' agent_1=0 agent_2 = 1'''
        #gamma_matrix = [np.linspace(start=-1, stop=1, num=20), np.linspace(start=-1, stop=1, num=20)]
        node.equilibrium_solutions = np.empty(shape= (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), dtype=object)
        if not hasattr(node, 'util_info'):
            node.util_info = dict()
        node.util_info['spe'] = dict()
        u = Utilities()
        veh_acts.sort(key=lambda x: x.length)
        ped_acts.sort(key=lambda x: x.length)
        ''' agent_2 best response to agent_1's trajectory length'''
        
        interac_dict = OrderedDict()
        for traj_frag in veh_acts:
            veh_traj_l = traj_frag.length
            if veh_traj_l not in interac_dict:
                interac_dict[veh_traj_l] = []
            interac_dict[veh_traj_l].append(traj_frag)
        
        peds_best_response = OrderedDict()
        ct,N = 0,len(interac_dict)
        ped_traj_l_list = set()
        for v_traj_l,traj_frag_list in interac_dict.items():
            
            ct += 1
            resp_vect = []
            ''' agent_1 can generate multiple distinct trajectories for the same trajectory length '''
            for veh_traj_frag in traj_frag_list:
                for peds_frag in ped_acts:
                    manv, manv_mode, ped_traj_l, dist_gap = peds_frag.manv, peds_frag.manv_mode, peds_frag.length, u.calc_dist_gap(veh_traj = veh_traj_frag.loaded_traj_frag, ped_traj = peds_frag.loaded_traj_frag)
                    ped_traj_l_list.add(ped_traj_l)
                    assert ped_traj_l >= 0
                    safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
                    step_util = u.combine_utils(u.progress_payoff_dist(ped_traj_l, 'agent_2'), safe_payoff_for_dist , gamma_matrix[1])
                    if node.level == last_decision_level:
                        ext_safe_utils = peds_frag._next_node.mean_safe_util_contd_ag2
                        ext_prog_utils = peds_frag._next_node.mean_progress_util_contd_ag2 if peds_frag._next_node.mean_progress_util_contd_ag2 is not None else u.progress_payoff_dist(ped_traj_l, 'agent_2')
                        cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[1])
                        _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
                    else:
                        f = np.vectorize(self._max_util)
                        cont_util = f(peds_frag._next_node.equilibrium_solutions,1)
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
                        #cont_util = cont_util.T
                        #cont_util = max([max(x.peds_eq_utils) for x in peds_frag._next_node.equilibrium_solutions])
                    _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
                    _util_entry_matrix = np.where(np.isnan(cont_util), cont_util, np.mean(np.array([ cont_util, step_util ]), axis=0 ))
                    node.util_info['spe'][(v_traj_l,ped_traj_l)] = [None,_util_entry_matrix]
                    if np.isnan(cont_util).all():
                        continue
                    #_util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                    
                    manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=manv)
                    traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ped_traj_l)
                    _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
                    resp_vect.append(_resp_entry)
                    
            resp_vect = np.array(resp_vect)
            try:
                resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            except ValueError:
                f=1
                raise
            upper_bound_matrix = np.copy(resp_vect_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], resp_vect_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(resp_vect_sorted)
            lower_bound_safety_matrix = np.copy(resp_vect_sorted)
            lower_safety_bound_matrix_check = resp_vect_sorted['safe_utils'] >= gamma_matrix[1]
            _x1 = resp_vect_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = resp_vect_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            if np.all(_x3):
                lower_bound_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(_x3)):
                lower_bound_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(_x3, axis=0) - 1
                lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            if np.all(lower_safety_bound_matrix_check):
                lower_bound_safety_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(lower_safety_bound_matrix_check)):
                lower_bound_safety_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(lower_safety_bound_matrix_check, axis=0) - 1
                lower_bound_safety_matrix = np.take_along_axis(lower_bound_safety_matrix,selected_indices[np.newaxis],axis=0)[0]
            
            upper_bound_matrix = upper_bound_matrix[0,:,:]
            
            #print('pedestrian responding',ct,'/',N,'to',v_traj_l)
            
            if v_traj_l not in peds_best_response:
                peds_best_response[v_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix), np.copy(lower_bound_safety_matrix))
            else:
                _merged_arr_ub = np.where(peds_best_response[v_traj_l][0]['utils'] > upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                _merged_arr_lb = np.where(peds_best_response[v_traj_l][1]['utils'] < lower_bound_matrix['utils'], peds_best_response[v_traj_l][1], lower_bound_matrix)
                _merged_arr_s_lb = np.where(peds_best_response[v_traj_l][1]['utils'] < lower_bound_safety_matrix['utils'], peds_best_response[v_traj_l][1], lower_bound_safety_matrix)
                peds_best_response[v_traj_l] = (_merged_arr_ub, _merged_arr_lb, _merged_arr_s_lb)
                
                
            
            
        #X_lb_ub.sort(key=lambda tup: tup[0])
        
        f=1
        '''
        if len(peds_best_response) < 2:
            node.equilibrium_solutions = None
            return None
        '''
        ''' agent_2 best response function'''
        p2_matrix = np.empty(shape= gamma_matrix[1].shape[0], dtype=object)
        for i in np.arange(p2_matrix.shape[0]):
            if np.nan in [x[0][0,i]['utils'] for x in peds_best_response.values()] or len(peds_best_response.values()) == 0:
                p2_matrix[i] = None
            elif len(peds_best_response.values()) == 1:
                p2 = (point.Point(list(zip([x[0][0,i]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))) , point.Point(list(zip([x[1][0,i]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))))
                p2_matrix[i] = p2
            else:
                p2 = (LineString(list(zip([x[0][0,i]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))) , LineString(list(zip([x[1][0,i]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))))
                p2_matrix[i] = p2  
            
                
        '''
        if show_plots:
            plt.figure()
            plt.plot([x for x in peds_best_response.keys()],[x[0][0]['traj_l'] for x in peds_best_response.values()],c='red')
            plt.plot([x for x in peds_best_response.keys()],[x[1][0]['traj_l'] for x in peds_best_response.values()],c=lighten_color('red', .5),label = 'agent_2 best response')
        '''
                
                
        
        
        
        
        ''' agent_1 best reponse to agent_2 trajectory choice'''
        interac_dict = OrderedDict()
        for traj_frag in ped_acts:
            ped_traj_l = traj_frag.length
            if ped_traj_l not in interac_dict:
                interac_dict[ped_traj_l] = []
            interac_dict[ped_traj_l].append(traj_frag)
        
        veh_best_response = OrderedDict()
        ct,N = 0,len(interac_dict)
        veh_traj_l_list = set()
        for p_traj_l,traj_frag_list in interac_dict.items():
            
            ct += 1
            resp_vect = []
            ''' agent_2 can generate multiple distinct trajectories for the same trajectory length '''
            for ped_traj_frag in traj_frag_list:
                for veh_frag in veh_acts:
                    manv, manv_mode, veh_traj_l, dist_gap = veh_frag.manv, veh_frag.manv_mode, veh_frag.length, u.calc_dist_gap(veh_traj = veh_frag.loaded_traj_frag, ped_traj = ped_traj_frag.loaded_traj_frag)
                    veh_traj_l_list.add(veh_traj_l)
                    assert veh_traj_l >= 0
                    safe_payoff_for_dist = u.calc_safe_payoff(dist_gap)
                    step_util = u.combine_utils(u.progress_payoff_dist(veh_traj_l, 'agent_1'), safe_payoff_for_dist, gamma_matrix[0])
                    if node.level == last_decision_level:
                        ext_safe_utils = veh_frag._next_node.mean_safe_util_contd_ag1
                        ext_prog_utils = veh_frag._next_node.mean_progress_util_contd_ag1 if veh_frag._next_node.mean_progress_util_contd_ag1 is not None else u.progress_payoff_dist(veh_traj_l, 'agent_1')
                        cont_util = u.combine_utils(ext_prog_utils, ext_safe_utils, gamma_matrix[0])
                        _safe_m = np.mean([ext_safe_utils, safe_payoff_for_dist])
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=_safe_m)
                    else:
                        f = np.vectorize(self._max_util)
                        cont_util = f(veh_frag._next_node.equilibrium_solutions,0)
                        safe_util = np.full(shape=gamma_matrix[1].shape, fill_value=safe_payoff_for_dist)
                        #cont_util = max([max(x.veh_eq_utils) for x in veh_frag._next_node.equilibrium_solutions])
                    _resp_entry = np.empty(shape= gamma_matrix[0].shape, dtype=object)
                    _util_entry_matrix = np.where(np.isnan(cont_util), cont_util, np.mean(np.array([ cont_util, step_util ]), axis=0 ))
                    if np.isnan(cont_util).all():
                        continue
                    
                    #_util_entry_matrix = np.mean( np.array([ cont_util, step_util ]), axis = 0)
                    node.util_info['spe'][(veh_traj_l,p_traj_l)][0] = _util_entry_matrix
                    manv_str_arr = np.full(shape = gamma_matrix[0].shape, fill_value=manv)
                    traj_l_arr = np.full(shape = gamma_matrix[0].shape, fill_value = veh_traj_l)
                    _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix, safe_util), names=('manv', 'traj_l', 'utils', 'safe_utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float), ('safe_utils', float)])
                    
                    resp_vect.append(_resp_entry)
                    
            resp_vect = np.array(resp_vect)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            upper_bound_matrix = np.copy(resp_vect_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], resp_vect_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(resp_vect_sorted)
            lower_bound_safety_matrix = np.copy(resp_vect_sorted)
            lower_safety_bound_matrix_check = resp_vect_sorted['safe_utils'] >= gamma_matrix[0]
            _x1 = resp_vect_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = resp_vect_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            if np.all(_x3):
                lower_bound_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(_x3)):
                lower_bound_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(_x3, axis=0) - 1
                lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            
            if np.all(lower_safety_bound_matrix_check):
                lower_bound_safety_matrix = upper_bound_matrix[-1,:,:]
            elif np.all(np.logical_not(lower_safety_bound_matrix_check)):
                lower_bound_safety_matrix = upper_bound_matrix[0,:,:]
            else:
                selected_indices = np.argmin(lower_safety_bound_matrix_check, axis=0) - 1
                lower_bound_safety_matrix = np.take_along_axis(lower_bound_safety_matrix,selected_indices[np.newaxis],axis=0)[0]
            
            upper_bound_matrix = upper_bound_matrix[0,:,:]
            
            #print('vehicle responding',ct,'/',N,'to',p_traj_l)
            if p_traj_l not in veh_best_response:
                    veh_best_response[p_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix), np.copy(lower_bound_safety_matrix))
            else:
                _merged_arr_ub = np.where(veh_best_response[p_traj_l][0]['utils'] > upper_bound_matrix['utils'], veh_best_response[p_traj_l][0], upper_bound_matrix)
                _merged_arr_lb = np.where(veh_best_response[p_traj_l][1]['utils'] < lower_bound_matrix['utils'], veh_best_response[p_traj_l][1], lower_bound_matrix)
                _merged_arr_s_lb = np.where(veh_best_response[p_traj_l][1]['utils'] < lower_bound_safety_matrix['utils'], veh_best_response[p_traj_l][1], lower_bound_safety_matrix)
                veh_best_response[p_traj_l] = (_merged_arr_ub, _merged_arr_lb, _merged_arr_s_lb)
                
        '''
        if len(veh_best_response) < 2:
            node.equilibrium_solutions = None
            return None
        '''
        ''' agent_1 best response function'''
        p1_matrix = np.empty(shape= gamma_matrix[0].shape[0], dtype=object)
        for i in np.arange(p1_matrix.shape[0]):
            if np.nan in [x[0][i,0]['utils'] for x in veh_best_response.values()] or len(veh_best_response.values()) == 0:
                p1_matrix[i] = None
            elif len(veh_best_response.values()) == 1:
                p1 = (point.Point(list(zip([x for x in veh_best_response.keys()],[x[0][i,0]['traj_l'] for x in veh_best_response.values()]))) , point.Point(list(zip([x for x in veh_best_response.keys()],[x[1][i,0]['traj_l'] for x in veh_best_response.values()])))) 
                p1_matrix[i] = p1
            else:
                p1 = (LineString(list(zip([x for x in veh_best_response.keys()],[x[0][i,0]['traj_l'] for x in veh_best_response.values()]))) , LineString(list(zip([x for x in veh_best_response.keys()],[x[1][i,0]['traj_l'] for x in veh_best_response.values()]))))
                p1_matrix[i] = p1  
        '''
        if show_plots:
            plt.plot([x[0][0]['traj_l'] for x in veh_best_response.values()],[x for x in veh_best_response.keys()],c='blue')
            plt.plot([x[1][0]['traj_l'] for x in veh_best_response.values()],[x for x in veh_best_response.keys()],c=lighten_color('blue', .5),label = 'agent_1 best response')
        '''
        if len(node.children) > 9:
            if show_plots:
                fig, axs = plt.subplots(5, 5)
        
        
        self.veh_best_response = veh_best_response
        self.peds_best_response= peds_best_response
        
        '''
        p2:
            x : agent_1's trajectory length choice
            y : agent_2 best response
            
        p1:
            x : agent_1 best reponse 
            y : agent_2's trajectory length choice
        
        p[0]: upper bound
        p[1]: lower bound
        '''
        for i in np.arange(p1_matrix.shape[0]):
            for j in np.arange(p2_matrix.shape[0]):
                p2,p1 = p2_matrix[j], p1_matrix[i]
                if p2 is None or p1 is None:
                    node.equilibrium_solutions[i,j] = None
                    continue
                eq_strat = None
                eq_obj = None
                ''' find the equilibrium with respect to upper bounds 
                    and expand the points based on best response curves '''
                if p2[0].intersects(p1[0]):
                    eq_obj = p2[0].intersection(p1[0])
                elif len(p1) > 1 and p2[0].intersects(p1[1]):
                    eq_obj = p2[0].intersection(p1[1])
                elif len(p2) > 1 and p2[1].intersects(p1[0]):
                    eq_obj = p2[1].intersection(p1[0])
                elif len(p2) > 1 and len(p1) > 1 and p2[1].intersects(p1[1]):
                    eq_obj = p2[1].intersection(p1[1])
                else:
                    pass
                    #print('Equilibrium doesn\'t exists',gamma_matrix[0][i],gamma_matrix[1][j])
                if eq_obj is not None:
                    if isinstance(eq_obj, multipoint.MultiPoint):
                        eq_strat = self._process_MultiPoint(eq_obj,(i,j))    
                        
                    elif isinstance(eq_obj, point.Point):
                        eq_strat = self._process_Point(eq_obj,(i,j))
                        
                    elif isinstance(eq_obj, linestring.LineString):
                        eq_strat = self._process_LineString(eq_obj,(i,j))
                        
                    elif isinstance(eq_obj, multilinestring.MultiLineString):
                        eq_strat = []
                        for eq_item in eq_obj:
                            eq_strat += self._process_LineString(eq_item,(i,j))
                        
                    elif isinstance(eq_obj, GeometryCollection):
                        eq_strat = []
                        for eq_item in eq_obj:
                            if isinstance(eq_item, multipoint.MultiPoint):
                                eq_strat += self._process_MultiPoint(eq_item,(i,j))    
                            elif isinstance(eq_item, point.Point):
                                eq_strat += self._process_Point(eq_item,(i,j))
                            elif isinstance(eq_item, linestring.LineString):
                                eq_strat += self._process_LineString(eq_item,(i,j))
                            else:
                                raise Exception('cannot process equilibrium of class '+type(eq_obj).__name__)
                        #print(node.level,'equilibrium',eq_strat)
                    else:
                        raise Exception('cannot process equilibrium of class '+type(eq_obj).__name__)
                #eq_reg = p2.intersection(p1)
                #print(eq_reg)
                if len(node.children) > 9:
                    if show_plots:
                        minx,maxx,miny,maxy = np.inf,-np.inf,np.inf,-np.inf
                        x, y = p2[0].xy
                        if min(x) < minx:
                            minx = min(x)
                        if max(x) > maxx:
                            maxx = max(x)
                        if min(y) < miny:
                            miny = min(y)
                        if max(y) > maxy:
                            maxy = max(y)
                        axs[i,j].plot(x, y, color='red')
                        x, y = p2[1].xy
                        if min(x) < minx:
                            minx = min(x)
                        if max(x) > maxx:
                            maxx = max(x)
                        if min(y) < miny:
                            miny = min(y)
                        if max(y) > maxy:
                            maxy = max(y)
                        
                        axs[i,j].plot(x, y, color=lighten_color('red', 0.5))
                        x, y = p1[0].xy
                        if min(x) < minx:
                            minx = min(x)
                        if max(x) > maxx:
                            maxx = max(x)
                        if min(y) < miny:
                            miny = min(y)
                        if max(y) > maxy:
                            maxy = max(y)
                        
                        axs[i,j].plot(x, y, color='blue')
                        x, y = p1[1].xy
                        if min(x) < minx:
                            minx = min(x)
                        if max(x) > maxx:
                            maxx = max(x)
                        if min(y) < miny:
                            miny = min(y)
                        if max(y) > maxy:
                            maxy = max(y)
                        
                        axs[i,j].plot(x, y, color=lighten_color('blue', 0.5))
                        axs[i,j].set_xlim(minx-1,maxx+1)
                        axs[i,j].set_ylim(miny-1,maxy+1)
                        '''
                        eq_reg = eq_obj
                        for ob in eq_reg:
                            x, y = ob.xy
                            if len(x) == 1:
                                ax.plot(x, y, 'o', color='BLUE', zorder=2)
                            else:
                                ax.plot(x, y, color='green', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
                        '''
                if eq_strat is not None:
                    eq_solns = []
                    for eq_item in eq_strat:
                        veh_eq_resp_ub,veh_eq_resp_lb,veh_eq_resp_lb_uspe = eq_item[0][0], eq_item[0][1], eq_item[0][2]
                        peds_eq_resp_ub,peds_eq_resp_lb,peds_eq_resp_lb_uspe = eq_item[1][0], eq_item[1][1], eq_item[1][2]
                        veh_eq_act = (veh_eq_resp_ub[1],veh_eq_resp_lb[1],veh_eq_resp_lb_uspe[1])
                        veh_eq_utils = (veh_eq_resp_ub[2],veh_eq_resp_lb[2],veh_eq_resp_lb_uspe[2])
                        peds_eq_act = (peds_eq_resp_ub[1],peds_eq_resp_lb[1],peds_eq_resp_lb_uspe[1])
                        peds_eq_utils = (peds_eq_resp_ub[2],peds_eq_resp_lb[2],peds_eq_resp_lb_uspe[2])
                        eqsoln_obj = EquilibriaSolution(veh_best_response,peds_best_response)
                        eqsoln_obj.set_veh_acts(veh_eq_act)
                        eqsoln_obj.set_peds_acts(peds_eq_act)
                        eqsoln_obj.set_veh_utils(veh_eq_utils)
                        eqsoln_obj.set_peds_utils(peds_eq_utils)
                        eq_solns.append(eqsoln_obj)
                    for ag,actions in node.actions.items():
                        for act in actions:
                            act_length = act.length
                            '''
                            if not hasattr(act._next_node, 'on_mspe_eq'):
                                act._next_node.on_mspe_eq = {'agent_1':np.full(shape = (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), fill_value=False), 'agent_2':np.full(shape = (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), fill_value=False)}
                            if ag == 'agent_1':
                                act._next_node.on_mspe_eq[ag][i,j] = any([min(x.veh_eq_acts) <= act_length <= max(x.veh_eq_acts) for x in eq_solns])
                            else:
                                act._next_node.on_mspe_eq[ag][i,j] = any([min(x.peds_eq_acts) <= act_length <= max(x.peds_eq_acts) for x in eq_solns])
                            '''
                    node.equilibrium_solutions[i,j] = eq_solns
                    #print(node.level,'equilibrium',[x.veh_eq_acts for x in eq_solns], [x.peds_eq_acts for x in eq_solns])
                else:
                    node.equilibrium_solutions[i,j] = None
        if len(node.children) > 9: 
            if show_plots:
                plt.show()
        if len(node.children) > 9:
            f=1
                
        auto_resp = AutoStrategyResponse(self.run_context) 
        auto_resp.calc_response(veh_acts, ped_acts, node, last_decision_level)       
          
        return node.equilibrium_solutions
        
            