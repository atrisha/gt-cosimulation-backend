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
from equilibrium.gametree_objects import TrajectoryFragment
from typing import List
from code_utils import utils
from shapely.geometry import multipoint, point, linestring, multilinestring, GeometryCollection
from statistics import mean
from collections import OrderedDict
from code_utils.code_util_objects import RunContext


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
            peds_actions = node.actions['agent_2']
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
    
    def expand_equil(self,eq_strat,veh_br_map, peds_br_map, belief_index):
        veh_eq_act = eq_strat.x
        peds_eq_act = eq_strat.y
        i,j = belief_index[0], belief_index[1]
        veh_br_key = min(veh_br_map.keys(), key=lambda x:abs(x-peds_eq_act))
        veh_br_range = (veh_br_map[veh_br_key][0][i,j], veh_br_map[veh_br_key][1][i,j])
        peds_br_key = min(peds_br_map.keys(), key=lambda x:abs(x-veh_eq_act))
        ''' for agent_2, the indexes should be flipped since i is always agent_1, and agent_2 br matrix had i as agent_2 threshold'''
        ''' on second thought, they need not be flipped, because peds_br_map has the entries based on the gamma matrix[1]'''
        peds_br_range = (peds_br_map[peds_br_key][0][i,j], peds_br_map[peds_br_key][1][i,j])
        return (veh_br_range,peds_br_range)
    
    def __init__(self,run_context = None):
        if run_context is None:
            self.run_context = RunContext()
        else:
            self.run_context = run_context
                
class AutoStrategyResponse(Equilibria):
    
    def calc_response(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        self.gamma_matrix = gamma_matrix
        ''' agent_1=0 agent_2 = 1'''
        #gamma_matrix = [np.linspace(start=-1, stop=1, num=20), np.linspace(start=-1, stop=1, num=20)]
        u = Utilities()
        veh_acts.sort(key=lambda x: x.length)
        ped_acts.sort(key=lambda x: x.length)
        ''' agent_2 best response to agent_1's trajectory length'''
        node.auto_strategy_response = dict()
        interac_dict = OrderedDict()
        ag_2_resp = []
        for ag1_tf in veh_acts:
            for ag2_tf in ped_acts:
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
                step_util = u.combine_utils(u.progress_payoff_dist(ag2_tf.length, 'agent_2'), u.calc_safe_payoff(dist_gap), gamma_matrix[1])
                if node.level == last_decision_level:
                    cont_util = np.copy(step_util)
                else:
                    f = np.vectorize(self._max_util)
                    cont_util = f(ag2_tf._next_node.equilibrium_solutions,1)
                    #cont_util = cont_util.T
                    #cont_util = max([max(x.peds_eq_utils) for x in peds_frag._next_node.equilibrium_solutions])
                _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
                if np.isnan(cont_util).any():
                    continue
                _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=ag2_tf.manv)
                traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ag2_tf.length)
                _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix), names=('manv', 'traj_l', 'utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float)])
                ag_2_resp.append(_resp_entry)
        if len(ag_2_resp) == 0:
            self.agent_2_auto_strategy_response = None
        else:
            ag_2_resp = np.array(ag_2_resp)
            ag_2_resp_sorted = np.sort(ag_2_resp,axis=0,order='utils')[::-1]
            upper_bound_matrix = np.copy(ag_2_resp_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], ag_2_resp_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(ag_2_resp_sorted)
            _x1 = ag_2_resp_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = ag_2_resp_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            selected_indices = np.argmin(_x3, axis=0) - 1
            ag2_lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            ag2_upper_bound_matrix = upper_bound_matrix[0,:,:]
            
            node.auto_strategy_response['agent_2'] = (ag2_upper_bound_matrix,ag2_lower_bound_matrix)
        
        ag_1_resp = []
        for ag2_tf in ped_acts:
            for ag1_tf in veh_acts:
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
                step_util = u.combine_utils(u.progress_payoff_dist(ag1_tf.length, 'agent_1'), u.calc_safe_payoff(dist_gap), gamma_matrix[0])
                if node.level == last_decision_level:
                    cont_util = np.copy(step_util)
                else:
                    f = np.vectorize(self._max_util)
                    cont_util = f(ag1_tf._next_node.equilibrium_solutions,0)
                    #cont_util = max([max(x.peds_eq_utils) for x in peds_frag._next_node.equilibrium_solutions])
                _resp_entry = np.empty(shape= gamma_matrix[0].shape, dtype=np.record)
                if np.isnan(cont_util).any():
                    continue
                _util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                manv_str_arr = np.full(shape = gamma_matrix[0].shape, fill_value=ag1_tf.manv)
                traj_l_arr = np.full(shape = gamma_matrix[0].shape, fill_value = ag1_tf.length)
                _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix), names=('manv', 'traj_l', 'utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float)])
                ag_1_resp.append(_resp_entry)
        if len(ag_1_resp) == 0:
            self.agent_1_auto_strategy_response = None
        else:
            ag_1_resp = np.array(ag_1_resp)
            ag_1_resp_sorted = np.sort(ag_1_resp,axis=0,order='utils')[::-1]               
            upper_bound_matrix = np.copy(ag_1_resp_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], ag_1_resp_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(ag_1_resp_sorted)
            _x1 = ag_1_resp_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = ag_1_resp_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            selected_indices = np.argmin(_x3, axis=0) - 1
            ag1_lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            ag1_upper_bound_matrix = upper_bound_matrix[0,:,:]
                         
            node.auto_strategy_response['agent_1'] = (ag1_upper_bound_matrix,ag1_lower_bound_matrix)
        
    
class RobustResponse(Equilibria):
    
    def calc_response(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        ''' agent_1=0 agent_2 = 1'''
        node.robust_response = {'agent_1' : np.empty(shape= (gamma_matrix[0].shape[0],1), dtype=object),
                                'agent_2' : np.empty(shape= (gamma_matrix[1].shape[1],1), dtype=object)}
        for i in np.arange(node.robust_response['agent_1'].shape[0]):
            ''' agent_1 private tolerance type is i '''
            all_eq = node.equilibrium_solutions[i,:]
            ''' this is just a response, so te best response function can be None'''
            soln = EquilibriaSolution(None,None)
            min_util = min([eq.veh_eq_utils[0] for eq in all_eq])
            min_idx = [eq.veh_eq_utils[0] for eq in all_eq].index(min_util)
            ''' just take the first one because for automata strategy response, all columns are same '''
            all_auto_resps = (node.auto_strategy_response['agent_1'][0][i,0], node.auto_strategy_response['agent_1'][1][i,0])
            robust_resp_to_mspe = all_eq[i,min_idx]
            f=1
   
        
class SatisficingEquilibria(Equilibria):
    
    
    def calc_equilibria(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node, last_decision_level):
        type(node).progress_ctr += 1
        #print('processing node level',node.level,'id:',node._ext_id)
        print('solving node',type(node).progress_ctr,'/',type(node).tree_size)
        gamma_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=5), np.linspace(start=-1, stop=1, num=5))
        ''' agent_1=0 agent_2 = 1'''
        #gamma_matrix = [np.linspace(start=-1, stop=1, num=20), np.linspace(start=-1, stop=1, num=20)]
        node.equilibrium_solutions = np.empty(shape= (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), dtype=object)
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
                    step_util = u.combine_utils(u.progress_payoff_dist(ped_traj_l, 'agent_2'), u.calc_safe_payoff(dist_gap), gamma_matrix[1])
                    if node.level == last_decision_level:
                        cont_util = np.copy(step_util)
                    else:
                        f = np.vectorize(self._max_util)
                        cont_util = f(peds_frag._next_node.equilibrium_solutions,1)
                        #cont_util = cont_util.T
                        #cont_util = max([max(x.peds_eq_utils) for x in peds_frag._next_node.equilibrium_solutions])
                    _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=np.record)
                    _util_entry_matrix = np.where(np.isnan(cont_util), cont_util, np.mean(np.array([ cont_util, step_util ]), axis=0 ))
                    #if np.isnan(cont_util).all():
                    #    continue
                    #_util_entry_matrix = np.mean(np.array([ cont_util, step_util ]), axis=0 )
                    manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=manv)
                    traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ped_traj_l)
                    _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix), names=('manv', 'traj_l', 'utils'), dtype=[('manv', object), ('traj_l', float), ('utils', float)])
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
            _x1 = resp_vect_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = resp_vect_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            selected_indices = np.argmin(_x3, axis=0) - 1
            lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            upper_bound_matrix = upper_bound_matrix[0,:,:]
            
            #print('pedestrian responding',ct,'/',N,'to',v_traj_l)
            
            if v_traj_l not in peds_best_response:
                peds_best_response[v_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix))
            else:
                _merged_arr_ub = np.where(peds_best_response[v_traj_l][0]['utils'] > upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                _merged_arr_lb = np.where(peds_best_response[v_traj_l][1]['utils'] < lower_bound_matrix['utils'], peds_best_response[v_traj_l][1], lower_bound_matrix)
                peds_best_response[v_traj_l] = (_merged_arr_ub, _merged_arr_lb)
                
                
            
            
        #X_lb_ub.sort(key=lambda tup: tup[0])
        
        f=1
        '''
        if len(peds_best_response) < 2:
            node.equilibrium_solutions = None
            return None
        '''
        ''' agent_2 best response function'''
        p1_matrix = np.empty(shape= gamma_matrix[1].shape[0], dtype=object)
        for i in np.arange(p1_matrix.shape[0]):
            if np.nan in [x[0][i,0]['utils'] for x in peds_best_response.values()] or len(peds_best_response.values()) == 0:
                p1_matrix[i] = None
            elif len(peds_best_response.values()) == 1:
                p1 = (point.Point(list(zip([x[0][i,0]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))) , point.Point(list(zip([x[1][i,0]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))))
                p1_matrix[i] = p1
            else:
                p1 = (LineString(list(zip([x[0][i,0]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))) , LineString(list(zip([x[1][i,0]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))))
                p1_matrix[i] = p1  
            
                
        
        if show_plots:
            plt.figure()
            plt.plot([x for x in peds_best_response.keys()],[x[0][0]['traj_l'] for x in peds_best_response.values()],c='red')
            plt.plot([x for x in peds_best_response.keys()],[x[1][0]['traj_l'] for x in peds_best_response.values()],c=lighten_color('red', .5),label = 'agent_2 best response')
        
                
                
        
        
        
        
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
                    step_util = u.combine_utils(u.progress_payoff_dist(veh_traj_l, 'agent_1'), u.calc_safe_payoff(dist_gap), gamma_matrix[0])
                    if node.level == last_decision_level:
                        cont_util = np.copy(step_util)
                    else:
                        f = np.vectorize(self._max_util)
                        cont_util = f(veh_frag._next_node.equilibrium_solutions,0)
                        #cont_util = max([max(x.veh_eq_utils) for x in veh_frag._next_node.equilibrium_solutions])
                    _resp_entry = np.empty(shape= gamma_matrix[0].shape, dtype=object)
                    _util_entry_matrix = np.where(np.isnan(cont_util), cont_util, np.mean(np.array([ cont_util, step_util ]), axis=0 ))
                    #_util_entry_matrix = np.mean( np.array([ cont_util, step_util ]), axis = 0)
                    manv_str_arr = np.full(shape = gamma_matrix[0].shape, fill_value=manv)
                    traj_l_arr = np.full(shape = gamma_matrix[0].shape, fill_value = veh_traj_l)
                    _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix), names=('manv', 'traj_l', 'utils'))
                    
                    resp_vect.append(_resp_entry)
                    
            resp_vect = np.array(resp_vect)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            upper_bound_matrix = np.copy(resp_vect_sorted[0,:,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:,:], resp_vect_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(resp_vect_sorted)
            _x1 = resp_vect_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = resp_vect_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            selected_indices = np.argmin(_x3, axis=0) - 1
            lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            upper_bound_matrix = upper_bound_matrix[0,:,:]
            
            #print('vehicle responding',ct,'/',N,'to',p_traj_l)
            if p_traj_l not in veh_best_response:
                    veh_best_response[p_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix))
            else:
                _merged_arr_ub = np.where(veh_best_response[p_traj_l][0]['utils'] > upper_bound_matrix['utils'], veh_best_response[p_traj_l][0], upper_bound_matrix)
                _merged_arr_lb = np.where(veh_best_response[p_traj_l][1]['utils'] < lower_bound_matrix['utils'], veh_best_response[p_traj_l][1], lower_bound_matrix)
                veh_best_response[p_traj_l] = (_merged_arr_ub, _merged_arr_lb)
        '''
        if len(veh_best_response) < 2:
            node.equilibrium_solutions = None
            return None
        '''
        ''' agent_1 best response function'''
        p2_matrix = np.empty(shape= gamma_matrix[0].shape[0], dtype=object)
        for i in np.arange(p2_matrix.shape[0]):
            if np.nan in [x[0][i,0]['utils'] for x in veh_best_response.values()] or len(veh_best_response.values()) == 0:
                p2_matrix[i] = None
            elif len(veh_best_response.values()) == 1:
                p2 = (point.Point(list(zip([x for x in veh_best_response.keys()],[x[0][i,0]['traj_l'] for x in veh_best_response.values()]))) , point.Point(list(zip([x for x in veh_best_response.keys()],[x[1][i,0]['traj_l'] for x in veh_best_response.values()])))) 
                p2_matrix[i] = p2
            else:
                p2 = (LineString(list(zip([x for x in veh_best_response.keys()],[x[0][i,0]['traj_l'] for x in veh_best_response.values()]))) , LineString(list(zip([x for x in veh_best_response.keys()],[x[1][i,0]['traj_l'] for x in veh_best_response.values()]))))
                p2_matrix[i] = p2  
        
        if show_plots:
            plt.plot([x[0][0]['traj_l'] for x in veh_best_response.values()],[x for x in veh_best_response.keys()],c='blue')
            plt.plot([x[1][0]['traj_l'] for x in veh_best_response.values()],[x for x in veh_best_response.keys()],c=lighten_color('blue', .5),label = 'agent_1 best response')
        
        
        
        self.veh_best_response = veh_best_response
        self.peds_best_response= peds_best_response
        
        '''
        p1:
            x : agent_1's trajectory length choice
            y : agent_2 best response
            
        p2:
            x : agent_1 best reponse 
            y : agent_2's trajectory length choice
        
        p[0]: upper bound
        p[1]: lower bound
        '''
        for i in np.arange(p2_matrix.shape[0]):
            for j in np.arange(p1_matrix.shape[0]):
                p1,p2 = p1_matrix[j], p2_matrix[i]
                if p1 is None or p2 is None:
                    node.equilibrium_solutions[i,j] = None
                    continue
                eq_strat = None
                eq_obj = None
                ''' find the equilibrium with respect to upper bounds 
                    and expand the points based on best response curves '''
                if p1[0].intersects(p2[0]):
                    eq_obj = p1[0].intersection(p2[0])
                elif len(p2) > 1 and p1[0].intersects(p2[1]):
                    eq_obj = p1[0].intersection(p2[1])
                elif len(p1) > 1 and p1[1].intersects(p2[0]):
                    eq_obj = p1[1].intersection(p2[0])
                elif len(p1) > 1 and len(p2) > 1 and p1[1].intersects(p2[1]):
                    eq_obj = p1[1].intersection(p2[1])
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
                        
                    elif isinstance(eq_obj, multilinestring.MultiLineString,(i,j)):
                        eq_strat = []
                        for eq_item in eq_obj:
                            eq_strat += self._process_LineString(eq_item)
                        
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
                #eq_reg = p1.intersection(p2)
                #print(eq_reg)
                '''
                fig = plt.figure()
                ax = fig.add_subplot(121)
                
                for ob in eq_reg:
                    x, y = ob.xy
                    if len(x) == 1:
                        ax.plot(x, y, 'o', color='BLUE', zorder=2)
                    else:
                        ax.plot(x, y, color='BLUE', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
                '''
                if show_plots:
                    plt.show()
                if eq_strat is not None:
                    eq_solns = []
                    for eq_item in eq_strat:
                        veh_eq_resp_ub,veh_eq_resp_lb = eq_item[0][0], eq_item[0][1]
                        peds_eq_resp_ub,peds_eq_resp_lb = eq_item[1][0], eq_item[1][1]
                        veh_eq_act = (veh_eq_resp_ub[1],veh_eq_resp_lb[1])
                        veh_eq_utils = (veh_eq_resp_ub[2],veh_eq_resp_lb[2])
                        peds_eq_act = (peds_eq_resp_ub[1],peds_eq_resp_lb[1])
                        peds_eq_utils = (peds_eq_resp_ub[2],peds_eq_resp_lb[2])
                        eqsoln_obj = EquilibriaSolution(veh_best_response,peds_best_response)
                        eqsoln_obj.set_veh_acts(veh_eq_act)
                        eqsoln_obj.set_peds_acts(peds_eq_act)
                        eqsoln_obj.set_veh_utils(veh_eq_utils)
                        eqsoln_obj.set_peds_utils(peds_eq_utils)
                        eq_solns.append(eqsoln_obj)
                    for ag,actions in node.actions.items():
                        for act in actions:
                            act_length = act.length
                            if not hasattr(act._next_node, 'on_mspe_eq'):
                                act._next_node.on_mspe_eq = {'agent_1':np.full(shape = (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), fill_value=False), 'agent_2':np.full(shape = (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), fill_value=False)}
                            if ag == 'agent_1':
                                act._next_node.on_mspe_eq[ag][i,j] = any([min(x.veh_eq_acts) <= act_length <= max(x.veh_eq_acts) for x in eq_solns])
                            else:
                                act._next_node.on_mspe_eq[ag][i,j] = any([min(x.peds_eq_acts) <= act_length <= max(x.peds_eq_acts) for x in eq_solns])
                            
                    node.equilibrium_solutions[i,j] = eq_solns
                    #print(node.level,'equilibrium',[x.veh_eq_acts for x in eq_solns], [x.peds_eq_acts for x in eq_solns])
                else:
                    node.equilibrium_solutions[i,j] = None
        
        auto_resp = AutoStrategyResponse(self.run_context) 
        auto_resp.calc_response(veh_acts, ped_acts, node, last_decision_level)        
                
           
        return node.equilibrium_solutions
        
            