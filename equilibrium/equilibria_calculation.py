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
from planners.planning_objects import TrajectoryFragment
from typing import List
from code_utils import utils
from shapely.geometry import multipoint, point, linestring, multilinestring, GeometryCollection
from statistics import mean
from collections import OrderedDict


show_plots = True


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
        

class SatisficingEquilibria:
        
    
    '''
        This takes in a key value pair of (m,t_idx) : u_i for a give -i action.
        Generates the best response set modulo manuver.
        Also generates the utility interval for the best response set.
    '''
    def solve(self,node):
        if node.level == 4:
            peds_actions = node.actions['pedestrian']
            veh_actions = node.actions['vehicle']
            all_strategies = itertools.product(veh_actions,peds_actions)
            self.calc_equilibria(veh_actions,peds_actions,node)
        else:
            if not node.is_leaf: 
                for c in node.children:
                    self.solve(c)
                peds_actions = node.actions['pedestrian']
                veh_actions = node.actions['vehicle']
                all_strategies = itertools.product(veh_actions,peds_actions)
                self.calc_equilibria(veh_actions,peds_actions,node)
                
    def _process_LineString(self, eq_obj):
        equil = []
        for eq_pt in eq_obj.coords:
            eq_strat = self.expand_equil(eq_strat = point.Point(eq_pt), veh_br_map = self.veh_best_response, peds_br_map = self.peds_best_response)
            equil.append(eq_strat)
        return equil
    
    def _process_Point(self, eq_obj):
        equil = []
        eq_strat = self.expand_equil(eq_strat = eq_obj, veh_br_map = self.veh_best_response, peds_br_map = self.peds_best_response)
        equil.append(eq_strat)
        return equil
    
    def _process_MultiPoint(self, eq_obj):
        equil = []
        for eq_pt in eq_obj:
            eq_strat = self.expand_equil(eq_strat = eq_pt, veh_br_map = self.veh_best_response, peds_br_map = self.peds_best_response)
            equil.append(eq_strat)
        return equil
    
    def expand_equil(self,eq_strat,veh_br_map, peds_br_map):
        veh_eq_act = eq_strat.x
        peds_eq_act = eq_strat.y
        veh_br_key = min(veh_br_map.keys(), key=lambda x:abs(x-peds_eq_act))
        veh_br_range = veh_br_map[veh_br_key]
        peds_br_key = min(peds_br_map.keys(), key=lambda x:abs(x-veh_eq_act))
        peds_br_range = peds_br_map[peds_br_key]
        return (veh_br_range,peds_br_range)
                
    def calc_equilibria(self,veh_acts : List[TrajectoryFragment], ped_acts : List[TrajectoryFragment], node):
        gamma_grid_matrix = np.meshgrid(np.linspace(start=-1, stop=1, num=20), np.linspace(start=-1, stop=1, num=20))
        ''' vehicle=0 pedestrian = 1'''
        gamma_matrix = [np.linspace(start=-1, stop=1, num=20), np.linspace(start=-1, stop=1, num=20)]
        equil_obj_mat_veh = np.empty(shape= gamma_matrix[0].shape, dtype=object)
        equil_obj_mat_peds = np.empty(shape= gamma_matrix[0].shape, dtype=object)
        node.equilibrium_solutions = np.empty(shape= (gamma_matrix[0].shape[0],gamma_matrix[1].shape[0]), dtype=object)
        u = Utilities()
        veh_acts.sort(key=lambda x: x.length)
        ped_acts.sort(key=lambda x: x.length)
        ''' pedestrian best response to vehicle's trajectory length'''
        
        interac_dict = OrderedDict()
        for traj_frag in veh_acts:
            veh_traj_l = traj_frag.length
            if veh_traj_l not in interac_dict:
                interac_dict[veh_traj_l] = []
            interac_dict[veh_traj_l].append(traj_frag)
        
        peds_best_response = OrderedDict()
        ct,N = 0,len(interac_dict)
        for v_traj_l,traj_frag_list in interac_dict.items():
            
            ct += 1
            resp_vect = []
            ''' vehicles can generate multiple distinct trajectories for the same trajectory length '''
            for veh_traj_frag in traj_frag_list:
                for peds_frag in ped_acts:
                    manv, manv_mode, ped_traj_l, dist_gap = peds_frag.manv, peds_frag.manv_mode, peds_frag.length, u.calc_dist_gap(veh_traj = veh_traj_frag.loaded_traj_frag, ped_traj = peds_frag.loaded_traj_frag)
                    assert ped_traj_l >= 0
                    if node.level == 4:
                        cont_util = np.zeros(shape= gamma_matrix[1].shape)
                    elif peds_frag._next_node.equilibrium_solutions is not None:
                        f = lambda x : max([y.peds_eq_utils for y in x])
                        cont_util = f(peds_frag._next_node.equilibrium_solutions)
                        #cont_util = max([max(x.peds_eq_utils) for x in peds_frag._next_node.equilibrium_solutions])
                    else:
                        continue 
                    step_util = u.combine_utils(u.progress_payoff_dist(ped_traj_l, 'pedestrian'), u.calc_safe_payoff(dist_gap), gamma_matrix[1])
                    _resp_entry = np.empty(shape= gamma_matrix[1].shape, dtype=object)
                    _util_entry_matrix = np.where(cont_util != 0, np.mean( np.array([ cont_util, step_util ]), axis=0 ), step_util)
                    manv_str_arr = np.full(shape = gamma_matrix[1].shape, fill_value=manv)
                    traj_l_arr = np.full(shape = gamma_matrix[1].shape, fill_value = ped_traj_l)
                    _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix), names=('manv', 'traj_l', 'utils'))
                    resp_vect.append(_resp_entry)
                    
            resp_vect = np.array(resp_vect)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            upper_bound_matrix = np.copy(resp_vect_sorted[0,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:], resp_vect_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(resp_vect_sorted)
            _x1 = resp_vect_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = resp_vect_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            selected_indices = np.argmin(_x3, axis=0) - 1
            lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            upper_bound_matrix = upper_bound_matrix[0,:]
            
            #print('pedestrian responding',ct,'/',N)
            if v_traj_l not in peds_best_response:
                if not np.array_equal(upper_bound_matrix, lower_bound_matrix):
                    peds_best_response[v_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix))
                else:
                    peds_best_response[v_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix))
            else:
                if len(peds_best_response[v_traj_l]) == 2 and not np.array_equal(upper_bound_matrix, lower_bound_matrix):
                    _merged_arr_ub = np.where(peds_best_response[v_traj_l][0]['utils'] > upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                    _merged_arr_lb = np.where(peds_best_response[v_traj_l][1]['utils'] < lower_bound_matrix['utils'], peds_best_response[v_traj_l][1], lower_bound_matrix)
                    peds_best_response[v_traj_l] = (_merged_arr_ub, _merged_arr_lb)
                elif len(peds_best_response[v_traj_l]) == 1 and not np.array_equal(upper_bound_matrix, lower_bound_matrix):
                    _merged_arr_ub = np.where(peds_best_response[v_traj_l][0]['utils'] > upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                    _merged_arr_lb = np.where(peds_best_response[v_traj_l][0]['utils'] < lower_bound_matrix['utils'], peds_best_response[v_traj_l][0], lower_bound_matrix)
                    peds_best_response[v_traj_l] = (_merged_arr_ub, _merged_arr_lb)
                elif len(peds_best_response[v_traj_l]) == 2 and np.array_equal(upper_bound_matrix, lower_bound_matrix):
                    _merged_arr_ub = np.where(peds_best_response[v_traj_l][0]['utils'] > upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                    _merged_arr_lb = np.where(peds_best_response[v_traj_l][1]['utils'] < upper_bound_matrix['utils'], peds_best_response[v_traj_l][1], upper_bound_matrix)
                    peds_best_response[v_traj_l] = (_merged_arr_ub, _merged_arr_lb)
                else:
                    _merged_arr_ub = np.where(peds_best_response[v_traj_l][0]['utils'] > upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                    _merged_arr_lb = np.where(peds_best_response[v_traj_l][0]['utils'] < upper_bound_matrix['utils'], peds_best_response[v_traj_l][0], upper_bound_matrix)
                    peds_best_response[v_traj_l] = (_merged_arr_ub, _merged_arr_lb)
                
            
            
        #X_lb_ub.sort(key=lambda tup: tup[0])
        
        f=1
        
        if len(peds_best_response) < 2:
            node.equilibrium_solutions = None
            return None
        
        ''' pedestrian best response function'''
        p1_matrix = np.empty(shape= gamma_matrix[1].shape, dtype=object)
        for i in np.arange(p1_matrix.shape[0]):
            p1 = (LineString(list(zip([x[0][i]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))) , LineString(list(zip([x[1][i]['traj_l'] for x in peds_best_response.values()],[x for x in peds_best_response.keys()]))))
            p1_matrix[i] = p1  
            
                
        
        if show_plots:
            plt.figure()
            plt.plot([x for x in peds_best_response.keys()],[x[0][0]['traj_l'] for x in peds_best_response.values()],c='red')
            plt.plot([x for x in peds_best_response.keys()],[x[1][0]['traj_l'] for x in peds_best_response.values()],c=lighten_color('red', .5),label = 'pedestrian best response')
        
                
                
        
        
        
        
        ''' vehicle best reponse to pedestrian trajectory choice'''
        interac_dict = OrderedDict()
        for traj_frag in ped_acts:
            ped_traj_l = traj_frag.length
            if ped_traj_l not in interac_dict:
                interac_dict[ped_traj_l] = []
            interac_dict[ped_traj_l].append(traj_frag)
        
        veh_best_response = OrderedDict()
        ct,N = 0,len(interac_dict)
        for p_traj_l,traj_frag_list in interac_dict.items():
            
            ct += 1
            resp_vect = []
            ''' pedestrians can generate multiple distinct trajectories for the same trajectory length '''
            for ped_traj_frag in traj_frag_list:
                for veh_frag in ped_acts:
                    manv, manv_mode, veh_traj_l, dist_gap = veh_frag.manv, veh_frag.manv_mode, veh_frag.length, u.calc_dist_gap(veh_traj = veh_traj_frag.loaded_traj_frag, ped_traj = peds_frag.loaded_traj_frag)
                    assert veh_traj_l >= 0
                    if node.level == 4:
                        cont_util = np.zeros(shape= gamma_matrix[0].shape)
                    elif peds_frag._next_node.equilibrium_solutions is not None:
                        f = lambda x: max([y.veh_eq_utils for y in x])
                        cont_util = f(veh_frag._next_node.equilibrium_solutions)
                        #cont_util = max([max(x.veh_eq_utils) for x in veh_frag._next_node.equilibrium_solutions])
                    else:
                        continue 
                    step_util = u.combine_utils(u.progress_payoff_dist(veh_traj_l, 'veh'), u.calc_safe_payoff(dist_gap), gamma_matrix[0])
                    _resp_entry = np.empty(shape= gamma_matrix[0].shape, dtype=object)
                    _util_entry_matrix = np.where(cont_util != 0, np.mean( np.array([ cont_util, step_util ]), axis=0 ), step_util)
                    manv_str_arr = np.full(shape = gamma_matrix[0].shape, fill_value=manv)
                    traj_l_arr = np.full(shape = gamma_matrix[0].shape, fill_value = veh_traj_l)
                    _resp_entry = np.rec.fromarrays((manv_str_arr, traj_l_arr, _util_entry_matrix), names=('manv', 'traj_l', 'utils'))
                    resp_vect.append(_resp_entry)
                    
            resp_vect = np.array(resp_vect)
            resp_vect_sorted = np.sort(resp_vect,axis=0,order='utils')[::-1]
            upper_bound_matrix = np.copy(resp_vect_sorted[0,:])
            upper_bound_matrix = np.repeat(upper_bound_matrix[np.newaxis,:], resp_vect_sorted.shape[0], axis=0)
            lower_bound_matrix = np.copy(resp_vect_sorted)
            _x1 = resp_vect_sorted['manv'] == upper_bound_matrix['manv']
            _x2 = resp_vect_sorted['utils'] == upper_bound_matrix['utils']
            _x3 = np.logical_or(_x1,_x2)
            selected_indices = np.argmin(_x3, axis=0) - 1
            lower_bound_matrix = np.take_along_axis(lower_bound_matrix,selected_indices[np.newaxis],axis=0)[0]
            upper_bound_matrix = upper_bound_matrix[0,:]
            
            #print('vehicle responding',ct,'/',N)
            if p_traj_l not in veh_best_response:
                    veh_best_response[p_traj_l] = (np.copy(upper_bound_matrix), np.copy(lower_bound_matrix))
            else:
                _merged_arr_ub = np.where(veh_best_response[p_traj_l][0]['utils'] > upper_bound_matrix['utils'], veh_best_response[p_traj_l][0], upper_bound_matrix)
                _merged_arr_lb = np.where(veh_best_response[p_traj_l][1]['utils'] < lower_bound_matrix['utils'], veh_best_response[p_traj_l][1], lower_bound_matrix)
                veh_best_response[p_traj_l] = (_merged_arr_ub, _merged_arr_lb)
        
        if len(veh_best_response) < 2:
            node.equilibrium_solutions = None
            return None
        
        ''' vehicle best response function'''
        p2_matrix = np.empty(shape= gamma_matrix[0].shape, dtype=object)
        for i in np.arange(p2_matrix.shape[0]):
            p2 = (LineString(list(zip([x for x in veh_best_response.keys()],[x[0][i]['traj_l'] for x in veh_best_response.values()]))) , LineString(list(zip([x for x in veh_best_response.keys()],[x[1][i]['traj_l'] for x in veh_best_response.values()]))))
            p2_matrix[i] = p2  
        
        if show_plots:
            plt.plot([x[0][0]['traj_l'] for x in veh_best_response.values()],[x for x in veh_best_response.keys()],c='blue')
            plt.plot([x[1][0]['traj_l'] for x in veh_best_response.values()],[x for x in veh_best_response.keys()],c=lighten_color('blue', .5),label = 'vehicle best response')
        
        
        
        self.veh_best_response = veh_best_response
        self.peds_best_response= peds_best_response
        
        '''
        p1:
            x : vehicle's trajectory length choice
            y : pedestrian best response
            
        p2:
            x : vehicle best reponse 
            y : pedestrian's trajectory length choice
        
        p[0]: upper bound
        p[1]: lower bound
        '''
        for i in np.arange(p2_matrix.shape[0]):
            for j in np.arange(p1_matrix.shape[0]):
                p1,p2 = p1_matrix[j], p2_matrix[i]
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
                    print('Equilibrium doesn\'t exists',gamma_matrix[0][i],gamma_matrix[1][j])
                if eq_obj is not None:
                    if isinstance(eq_obj, multipoint.MultiPoint):
                        eq_strat = self._process_MultiPoint(eq_obj)    
                        print(node.level,'equilibrium',eq_strat)
                    elif isinstance(eq_obj, point.Point):
                        eq_strat = self._process_Point(eq_obj)
                        print(node.level,'equilibrium',eq_strat)
                    elif isinstance(eq_obj, linestring.LineString):
                        eq_strat = self._process_LineString(eq_obj)
                        print(node.level,'equilibrium',eq_strat)
                    elif isinstance(eq_obj, multilinestring.MultiLineString):
                        eq_strat = []
                        for eq_item in eq_obj:
                            eq_strat += self._process_LineString(eq_item)
                        print(node.level,'equilibrium',eq_strat)
                    elif isinstance(eq_obj, GeometryCollection):
                        eq_strat = []
                        for eq_item in eq_obj:
                            if isinstance(eq_item, multipoint.MultiPoint):
                                eq_strat += self._process_MultiPoint(eq_item)    
                            elif isinstance(eq_item, point.Point):
                                eq_strat += self._process_Point(eq_item)
                            elif isinstance(eq_item, linestring.LineString):
                                eq_strat += self._process_LineString(eq_item)
                            else:
                                raise Exception('cannot process equilibrium of class '+type(eq_obj).__name__)
                        print(node.level,'equilibrium',eq_strat)
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
                    node.equilibrium_solutions[i,j] = eq_solns
                else:
                    node.equilibrium_solutions[i,j] = None
                
        return node.equilibrium_solutions
        
        