'''
Created on Apr 2, 2021

@author: Atrisha
'''

import numpy as np
from equilibrium.utilities import Utilities

class AutomataStrategies:
    
    def __init__(self,wait_manv,proceed_manv):
        
        self.wait_manv = wait_manv
        self.proceed_manv = proceed_manv
        self._util_obj = Utilities()
        
    def dg_2_safe_utils(self,dist_gap):
        return self._util_obj.calc_safe_payoff(dist_gap)
        
    
class AccommodatingGamma(AutomataStrategies):
    ''' accomodates always. and turn only if best case waiting is below a safety threshold (gamma). '''
    ''' word1 - [(empirical manv, {manv:(lower bound utility, upper bound utility)}),... for all sequence of nodes till this node]'''
    def accepts(self,word1):
        wait_ranges_ub = [self.dg_2_safe_utils(x[1][self.wait_manv][1]) for x in word1 if x[0]==self.wait_manv and not np.isnan(x[1][self.wait_manv][1])]
        turn_ranges_ub = [self.dg_2_safe_utils(x[1][self.wait_manv][1]) for x in word1 if x[0]==self.proceed_manv and not np.isnan(x[1][self.wait_manv][1])]
        ''' gamma has to be between the highest safety value for waiting at which i turned and lowest safety value for which i waited.'''
        gamma = (max(turn_ranges_ub) if len(turn_ranges_ub)>0 else -1,min(wait_ranges_ub) if len(wait_ranges_ub)>0 else -1)
        if gamma[0] <= gamma[1]:
            return gamma
        else:
            return False
    
class NonAccomodatingGamma(AutomataStrategies):
    ''' turns always. and wait only if best case turning is below a safety threshold (gamma). '''
        
    def accepts(self,word1):
        wait_ranges_ub = [self.dg_2_safe_utils(x[1][self.proceed_manv][1]) for x in word1 if x[0]==self.wait_manv and not np.isnan(x[1][self.proceed_manv][1])]
        turn_ranges_ub = [self.dg_2_safe_utils(x[1][self.proceed_manv][1]) for x in word1 if x[0]==self.proceed_manv and not np.isnan(x[1][self.proceed_manv][1])]
        
        gamma = (max(wait_ranges_ub) if len(wait_ranges_ub)>0 else -1, min(turn_ranges_ub) if len(turn_ranges_ub)>0 else -1)
        if gamma[0] <= gamma[1]:
            return gamma
        else:
            return False
    
    
class RuleFollowingGamma(AutomataStrategies):
    ''' always follows rules, but waits if the best case rule actions below a safety threshold (gamma). '''
        
    def accepts(self,word1,rule_actions):
        assert len(word1) == len(rule_actions)
        rule_actions_ub = [self.dg_2_safe_utils(x[1][x[0]][1]) for idx,x in enumerate(word1) if x[0]==rule_actions[idx] and not np.isnan(x[1][x[0]][1])]
        rule_violations_ub = [self.dg_2_safe_utils(x[1][x[0]][1]) for idx,x in enumerate(word1) if x[0]!=rule_actions[idx] and x[0]==self.wait_manv and not np.isnan(x[1][x[0]][1])]
        
        gamma = (max(rule_violations_ub) if len(rule_violations_ub)>0 else -1 , min(rule_actions_ub) if len(rule_actions_ub)>0 else 1)
        if gamma[0] <= gamma[1]:
            return gamma
        else:
            return False
                   
                
                 
                
                