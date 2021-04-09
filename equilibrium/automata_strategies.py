'''
Created on Apr 2, 2021

@author: Atrisha
'''

import numpy as np

class AccomodatingGamma:
    ''' accomodates always. and turn only if best case waiting is below a safety threshold (gamma). '''
    def __init__(self,gamma):
        self.gamma = gamma
        
    def accepts(self,word1,word2):
        gamma = self.gamma
        
        wait_ranges_ub = [x[1]['wait'][1] for x in word1 if x[0]=='wait']
        turn_ranges_ub = [x[1]['wait'][1] for x in word1 if x[0]=='turn']
        ''' gamma has to be between the highest safety value for waiting at which i turned and lowest safety value for which i waited.'''
        gamma = (max(turn_ranges_ub),min(wait_ranges_ub))
        if gamma[0] <= gamma[1]:
            return gamma
        else:
            return False
    
class NonAccomodatingGamma:
    ''' turns always. and wait only if best case turning is below a safety threshold (gamma). '''
    def __init__(self,gamma):
        self.gamma = gamma
        
    def accepts(self,word1,word2):
        gamma = self.gamma
        
        wait_ranges_ub = [x[1]['turn'][1] for x in word1 if x[0]=='wait']
        turn_ranges_ub = [x[1]['turn'][1] for x in word1 if x[0]=='turn']
        
        gamma = (max(wait_ranges_ub),min(turn_ranges_ub))
        if gamma[0] <= gamma[1]:
            return gamma
        else:
            return False
    
    
class RuleFollowingGamma:
    ''' always follows rules, but waits if the best case rule actions below a safety threshold (gamma). '''
    def __init__(self,gamma):
        self.gamma = gamma
        
    def accepts(self,word1,rule_actions):
        gamma = self.gamma
        
        rule_actions_ub = [x[1][x[0]][1] for idx,x in enumerate(word1) if x[0]==rule_actions[idx]]
        rule_violations_ub = [x[1][x[0]][1] for idx,x in enumerate(word1) if x[0]!=rule_actions[idx] and x[0]=='wait']
        
        gamma = (max(rule_violations_ub),min(rule_actions_ub))
        if gamma[0] <= gamma[1]:
            return gamma
        else:
            return False
                   
                
                 
                
                