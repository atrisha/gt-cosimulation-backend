'''
Created on Jan 26, 2021

@author: Atrisha
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

from equilibria import equilibria_core

def fun1():
    ''' prog, veh_inh, ped_inh'''
    payoff_dict = {('lt_go','ls_go','rs_go'):[(1,-1,1),(1,-1,1),(1,-1,1)],
                     ('lt_go','ls_wait','rs_go'):[(1,-1,1),(1,.5,1),(1,-1,1)],
                     ('lt_wait','ls_go','rs_go'):[(.5,1,1),(1,1,1),(1,1,1)],
                     ('lt_wait','ls_wait','rs_go'):[(.5,1,1),(.5,1,1),(1,1,1)],
                     ('lt_go','ls_go','rs_wait'):[(1,-1,1),(1,-1,1),(.5,1,1)],
                     ('lt_go','ls_wait','rs_wait'):[(1,1,1),(.5,1,1),(.5,1,1)],
                     ('lt_wait','ls_go','rs_wait'):[(.5,1,1),(1,1,1),(.5,1,1)],
                     ('lt_wait','ls_wait','rs_wait'):[(.5,1,1),(.5,1,1),(.5,1,1)]}
    lt_table = {('lt_go','rs_go'):[(1,-1,1),(1,-1,1)],
                     ('lt_wait','rs_go'):[(.5,1,1),(1,1,1)],
                     ('lt_go','rs_wait'):[(1,1,1),(.5,1,1)],
                     ('lt_wait','rs_wait'):[(.5,1,1),(.5,1,1)],
                     }
    
    rs_table = {('lt_go','ls_go','rs_go'):[(1,-1,1),(1,-1,1),(1,-1,1)],
                     ('lt_go','ls_wait','rs_go'):[(1,-1,1),(1,.5,1),(1,-1,1)],
                     ('lt_wait','ls_go','rs_go'):[(.5,1,1),(1,1,1),(1,1,1)],
                     ('lt_wait','ls_wait','rs_go'):[(.5,1,1),(.5,1,1),(1,1,1)],
                     ('lt_go','ls_go','rs_wait'):[(1,-1,1),(1,-1,1),(.5,1,1)],
                     ('lt_go','ls_wait','rs_wait'):[(1,1,1),(.5,1,1),(.5,1,1)],
                     ('lt_wait','ls_go','rs_wait'):[(.5,1,1),(1,1,1),(.5,1,1)],
                     ('lt_wait','ls_wait','rs_wait'):[(.5,1,1),(.5,1,1),(.5,1,1)]}
    
    ls_table = {('ls_go','rs_go'):[(1,1,1),(1,1,1)],
                     ('ls_wait','rs_go'):[(0,1,1),(1,1,1)],
                     ('ls_go','rs_wait'):[(1,1,1),(0,1,1)],
                     ('ls_wait','rs_wait'):[(0,1,1),(0,1,1)],
                     }
    w = [0.25,0.25,0.5]
    payoff_dict = {k:[w[0]*x[0]+w[1]*x[1]+w[2]*x[2] for x in v] for k,v in payoff_dict.items()}
    lt_table = {k:[w[0]*x[0]+w[1]*x[1]+w[2]*x[2] for x in v] for k,v in lt_table.items()}
    ls_table = {k:[w[0]*x[0]+w[1]*x[1]+w[2]*x[2] for x in v] for k,v in ls_table.items()}
    split_dicts = [dict(),dict()]
    for k,v in payoff_dict.items():
        if (k[0],k[1]) not in split_dicts[0]:
            split_dicts[0][(k[0],k[1])] = [v[0],v[1]]
        if (k[1],k[2]) not in split_dicts[1]:
            split_dicts[1][(k[1],k[2])] = [v[1],v[2]]
    eq_core = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(payoff_dict, True)
    print(eq_core)
    print('------------split 1-----------')
    eq_core = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(lt_table, True)
    print(eq_core)
    print('------------split 2-----------')
    eq_core = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(ls_table, True)
    print(eq_core)


def plot_action_sample_eq_chart():
    cl,cr,du,dd = .5,.8,.8,.9
    n = 7
    
    def _alpha_u(k):
        if k/n < cr / (cl+cr):
            return 1
        elif k/n > cr / (cl+cr):
            return 0
        else:
            return 0.5
        
    def _alpha_l(m):
        if m/n > du / (du+dd):
            return 1
        elif m/n < du / (du+dd):
            return 0
        else:
            return 0.5
    for n in [3,7]:
        plt.figure()
        X,Y = [],[]
        for ql in np.arange(0,1.01,.01):
            Y.append(ql)
            sum = 0
            for k in np.arange(n+1):
                rv = binom(n, ql)
                sum += rv.pmf(k)*_alpha_u(k)
            X.append(sum)
        plt.plot(X,Y,c='blue')
        X,Y = [],[]
        n = n+1
        for pu in np.arange(0,1.01,.01):
            X.append(pu)
            sum = 0
            for m in np.arange(n+1):
                rv = binom(n, pu)
                sum += rv.pmf(m)*_alpha_l(m)
            Y.append(1-sum)
        plt.plot(X,Y,c='green')   
        plt.xlabel('prob of speeding up')
        plt.ylabel('prob of turning')
        plt.title('N='+str(n)+' samples')     
    plt.show()

def cournot_motion():
    x1,x2 = 1,1
    u_s1 = ((1-x1)**2 + (1-x2)**2)**.5
    u_p1 = x1
    u_p2 = x2
    u1 = u_p1 + u_s1
    


#plot_action_sample_eq_chart()