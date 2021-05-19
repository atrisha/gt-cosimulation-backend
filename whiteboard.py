'''
Created on Jan 26, 2021

@author: Atrisha
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sqlite3
from scipy.stats import binom
from planners.trajectory_planner import TrajectoryPlanner, VehicleTrajectoryPlanner, PedestrianTrajectoryPlanner
from planners.trajectory_planner import WaitTrajectoryConstraints, ProceedTrajectoryConstraints
from equilibria import equilibria_core
import time
import constants
from motion_planners.planning_objects import VehicleState
import all_utils.utils

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

WAIT_ACTIONS = ['yield-to-merging','wait_for_lead_to_cross','wait-for-oncoming','decelerate-to-stop','wait-on-red','wait-for-pedestrian']
def test_ws_freeturn():
    start_time = time.time()
    rt_manv = 'wait-for-oncoming'
    st_manv = 'track_speed'
    """
    Test that trajectory generation works for west to south free turn
    """
    
    ''' Get the track of a representative right turn vehicle to construct a path centerline '''
    
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+'769'+'\\uni_weber_'+'769'+'.db')
    c = conn.cursor()
    q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID=2 ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    rt_path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
    
    ''' Get the track of a representative straight through vehicle to construct a path centerline '''
    q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID=28 ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    st_path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
    
    ''' Construct the velocity range samples. First point is the initial velocity (v0,)
    The mid point velocity is a reasonable sampling range for a turning trajectory 
    The exit point velocity is (5,8.3) 
    '''
    rt_vel_pts = [(6,)] + [(None,) if i != len(np.arange(1,len(rt_path)-1))//2 else (2,4) for i in np.arange(1,len(rt_path)-1)] + [(5,8.3)]
    
    ''' 
    Straight through vehicles can just have an initial and exit velocity point constraints.
    '''
    st_vel_pts = [(8,)] + [(None,)]*(len(st_path)-2) + [(8,14)]
    
    rt_trajs,st_trajs = dict(), dict()
    
    ''' Some housekeeping to translate the internal maneuver codes '''
    if rt_manv in WAIT_ACTIONS:
        rt_manv = 'wait'
    else:
        rt_manv = 'turn'
    if st_manv in WAIT_ACTIONS:
        st_manv = 'wait'
    
    ''' Setup the constraints object to generate the trajectory '''
    
    if rt_manv == 'wait':
        rt_traj_constr = WaitTrajectoryConstraints(init_vel=8,waypoints=rt_path,stop_horizon_dist_sampling_range=(20,30),stop_horizon_time_sampling_range=(4,8))
    else:
        rt_traj_constr = ProceedTrajectoryConstraints(waypoints=rt_path,waypoint_vel_sampling_range=rt_vel_pts)
    rt_traj_constr.set_limit_constraints()
    rt_motion = VehicleTrajectoryPlanner(rt_traj_constr,rt_manv,None,6)
    rt_motion.generate_trajectory(True)
    rt_trajs[rt_manv] = rt_motion.all_trajectories
    assert len(rt_trajs[rt_manv]) > 0
    
    
    
    #plt.figure()
    for traj_obj,traj_list in rt_trajs[rt_manv].items():
        for traj in traj_list:
            #plt.plot([x[0] for x in traj], [x[3] for x in traj])
            f=1
    #plt.title('right turn velocities')
    print('right turn trajectories generated')
    sel_rt_traj = list(zip([float(x[1]) for x in traj], [float(x[2]) for x in traj]))
    
    st_traj_constr = ProceedTrajectoryConstraints(waypoints=st_path,waypoint_vel_sampling_range=st_vel_pts)
    st_traj_constr.set_limit_constraints()
    st_motion = VehicleTrajectoryPlanner(st_traj_constr,st_manv,None,6)
    st_motion.generate_trajectory(True)
    st_trajs[st_manv] = st_motion.all_trajectories
    
    assert len(st_trajs[st_manv]) > 0
    
    #plt.figure()
    for traj_obj,traj_list in st_trajs[st_manv].items():
        for traj in traj_list:
            #plt.plot([x[0] for x in traj], [x[3] for x in traj])
            f=1
    #plt.title('straight through velocities')
    print('straight through trajectories generated')
    
    sel_st_traj = list(zip([float(x[1]) for x in traj], [float(x[2]) for x in traj]))
    
    
    #plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    ''' Just take one example, the one in sel_xx_traj, and show the animation '''
    '''
    fig, ax = plt.subplots()
    ax = plt.axes(xlim=(min([x[0] for x in sel_st_traj]+[x[0] for x in sel_rt_traj])-10, max([x[0] for x in sel_st_traj]+[x[0] for x in sel_rt_traj])+10), ylim=(min([x[1] for x in sel_st_traj]+[x[1] for x in sel_rt_traj])-10, max([x[1] for x in sel_st_traj]+[x[1] for x in sel_rt_traj])+10))
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1,line2,
    
    # animation function.  This is called sequentially
    def animate(i):
        line1.set_data([x[0] for x in sel_rt_traj[:i]], [x[1] for x in sel_rt_traj[:i]])
        line2.set_data([x[0] for x in sel_st_traj[:i]], [x[1] for x in sel_st_traj[:i]])
        return line1,line2,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(sel_st_traj), interval=20, blit=True, repeat = False) 
    plt.show()
    '''

#test_ws_freeturn()
'''
a = np.array([[[3,5],[32,5]], [[5,6],[30,0]], [[13,5],[32,95]], [[3,65],[352,5]]])
b = np.array([[[14,25],[332,35]], [[35,67],[302,50]], [[413,75],[352,965]], [[33,65],[3252,65]]])
arr = np.rec.fromarrays((a,  b), names=('manv', 'utils'))
arr_sorted = np.sort(arr,axis=0,order='utils')
a_idxarr = arr.argsort(order='utils')
'''
    
SEGMENT_MAP = {
               'prep-turn_s':'prep-left-turn',
               'prep-turn_n':'prep-left-turn',
               'prep-turn_e':'prep-left-turn',
               'exec-turn_e':'exec-left-turn',
               'exec-turn_n':'exec-left-turn',
               'exec-turn_s':'exec-left-turn',
               'prep-turn_w':'prep-left-turn',
               'exec-turn_w':'exec-left-turn',
               'rt_prep-turn_s':'prep-right-turn',
               'rt_exec-turn_s':'exec-right-turn',
               'rt_prep-turn_w':'prep-right-turn',
               'rt_exec-turn_w':'exec-right-turn',
               'ln_w_-2':'exit-lane',
               'ln_w_-1':'exit-lane',
               'ln_w_1':'left-turn-lane',
               'ln_w_2':'through-lane-entry',
               'ln_w_3':'through-lane-entry',
               'ln_w_4':'right-turn-lane',
               'ln_n_-2':'exit-lane',
               'ln_n_-1':'exit-lane',
               'ln_n_1':'left-turn-lane',
               'ln_n_2':'through-lane-entry',
               'ln_n_3':'through-lane-entry',
               'ln_s_-2':'exit-lane',
               'ln_s_-1':'exit-lane',
               'ln_s_1':'left-turn-lane',
               'ln_s_2':'through-lane-entry',
               'ln_s_3':'through-lane-entry',
               'ln_s_4':'right-turn-lane',
               'ln_e_-2':'exit-lane',
               'ln_e_-1':'exit-lane',
               'ln_e_1':'left-turn-lane',
               'ln_e_2':'through-lane-entry',
               'ln_e_3':'through-lane-entry',
               'l_n_s_l':'through-lane',
               'l_n_s_r':'through-lane',
               'l_s_n_l':'through-lane',
               'l_s_n_r':'through-lane',
               'l_e_w_l':'through-lane',
               'l_e_w_r':'through-lane',
               'l_w_e_l':'through-lane',
               'l_w_e_r':'through-lane',
               }

def get_available_actions(vehicle):
    segment = vehicle.current_segment
    segment = constants.SEGMENT_MAP[segment]
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    cur = conn.cursor()
    command = f'SELECT L1_ACTION FROM ACTIONS WHERE SEGMENT="{segment}" AND (TRAFFIC_SIGNAL="{vehicle.signal}" OR TRAFFIC_SIGNAL="*")'
    cur.execute(command)
    query_results = cur.fetchall()
    actions = [a[0] for a in query_results]
    return actions    

def generate_trajectories(vehicle):
    # If vehicle id is not -1 then it is not an occluding vehicle and we
    # can retrieve its trajectory from the database
    constants.CURRENT_FILE_ID = '769'
    
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    q_string = f"select * from TRAJECTORIES_0{constants.CURRENT_FILE_ID} WHERE TRAJECTORIES_0{constants.CURRENT_FILE_ID}.TRACK_ID={vehicle.id} AND TIME>={vehicle.current_time} ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    path = [(x[1],x[2]) for idx,x in enumerate(res) if idx in [int(y) for y in np.linspace(start=0, stop=len(res)-1, num=10)]]
    # THIS IS TO SOLVE THE PROBLEM OF INTERPOLATED RELEVANT VEHICLES
    print(f"PATH[0] == {path[0]}; VEHICLE POS: {(vehicle.x, vehicle.y)}")
    if path[0] != (vehicle.x, vehicle.y):
        path.insert(0, (vehicle.x, vehicle.y))
    maneuver = 'right-turn'
    if maneuver == 'right-turn':
        vel_pts = [(6,)] + [(None,) if i != len(np.arange(1,len(path)-1))//2 else (2,4) for i in np.arange(1,len(path)-1)] + [(5,8.3)]
    elif maneuver == 'left-turn':
        vel_pts = [(3,)] + [(None,) if i != len(np.arange(1,len(path)-1))//2 else (3,8) for i in np.arange(1,len(path)-1)] + [(8,11)]
    else:
        vel_pts = [(8,)] + [(None,)]*(len(path)-2) + [(8,14)]
    trajectories = {}
    actions = ['proceed-turn']
    for action in actions:
        if maneuver == 'right-turn' or maneuver == 'right-turn':
            action_type = 'turn'
        elif action in constants.WAIT_ACTIONS:
                action_type = 'wait'
        else:
            action_type = 'track_speed'
        if action_type == 'wait':
            traj_constr = WaitTrajectoryConstraints(init_vel=8,waypoints=path,stop_horizon_dist_sampling_range=(20,30),stop_horizon_time_sampling_range=(4,8))
        else:
            traj_constr = ProceedTrajectoryConstraints(waypoints=path,waypoint_vel_sampling_range=vel_pts)
        traj_constr.set_limit_constraints()
        motion = VehicleTrajectoryPlanner(traj_constr,action_type,None,6)
        motion.generate_trajectory(True)
        trajectories[action] = motion.all_trajectories
    return trajectories

constants.CURRENT_FILE_ID = '769'
vehicle = all_utils.utils.setup_vehicle_state(2, 4.004)
trajs = generate_trajectories(vehicle)
f=1