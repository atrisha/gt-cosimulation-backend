'''
Created on Mar 2, 2021

@author: Atrisha
'''

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import animation
from scipy.ndimage import gaussian_filter1d
import math
from equilibrium.range_estimation import MinDistanceGapModel
import rg_constants



def gaussian_smoothing(a):
    x, y = a.T
    t = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, 100)
    
    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    sigma = 10
    x3 = gaussian_filter1d(x2, sigma)
    y3 = gaussian_filter1d(y2, sigma)
    
    x4 = np.interp(t, t2, x3)
    y4 = np.interp(t, t2, y3)
    
    return x4,y4

def rotate_line(o_x,o_y,X,Y,yaw):
    F_X,F_Y = [],[]
    a = yaw
    h,k = o_x,o_y
    ''' from https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    rotation_and_translation_matrix = np.asarray([[np.cos(a), -np.sin(a), h],\
                                                 [np.sin(a), np.cos(a), k],\
                                                 [0, 0, 1]])
    '''
    translation_matrix = np.asarray([[1, 0, -h],\
                                    [0, 1, -k],\
                                    [0, 0, 1]])
    
    rotation_matrix = np.asarray([[np.cos(a), np.sin(a), 0],\
                                 [-np.sin(a), np.cos(a), 0],\
                                 [0, 0, 1]])
    
    for x,y in zip(X,Y):
        point = np.asarray([x, y, 1]).T
        translated_point = np.matmul(translation_matrix,point)
        #new_point = translated_point
        new_point = np.matmul(rotation_matrix, translated_point)
        F_X.append(new_point[0])
        F_Y.append(new_point[1])
    return F_X,F_Y

class Visualization:
    
    def show_lanes_trasformed(self,ax):
        lanes,cluster_ids = self.show_lanes()
        cols = ['r','g','b','y','c']
        for i,l in enumerate(lanes):
            RX,RY = rotate_line(-3,27,[x[0] for x in l],[x[1] for x in l], math.pi*.1)
            print(cluster_ids[i],cols[i])
            if cluster_ids[i] == 2:
                ax.plot([-3.8,-4]+RX+[-9.8],[19,5]+RY+[-11.3],c=cols[i])
            elif cluster_ids[i] == 1:
                ax.plot(RX+[-9.6],RY+[-14],c=cols[i])
            else:
                ax.plot(RX,RY,c=cols[i])
        #plt.show()
        
      
    def show_lanes(self):
        db = sqlite3.connect('D:\\influence-net\\influence-net\\trajectories.db')
        cursor = db.cursor()
        string = "SELECT OBJECT.ID FROM OBJECT,ANNOTATION WHERE ANNOTATION.ID=OBJECT.PID AND ANNOTATION.FILENAME='reference'"
        cursor.execute(string)
        res = cursor.fetchone()
        lanes = []
        cluster_ids = []
        while res is not None:
            list_x,list_y = [],[]
            cursor_1 = db.cursor()
            string = "SELECT OBJECT.ID,PT_CAMERA_COOR.T, PT_CAMERA_COOR.X,PT_CAMERA_COOR.Y,PT_CAMERA_COOR_ADD_OPTS.CLUSTER FROM PT_CAMERA_COOR,OBJECT,ANNOTATION,PT_CAMERA_COOR_ADD_OPTS WHERE OBJECT.ID = PT_CAMERA_COOR.PID AND ANNOTATION.ID=OBJECT.PID AND ANNOTATION.FILENAME='reference' AND OBJECT.ID=? AND PT_CAMERA_COOR_ADD_OPTS.ID = PT_CAMERA_COOR.ID ORDER BY CAST(PT_CAMERA_COOR.T AS UNSIGNED) ASC"
            cursor_1.execute(string,[str(res[0])])
            res_1 = cursor_1.fetchone()
            cluster_id = None
            while res_1 is not None:
                list_x.append(float(res_1[2]))
                list_y.append(float(res_1[3]))
                if cluster_id == None:
                    cluster_id = res_1[4]
                res_1 = cursor_1.fetchone()
            
            '''smooth_list_x = s.smooth(np.asarray(list_x), window_len=3,window='flat')
            smooth_list_y = s.smooth(np.asarray(list_y), window_len=3,window='flat')'''
            z = [[e1,e2] for e1,e2 in zip(list_x,list_y)]
            a = np.asarray(z)
            smooth_list_x, smooth_list_y = gaussian_smoothing(a)
            cluster_ids.append(cluster_id)
            data_points_path = [(a,b) for a,b in zip(smooth_list_x,smooth_list_y)]
            lanes.append(data_points_path)
            res = cursor.fetchone()
        
        db.close()
        '''
        for l in lanes:
            plt.plot([x[0] for x in l],[x[1] for x in l])
        plt.show()
        '''
        return lanes,cluster_ids 
    
    def show_nyc_trajectories_static(self):
        
        conn = sqlite3.connect('D:\\influence-net\\influence-net\\trajectories.db')
        c = conn.cursor()
        q_string = "SELECT ANNOTATION.id AS FILE_ID,OBJECT.id AS OBJECT_ID, PT_CAMERA_COOR.* \
                    FROM OBJECT INNER JOIN \
                            ANNOTATION \
                    ON \
                            OBJECT.pid=ANNOTATION.id \
                        INNER JOIN PT_CAMERA_COOR ON PT_CAMERA_COOR.pid=OBJECT.id"
        c.execute(q_string)
        res = c.fetchall()
        trajs = dict()
        for row in res:
            if (row[0],row[1]) not in trajs: 
                trajs[(row[0],row[1])] = []
            trajs[(row[0],row[1])].append((float(row[4]),float(row[5])))
        for k,v in trajs.items():
            plt.scatter([x[0] for x in v],[x[1] for x in v],s=5)
        
        plt.show()
        
    def  show_nyc_trajectories_dynamic(self):
        conn = sqlite3.connect('D:\\influence-net\\influence-net\\trajectories.db')
        c = conn.cursor()
        q_string = "SELECT ANNOTATION.id AS FILE_ID,OBJECT.id AS OBJECT_ID, PT_CAMERA_COOR.*, PT_CAMERA_COOR_ADD_OPTS.cluster, PT_CAMERA_COOR_ADD_OPTS.x_v, PT_CAMERA_COOR_ADD_OPTS.y_v, PT_CAMERA_COOR_ADD_OPTS.theta \
                    FROM OBJECT INNER JOIN \
                            ANNOTATION \
                    ON \
                            OBJECT.pid=ANNOTATION.id \
                        INNER JOIN PT_CAMERA_COOR ON PT_CAMERA_COOR.pid=OBJECT.id INNER JOIN \
                            PT_CAMERA_COOR_ADD_OPTS \
                        ON \
                            PT_CAMERA_COOR_ADD_OPTS.id=PT_CAMERA_COOR.id \
                        order by file_id,CAST(t AS INTEGER)"
        c.execute(q_string)
        res = c.fetchall()
        pts = OrderedDict()
        for row in res:
            if (row[0],row[6]) not in pts: 
                pts[(row[0],row[6])] = []
            #pts[(row[0],row[6])].append((float(row[4]),float(row[5])))
            new_pt = rotate_line(-3,27,[float(row[4])],[float(row[5])], math.pi*.1)
            #print(new_pt)
            pts[(row[0],row[6])].append((new_pt[0][0],new_pt[1][0]))
        
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(-10, 10), ylim=(-20, 20))
        self.show_lanes_trasformed(ax)
        scat = ax.scatter([], [], s=5)
        
        def init():
            scat.set_offsets([])
            return scat,
        
        def animate(i):
            scat.set_offsets(list(pts.values())[i])
            print(i,list(pts.values())[i])
            return scat,
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(pts)+1, 
                               interval=20, blit=False, repeat=False)
        plt.show()
        
class UniWeberAnalytics:
    
    def __init__(self,file_id):
        self.file_id = file_id
        
    def plot_velocities(self):
        
        conn = sqlite3.connect('D:\\repeated_games_data\\intersection_dataset\\db_files\\'+self.file_id+'.db')
        c = conn.cursor()
        ''' get all vehicle trajectories '''
        q_string = "select TRAJECTORY_METADATA.TRAJ_ID, TRAJECTORY_METADATA.MANEUVER,TRAJECTORY_METADATA.MANEUVER_MODE from TRAJECTORY_METADATA WHERE TRAJECTORY_METADATA.INIT_TIME=0 AND TRAJECTORY_METADATA.AGENT_TYPE='agent_1'"
        c.execute(q_string)
        res = c.fetchall()
        m = MinDistanceGapModel(self.file_id)
        
    def animate_scene(self,trajs_list):
        
        fig, ax = plt.subplots()
        lines = [plt.plot([], [],lw=2)[0] for _ in range(len(trajs_list))] 
        #plt.xlim(538780, 538890)
        #plt.ylim(4813970, 4814055)
        
        #plot_traffic_regions(ax)
        
        img = plt.imread("D:\\behavior modeling\\background.jpg")
        ax.imshow(img, extent=[538780, 538890, 4813970, 4814055])
        
        # initialization function: plot the background of each frame
        patches = lines
        def init():
            for ln in lines:
                ln.set_data([], [])
            
            return patches
        
        # animation function.  This is called sequentially
        def animate(i):
            #print('called in loop',i)
            for idx,ln in enumerate(lines):
                if i > len(trajs_list[idx]):
                    ln.set_data([], [])
                else:
                    ln.set_data([x[0] for x in trajs_list[idx][:i]], [x[1] for x in trajs_list[idx][:i]])
            
            return patches
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=max([len(x) for x in trajs_list]), interval=100, blit=True, repeat = True) 
        plt.show()
        
        
        
        
    

class UniWeberResultsAnalysis:
    
    def plot_velocity_profiles(self,gt,scene_def,freq):
        ag1_vel_profiles,ag2_vel_profiles = {'1':[],'2':[],'3':[]},{'1':[],'2':[],'3':[]}
        l2_nodes = get_all_level_nodes(node=gt.root,node_list=[],tree_level=int(1/gt.freq))
        if not (len(scene_def.agent1_emp_traj)==0 or len(scene_def.agent2_emp_traj)==0):
            emp_2l = get_nearest_node(l2_nodes, scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[0])
            for n2l in l2_nodes:
                '''assign emp path value '''
                ag1_vel_profiles['1'].append([x[3] for x in n2l.path_from_root['agent_1'].loaded_traj_frag])
                ag2_vel_profiles['1'].append([x[3] for x in n2l.path_from_root['agent_2'].loaded_traj_frag])
                if n2l._ext_id == emp_2l[0]:
                    n2l.emp_path = True
                else:
                    n2l.emp_path = False
                if len(scene_def.agent1_emp_traj) > 1 and n2l.children is not None:
                        emp_4l = get_nearest_node(n2l.children, scene_def.agent1_emp_traj[1]-scene_def.agent1_emp_traj[0], scene_def.agent2_emp_traj[1]-scene_def.agent2_emp_traj[0])
                        for n4l in n2l.children:
                            if n4l._ext_id == emp_4l[0]:
                                n4l.emp_path = True
                            else:
                                n4l.emp_path = False
                            ag1_vel_profiles['2'].append([x[3] for x in n4l.path_from_root['agent_1'].get_last().loaded_traj_frag])
                            ag2_vel_profiles['2'].append([x[3] for x in n4l.path_from_root['agent_2'].get_last().loaded_traj_frag])
                            if len(scene_def.agent1_emp_traj) > 2 and n4l.children is not None:
                                emp_6l = get_nearest_node(n4l.children, scene_def.agent1_emp_traj[2]-scene_def.agent1_emp_traj[1], scene_def.agent2_emp_traj[2]-scene_def.agent2_emp_traj[1])
                                for n6l in n4l.children:
                                    if n6l._ext_id == emp_6l[0]:
                                        n6l.emp_path = True
                                    else:
                                        n6l.emp_path = False
                                    ag1_vel_profiles['3'].append([x[3] for x in n6l.path_from_root['agent_1'].get_last().loaded_traj_frag])
                                    ag2_vel_profiles['3'].append([x[3] for x in n6l.path_from_root['agent_2'].get_last().loaded_traj_frag])
        
        f=1
        
        
class RGvisualize():
    
    def plot_all_paths(self):
        rg_constants.SCENE_TYPE = ('synthetic','intersection_clearance')
        file_id = '1_1-3_0,33_17'
        conn = sqlite3.connect(rg_constants.get_rg_db_path(file_id))
        c = conn.cursor()
        q_string = "select traj_id,time,x,y from TRAJECTORY_METADATA INNER JOIN TRAJECTORIES ON TRAJECTORY_METADATA.TRAJ_ID=TRAJECTORIES.TRACK_ID AND TRAJECTORY_METADATA.AGENT_TYPE='agent_2' order by traj_id,time"
        c.execute(q_string)
        res = c.fetchall()
        traj_dict = dict()
        for row in res:
            if row[0] not in traj_dict:
                traj_dict[row[0]] = [(row[2],row[3])]
            else:
                traj_dict[row[0]].append((row[2],row[3]))
        for k,v in traj_dict.items():
            plt.plot([x[0] for x in v],[x[1] for x in v])
            plt.arrow(v[-2][0], v[-2][1],v[-1][0]-v[-2][0] , v[-1][1]-v[-2][1], width=.5)
        plt.ylim(4813900.6429128, 4814099.18)
        plt.xlim(538786.97, 538864.038146639)
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    vis = RGvisualize()
    vis.plot_all_paths()
                
        