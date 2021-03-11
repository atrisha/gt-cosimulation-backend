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
        
viz = Visualization()
viz.show_nyc_trajectories_dynamic()