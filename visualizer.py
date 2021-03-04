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


class Visualization:
    
      
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
        for l in lanes:
            plt.plot([x[0] for x in l],[x[1] for x in l])
        plt.show()
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
            pts[(row[0],row[6])].append((float(row[4]),float(row[5])))
            
        
        fig, ax = plt.subplots()
        ax = plt.axes(xlim=(-13, 7), ylim=(0, 56))
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
viz.show_lanes()