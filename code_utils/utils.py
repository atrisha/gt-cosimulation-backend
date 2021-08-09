'''
Created on Apr 15, 2021

@author: Atrisha
'''


from matplotlib import pyplot
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import numpy as np
import constants
import rg_constants
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from shapely.geometry import LineString, Polygon, Point

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

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
cmap = get_cmap(100)
def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, c=np.random.rand(3,), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


def get_all_level_nodes(node,node_list,tree_level):
    if node.level == tree_level:
        node_list += [node]
        return node_list
    else:
        for c in node.children:
            get_all_level_nodes(c,node_list,tree_level)
        return node_list

def get_reasonable_velocities(seg,direction=None,ag_obj=None):
    seg_type = constants.SEGMENT_MAP[seg]
    target_vels = rg_constants.PROCEED_VEL_RANGES[seg_type] 
    if direction is not None and direction in ['L_W_S','L_W_N'] and seg_type=='exit-lane':
        target_vels = (target_vels[0]/2,target_vels[1]/2)
    if ag_obj is not None and ag_obj.velocity < 0.5:
        target_vels = (0.5,target_vels[1]/2)
    if ag_obj is not None and ag_obj.velocity < target_vels[0]:
        target_vels = (ag_obj.velocity,target_vels[1])
    return target_vels
    


def get_within_node(node_list,ag1_emptrajl,ag2_emptrajl):
    trajl_errs = []
    for c in node_list:
        ag_1l = c.path_from_root['agent_1'].get_last().length
        ag_2l = c.path_from_root['agent_2'].get_last().length
        if min(ag1_emptrajl) <= ag_1l <= max(ag1_emptrajl) and min(ag2_emptrajl) <= ag_2l <= max(ag2_emptrajl):
            trajl_errs.append(c._ext_id)
    return trajl_errs

def get_nearest_node(node_list,ag1_emptrajl,ag2_emptrajl):
    trajl_errs = []
    for c in node_list:
        ag_1l,ag_2l = [],[]
        ag_1l = c.path_from_root['agent_1'].get_last().length
        ag_2l = c.path_from_root['agent_2'].get_last().length
        '''
        while True:
            ag_1l.append(_tf.length if len(ag_1l)==0 else ag_1l[-1]+_tf.length)
            if _tf.next_fragment is None:
                break
            _tf = _tf.next_fragment 
        _tf = c.path_from_root['agent_2']
        while True:
            ag_2l.append(_tf.length if len(ag_2l)==0 else ag_2l[-1]+_tf.length)
            if _tf.next_fragment is None:
                break
            _tf = _tf.next_fragment    
        '''
        trajl_errs.append((c._ext_id,abs(ag1_emptrajl-ag_1l),abs(ag2_emptrajl-ag_2l)))
    trajl_errs.sort(key=lambda tup: tup[1]+tup[2])
    return trajl_errs[0]


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def to_type(idx_list):
    idx_to_type = {0:-1,1:-0.5,2:0,3:0.5,4:1}
    return [idx_to_type[x] for x in idx_list]

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def plot_heatmap(plt,fig,axis,data):
    data = data.astype(float)
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
    [c('white'), 0.25,c('red'), c('violet'), c('blue')])
    rvb = make_colormap(
    [c('red'), c('violet'), c('blue')])
    
    row_labels = ["extreme high",
              "high",
              "normal",
              "low",
              "extreme low"]
    heatmap = axis.pcolor(data,cmap=rvb ) # cmap='autumn,'rvb' heatmap contient les valeurs
    
    axis.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    axis.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    
    axis.invert_yaxis()
    
    #axis.set_yticklabels(row_labels, minor=False)
    #axis.set_xticklabels(column_labels, minor=False)
    axis.get_xaxis().set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.get_xaxis().set_ticks([])
    #fig.set_size_inches(11.03, 3.5)
    plt.colorbar(heatmap)

def has_crossed(pt,origin_pt,line):
    exit_pos_X = [x[0] for x in line.coords]
    exit_pos_Y = [x[1] for x in line.coords]
    m = (exit_pos_Y[1] - exit_pos_Y[0]) / (exit_pos_X[1] - exit_pos_X[0])
    c = (exit_pos_Y[0] - (m * exit_pos_X[0]))
    
    veh_pos_x, veh_pos_y = pt[0],pt[1]
    #dist_to_exit_boundary = distance_numpy([exit_pos_X[0],exit_pos_Y[0]], [exit_pos_X[1],exit_pos_Y[1]], [veh_pos_x,veh_pos_y])
    #dist_from_veh_origin_to_exit_boundary = distance_numpy([exit_pos_X[0],exit_pos_Y[0]], [exit_pos_X[1],exit_pos_Y[1]], [veh_orig_x,veh_orig_y])
    veh_orig_x,veh_orig_y = origin_pt[0], origin_pt[1]
    res_wrt_origin = veh_orig_y - (m*veh_orig_x) - c
    res_wrt_point = veh_pos_y - (m*veh_pos_x) - c
    
    return True if np.sign(res_wrt_origin) != np.sign(res_wrt_point) else False


def redistribute_vertices(geom, distance):
    num_vert = int(round(geom.length / distance))
    if num_vert == 0:
        num_vert = 1
    return LineString(
        [geom.interpolate(float(n) / num_vert, normalized=True)
         for n in range(num_vert + 1)])
    
def cut_line(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]