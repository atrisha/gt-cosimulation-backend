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

def get_reasonable_velocities(seg,direction=None):
    seg_type = constants.SEGMENT_MAP[seg]
    target_vels = rg_constants.PROCEED_VEL_RANGES[seg_type] 
    if direction is not None and direction in ['L_W_S','L_W_N'] and seg_type=='exit-lane':
        target_vels = (target_vels[0]/2,target_vels[1]/2)
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
