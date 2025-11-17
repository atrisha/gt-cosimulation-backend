'''
Created on Aug 9, 2021

@author: Atrisha
'''
import xml.etree.ElementTree as ET
import ast
from pyproj import Proj
from shapely.geometry import LineString, Polygon


def parse_scenes(inp_scene,segs):
    lane_map = dict()
    for scene_id in [str(x) for x in [inp_scene]]:
        myProj = Proj("+proj=utm +zone=32N, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        root = ET.parse('D:\\repeated_games_data\\intersection_dataset\\maps\\inD_scene_'+scene_id+'.osm').getroot()
        node_map = dict()
        for type_tag in root.findall('node'):
            if type_tag.attrib['id'] not in node_map:
                node_map[type_tag.attrib['id']] = (type_tag.attrib['lat'],type_tag.attrib['lon'])
        for type_tag in root.findall('way'):
            lane_tag = type_tag.find('tag').attrib['v']
            print('----',lane_tag)
            for nd in type_tag:
                if 'ref' in nd.attrib:
                    lat,lon = float(node_map[nd.attrib['ref']][0]), float(node_map[nd.attrib['ref']][1]) 
                    utm_proj = myProj(lon, lat)
                    print(utm_proj)
                    if scene_id not in lane_map:
                        lane_map[scene_id] = dict()
                    if lane_tag not in lane_map[scene_id]:
                        lane_map[scene_id][lane_tag] = [utm_proj]
                    else:
                        lane_map[scene_id][lane_tag].append(utm_proj)
    for k,v in lane_map.items():
        print(k)
        for k1,v1 in v.items():
            print(k1,v1)
    if inp_scene is not None:
        regions = []
        for s in segs:
            try:
                regions.append(Polygon(lane_map[str(inp_scene)][s]))
            except:
                f=1
                raise
        return regions
#parse_scenes()