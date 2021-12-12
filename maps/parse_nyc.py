'''
Created on Aug 9, 2021

@author: Atrisha
'''
import xml.etree.ElementTree as ET
import ast
from pyproj import Proj
from shapely.geometry import LineString, Polygon


def parse_scenes():
    lane_map = dict()
    #myProj = Proj("+proj=utm +zone=18T, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    #root = ET.parse('D:\\repeated_games_data\\intersection_dataset\\maps\\nyc_lanes.osm').getroot()
    myProj = Proj("+proj=utm +zone=17T, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    root = ET.parse('C:\\Users\\pc-admin\\Downloads\\roundabout_ll2.osm').getroot()
    node_map = dict()
    for type_tag in root.findall('node'):
        if type_tag.attrib['id'] not in node_map:
            node_map[type_tag.attrib['id']] = (type_tag.attrib['lat'],type_tag.attrib['lon'])
        node_tags = type_tag.findall('tag')
        if len(node_tags) > 0:
            lat,lon = float(type_tag.attrib['lat']), float(type_tag.attrib['lon']) 
            utm_proj = myProj(lon, lat)
            if node_tags[0].attrib['k'] == 'name':
                print(node_tags[0].attrib['v'],utm_proj)
    for type_tag in root.findall('way'):
        if type_tag.find('tag') is not None:
            lane_tag = type_tag.find('tag').attrib['v']
            print('----',lane_tag)
            for nd in type_tag:
                if 'ref' in nd.attrib:
                    lat,lon = float(node_map[nd.attrib['ref']][0]), float(node_map[nd.attrib['ref']][1]) 
                    utm_proj = myProj(lon, lat)
                    print(utm_proj)
                    if lane_tag not in lane_map:
                        lane_map[lane_tag] = [utm_proj]
                    else:
                        lane_map[lane_tag].append(utm_proj)
    for k,v in lane_map.items():
        print(k,v)
        
    
parse_scenes()