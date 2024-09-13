import numpy as np
import cv2
from bs4 import BeautifulSoup
from sklearn.neighbors import NearestNeighbors

with open(f'venue.txt','r') as f:
    venue = f.readlines()[0].strip()

with open(f'{venue}/map.osm', 'r', encoding="utf-8") as f:
    data = f.read()

bs = BeautifulSoup(data, "xml")

with open(f'{venue}/area.xml', 'r', encoding="utf-8") as f:
    data2 = f.read()

bs2 = BeautifulSoup(data2, "xml")

refs = [ int(nd['ref']) for nd in bs2.select("nd") ]

with open(f'{venue}/range.xml', 'r', encoding="utf-8") as f:
    data3 = f.read()

bs3 = BeautifulSoup(data3, "xml")

maxlat = float(bs3.range['maxlat'])
minlon = float(bs3.range['minlon'])
minlat = float(bs3.range['minlat'])
maxlon = float(bs3.range['maxlon'])

def pin(point):
    lat, lon = point
    y = (map.shape[0]-1) - map.shape[0]*(lat-minlat)/(maxlat-minlat)
    x = map.shape[1]*(lon-minlon)/(maxlon-minlon)
    return (int(x), int(y))

def unpin(pixel):
    x, y = pixel
    lon = (x / map.shape[1]) * (maxlon - minlon) + minlon
    lat = (1 - (y / (map.shape[0] - 1))) * (maxlat - minlat) + minlat
    return (lat, lon)

all = dict()
for node in bs.select("node"):
    id = int(node['id'])
    lat= float(node['lat'])
    lon= float(node['lon'])
    all[id] = (lat,lon)
    
polygon = np.array([ all[ref] for ref in refs ],np.float32)
    
nodes = dict()
for node in bs.select("node"):
    id = int(node['id'])
    lat= float(node['lat'])
    lon= float(node['lon'])
    if cv2.pointPolygonTest(polygon, (lat,lon), measureDist=True) > 0:
        nodes[id] = (lat,lon)

map = cv2.imread(f'{venue}/map.png')

for id in nodes.keys():
    _ = cv2.circle(map,pin(nodes[id]),1,(0,0,255),cv2.FILLED)

cv2.imwrite(f'{venue}/nodes.png',map)

map = cv2.imread(f'{venue}/map.png')

ways = []
for way in bs.select("way"):
    nds = [ int(nd['ref']) for nd in way.select("nd") ]
    tags = [ tag['k'] for tag in way.select('tag') ]
    if (("highway" in tags) or ("footage" in tags)) and not ("barrier" in tags) and not("indoor" in tags):
        valid_nds = []
        for nd in nds:
            if nd in nodes.keys():
                valid_nds.append(nd)
        if len(valid_nds) > 1:
            ways.append(valid_nds)

for way in ways:
    prev = None
    for nd in way:
        if prev is not None:
            _ = cv2.line(map,pin(nodes[prev]),pin(nodes[nd]),(0,255,0),1)
        prev = nd

cnt = 0
indices = dict()
for way in ways:
    for nd in way:
        indices[nd] = cnt
        cnt += 1
        _ = cv2.circle(map,pin(nodes[nd]),1,(255,0,255),cv2.FILLED)

cv2.imwrite(f'{venue}/ways.png',map)

mat = -np.ones((cnt,cnt),np.float32)
lats = np.zeros((cnt),np.float32)
lons = np.zeros((cnt),np.float32)
for way in ways:
    prev = None
    for nd in way:
        lat, lon = nodes[nd]
        index2 = indices[nd]
        lats[index2] = lat
        lons[index2] = lon
        if prev is not None:
            index1 = indices[prev]
            distance = np.sqrt((lats[index2]-lats[index1])**2 + (lons[index2]-lons[index1])**2)
            mat[index1,index2] = distance
            mat[index2,index1] = distance
        prev = nd

map = cv2.imread(f'{venue}/map.png')

for n1 in range(cnt-1):
    for n2 in range(n1+1,cnt):
        if mat[n1,n2] > 0:
            p1 = (lats[n1],lons[n1])
            p2 = (lats[n2],lons[n2])
            _ = cv2.line(map,pin(p1),pin(p2),(0,255,0),1)

for n in range(cnt):
    p = (lats[n],lons[n])
    _ = cv2.circle(map,pin(p),1,(255,0,255),cv2.FILLED)

cv2.imwrite(f'{venue}/ways2.png',map)

print("map ready")
