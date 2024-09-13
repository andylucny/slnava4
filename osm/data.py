import numpy as np
import cv2
from bs4 import BeautifulSoup
import os 

venue = 'buchlovice'
os.chdir(venue)

with open('map.osm', 'r', encoding="utf-8") as f:
    data = f.read()

bs = BeautifulSoup(data, "xml")

with open('area.xml', 'r', encoding="utf-8") as f:
    data2 = f.read()

bs2 = BeautifulSoup(data2, "xml")

refs = [ int(nd['ref']) for nd in bs2.select("nd") ]

#polygon = np.array([(112,28), (200,8), (282,16), (584,88), (816,320), (818,826), (802,984), (678,1284), (584,1284), (406,1186), (90,834), (60,724), (46,640), (40,594), (12,234), (28,50)])
#polygon2 = np.array([(180,796),(350,954),(426,1138),(342,1276),(84,836)])

#maxlat = float(bs.bounds['maxlat'])
#minlat = float(bs.bounds['minlat'])
#maxlon = float(bs.bounds['maxlon'])
#minlon = float(bs.bounds['minlon'])

maxlat, minlon = (48.59586361614979, 17.8339650310559)
minlat, maxlon = (48.58852669397308, 17.841023649068326)

def pin(point):
    lat, lon = point
    y = (map.shape[0]-1) - map.shape[0]*(lat-minlat)/(maxlat-minlat)
    x = map.shape[1]*(lon-minlon)/(maxlon-minlon)
    return (int(x), int(y))

#def unpin(point):
#    x, y = point
#    lat = ((map.shape[0]-1) - y)*(maxlat-minlat)/map.shape[0] + minlat
#    lon = x*(maxlon-minlon)/map.shape[1] + minlon
#    return (lat, lon)

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
    #x, y = pin((lat,lon))
    #if cv2.pointPolygonTest(polygon, (x,y), measureDist=True) > 0:
    #    if cv2.pointPolygonTest(polygon2, (x,y), measureDist=True) < 0:
    #        nodes[id] = (lat,lon)

map = cv2.imread('map.png')

for id in nodes.keys():
    _ = cv2.circle(map,pin(nodes[id]),1,(0,0,255),cv2.FILLED)

cv2.imwrite('nodes.png',map)

map = cv2.imread('map.png')

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

cv2.imwrite('ways.png',map)

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

map = cv2.imread('map.png')

for n1 in range(cnt-1):
    for n2 in range(n1+1,cnt):
        if mat[n1,n2] > 0:
            p1 = (lats[n1],lons[n1])
            p2 = (lats[n2],lons[n2])
            _ = cv2.line(map,pin(p1),pin(p2),(0,255,0),1)

for n in range(cnt):
    p = (lats[n],lons[n])
    _ = cv2.circle(map,pin(p),1,(255,0,255),cv2.FILLED)

cv2.imwrite('ways2.png',map)

print("map ready")

def localize(p):
    lat, lon = p
    distances = np.sqrt((lats-lat)**2 + (lons-lon)**2)
    n = np.argmin(distances)
    return n

def dijkstra(i,j):
    infinity = 10000000
    v = np.full((cnt),infinity,np.float32)
    b = np.full((cnt),True,bool)
    u = np.full((cnt),-1,int)
    v[i] = 0
    while v[j] == infinity:
        m = -1
        vv = infinity
        for n in range(cnt):
            if b[n]:
                if v[n] < vv:
                    m = n
                    vv = v[n]
        if m == -1:
            return []
        b[m] = False
        for n in range(cnt):
            if mat[m,n] > 0:
                if v[m]+mat[m,n] < v[n]:
                    v[n] = v[m]+mat[m,n]
                    u[n] = m
    ret = [j]
    n = j
    while u[n] != -1:
        ret.append(u[n])
        n = u[n]
    return ret

def findPath(p,q):
    pi = localize(p)
    qi = localize(q)
    pathi = dijkstra(pi,qi)
    return [ (lats[i],lons[i]) for i in pathi ]

s1 = (48.5934894,17.8365257) # fontana v parku
s2 = (48.5919247,17.8354146) # chlapec s rybkami
path = findPath(s1,s2)
print(path)

q = None
for p in path:
    if q is not None:
        _ = cv2.line(map,pin(q),pin(p),(255,255,0),3)
    _ = cv2.circle(map,pin(p),4,(255,0,0),cv2.FILLED)
    q = p

cv2.imwrite('path.png',map)
