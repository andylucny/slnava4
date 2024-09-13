import numpy as np
import cv2
from bs4 import BeautifulSoup

map = cv2.imread('map-org.png') 

with open('map.osm', 'r', encoding="utf-8") as f:
    data = f.read()

bs = BeautifulSoup(data, "xml")

maxlat = float(bs.bounds['maxlat'])
minlat = float(bs.bounds['minlat'])
maxlon = float(bs.bounds['maxlon'])
minlon = float(bs.bounds['minlon'])

def pin(point,offset,shape):
    lat, lon = point
    y = (shape[0]-1) - shape[0]*(lat-minlat)/(maxlat-minlat) + offset[0]
    x = shape[1]*(lon-minlon)/(maxlon-minlon) + offset[1]
    return (int(x), int(y))

nodes = dict()
for node in bs.select("node"):
    id = int(node['id'])
    lat= float(node['lat'])
    lon= float(node['lon'])
    nodes[id] = (lat,lon)
       
#offset = [0,0]
#shape = list(map.shape[:2])
#offset=[2, 436]
#shape=[1713, 644]
offset=[5, 436]
shape=[1709, 644]
while True:
    disp_map = np.copy(map)
    
    for id in nodes.keys():
        point = pin(nodes[id],offset,shape)
        cv2.circle(disp_map,point,1,(0,0,255),cv2.FILLED)
    
    position = (48.5934894,17.8365257) # fontana v parku
    point = pin(position,offset,shape)
    cv2.circle(disp_map,point,2,(255,0,255),cv2.FILLED)
    
    rect = (offset[1],offset[0],shape[1],shape[0])
    cv2.rectangle(disp_map,rect,(0,255,0),2)
    
    cv2.imshow("map",cv2.resize(disp_map,(disp_map.shape[1]//2,disp_map.shape[0]//2)))
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('a'):
        offset[1] -= 1
    elif key == ord('s'):
        offset[1] += 1
    elif key == ord('w'):
        offset[0] -= 1
    elif key == ord('z'):
        offset[0] += 1
    elif key == ord('j'):
        shape[1] -= 1
    elif key == ord('k'):
        shape[1] += 1
    elif key == ord('i'):
        shape[0] -= 1
    elif key == ord('m'):
        shape[0] += 1

print('offset',offset)
print('shape',shape)

roi = (204, 246, 825, 1290)
tl = (roi[0],roi[1])
br = (roi[0]+roi[2],roi[1]+roi[3])

def unpin(point,offset,shape):
    x, y = point
    lat = ((shape[0]-1) - (y - offset[0]))*(maxlat-minlat)/shape[0] + minlat
    lon = (x - offset[1])*(maxlon-minlon)/shape[1] + minlon
    return (lat, lon)

print(unpin(tl,offset,shape))
print(unpin(br,offset,shape))

map_final = map[tl[1]:br[1],tl[0]:br[0]]
cv2.imwrite('map.png',map_final)
