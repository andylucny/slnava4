import numpy as np
import cv2
from bs4 import BeautifulSoup #pip install beautifulsoup4 lxml

#https://www.openstreetmap.org/#map=17/48.59238/17.83871
#https://www.openstreetmap.org/way/33267920#map=17/48.59238/17.83871

with open('map.osm', 'r', encoding="utf-8") as f:
    data = f.read()

bs = BeautifulSoup(data, "xml")

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

points = []
with open('../gps/gps.txt', 'r') as f:
    for line in f:
        values = line.strip().split(',')
        if len(values) <= 1:
            break
        if values[1] == -1:
            continue
        points.append((float(values[0]),float(values[1])))

#tl = (48.59475,17.83234)
#br = (48.59001,17.84507)

map = cv2.imread('ways2.png')

disp_map = np.copy(map)
for position in points:
    point = pin(position)
    cv2.circle(disp_map,point,2,(0,0,255),cv2.FILLED)

position = (48.5934894,17.8365257) # fontana v parku
point = pin(position)
cv2.circle(disp_map,point,2,(255,0,255),cv2.FILLED)
   
cv2.imwrite('trace.png',disp_map)