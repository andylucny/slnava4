import cv2
import numpy as np
from bs4 import BeautifulSoup
import time

with open(f'venue.txt','r') as f:
    venue = f.readlines()[0].strip()

# Create a black image
map = cv2.imread(f'{venue}/map.png')

with open(f'{venue}/map.osm', 'r', encoding="utf-8") as f:
    data = f.read()

bs = BeautifulSoup(data, "xml")

all = dict()
for node in bs.select("node"):
    id = int(node['id'])
    lat= float(node['lat'])
    lon= float(node['lon'])
    all[id] = (lat,lon)

# Draw points
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

for id in all:
    point = all[id]
    pixel = pin(point)
    cv2.circle(map, pixel, 2, (255, 0, 0), -1)

# Global variables
ids = []

def find_closest_refid(point):
    target_point = np.array(point)
    refids = list(all.keys())
    points = np.array(list(all.values()))
    distances = np.linalg.norm(points - target_point, axis=1)
    index = np.argmin(distances)
    return refids[index]

last = None
def draw_polygon(event, x, y, flags, param):
    global ids, last
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = (x, y)
        point = unpin(pixel)
        id = find_closest_refid(point)
        ids.append(id)
        x, y = pin(all[id])
        if last:
            cv2.line(map, last, (x,y), (0, 255, 255), 1)
        cv2.circle(map, (x, y), 3, (0, 0, 255), -1)
        last = (x,y)

cv2.namedWindow('Select Polygon')
cv2.setMouseCallback('Select Polygon', draw_polygon)

while True:
    cv2.imshow('Select Polygon', map)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == 'x':
        id = ids.pop()
        pixel = pin(id)
        last_pixel = pin(last)
        cv2.line(map, last_pixel, pixel, (180, 180, 180), 1)
        cv2.circle(map, pixel, 3, (0, 0, 255), -1)
        last = all[ids[-1]]

cv2.destroyAllWindows()

# Create the root XML structure
osm = BeautifulSoup('<?xml version="1.0" encoding="UTF-8"?><osm version="0.6" generator="CGImap 0.8.8 (717435 spike-06.openstreetmap.org)" copyright="OpenStreetMap and contributors" attribution="http://www.openstreetmap.org/copyright" license="http://opendatacommons.org/licenses/odbl/1-0/"></osm>', 'xml')

# Create the <way> tag with attributes
way_tag = osm.new_tag('way', id="33267920", visible="true", version="30", changeset="141273255", timestamp="2023-09-14T19:26:09Z", user="Tulkun", uid="18107650")

# Add <nd> tags for each ref ID
for refid in ids:
    nd_tag = osm.new_tag('nd', ref=str(refid))
    way_tag.append(nd_tag)

# Add <tag> elements
tags = {
    "alt_name": "Sad Andreja Kmeťa",
    "leisure": "park",
    "name": "Mestský park",
    "old_name": "Gottwaldove sady",
    "wikidata": "Q12775735",
    "wikipedia": "sk:Sad Andreja Kmeťa"
}

for k, v in tags.items():
    tag = osm.new_tag('tag', k=k, v=v)
    way_tag.append(tag)

# Append the <way> tag to the <osm> element
osm.osm.append(way_tag)

# Print the formatted XML

with open(f'{venue}/area-{int(time.time())}.xml', 'wt', encoding="utf-8") as f:
    f.write(osm.prettify())
