import math
import numpy as np
import cv2
from bs4 import BeautifulSoup
from agentspace import Agent, space, Trigger
from sklearn.neighbors import NearestNeighbors
import time
from dgmath import vector2azimuth, dgaverage, dgdiff

with open(f'osm/venue.txt','r') as f:
    venue = f.readlines()[0].strip()

with open(f'osm/{venue}/map.osm', 'r', encoding="utf-8") as f:
    data = f.read()

bs = BeautifulSoup(data, "xml")

with open(f'osm/{venue}/area.xml', 'r', encoding="utf-8") as f:
    data2 = f.read()

bs2 = BeautifulSoup(data2, "xml")

refs = [ int(nd['ref']) for nd in bs2.select("nd") ]

with open(f'osm/{venue}/range.xml', 'r', encoding="utf-8") as f:
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

cnt = 0
indices = dict()
for way in ways:
    for nd in way:
        indices[nd] = cnt
        cnt += 1

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

print("map ready")

def distance_point_to_segment(point, segment_start, segment_end):
    # Calculate the vector from the segment's starting point to the given point
    vector_to_point = (point[0] - segment_start[0], point[1] - segment_start[1])
    # Calculate the vector representing the line segment
    segment_vector = (segment_end[0] - segment_start[0], segment_end[1] - segment_start[1])
    # Calculate the dot product of vector_to_point and segment_vector
    dot_product = vector_to_point[0] * segment_vector[0] + vector_to_point[1] * segment_vector[1]
    # Calculate the squared length of segment_vector
    segment_length_squared = segment_vector[0] ** 2 + segment_vector[1] ** 2
    # Calculate the parameter t for the projection onto the line segment
    t = dot_product / segment_length_squared
    if t < 0:
        # The projected point is before the segment
        distance = np.sqrt((point[0] - segment_start[0]) ** 2 + (point[1] - segment_start[1]) ** 2)
        closest_point = segment_start
        t = 0
    elif t > 1:
        # The projected point is after the segment
        distance = np.sqrt((point[0] - segment_end[0]) ** 2 + (point[1] - segment_end[1]) ** 2)
        closest_point = segment_end
        t = 1
    else:
        # The projected point is on the segment
        projected_point = (segment_start[0] + t * segment_vector[0], segment_start[1] + t * segment_vector[1])
        distance = np.sqrt((point[0] - projected_point[0]) ** 2 + (point[1] - projected_point[1]) ** 2)
        closest_point = (segment_start[0] + t * segment_vector[0], segment_start[1] + t * segment_vector[1])
        
    return distance, closest_point, t

def localize_on_way(point):
    optimal_distance = 1e10
    optimal_point = point
    for way in ways:
        prev = None
        for nd in way:
            if prev is not None:
                distance, closest_point, t = distance_point_to_segment(point, nodes[prev], nodes[nd])
                if distance < optimal_distance:
                    optimal_distance = distance
                    optimal_point = closest_point
                    optimal_nodes = (prev,nd,t)
            prev = nd
    return optimal_point, optimal_distance, optimal_nodes

def icp(a, b, pivot=(0,0), angle=0, iterations = 13):
    src = np.array(a,np.float32)
    dst = np.array(b,np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(angle),-np.sin(angle),pivot[0]],
                   [np.sin(angle), np.cos(angle),pivot[1]],
                   [            0,            0,        1]])

    src = src @ Tr[:2,:2].T + Tr[:2,2]

    for i in range(iterations):
        #Find the nearest neighbours between the current source and the destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
        distances, indices = nbrs.kneighbors(src)

        #Compute the transformation between the current source and destination cloudpoint
        T = cv2.estimateAffinePartial2D(src, dst[indices.T])[0]
        if T is None:
            return np.array([[1,0,0],[0,1,0]],np.float32)
        #Transform the previous source and update the current source cloudpoint
        src = src @ T[:,:2].T + T[:,2]
        #Save the transformation from the actual source cloudpoint to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
        
    return Tr[0:2]

dlat = (maxlat-minlat)/400
dlon = (maxlon-minlon)/200
dl = np.sqrt(dlat**2+dlon**2)

cloud = []
for n1 in range(cnt-1):
    for n2 in range(n1+1,cnt):
        if mat[n1,n2] > 0:
            p1 = np.array([lats[n1],lons[n1]],np.float32)
            p2 = np.array([lats[n2],lons[n2]],np.float32)
            segment = p2 - p1
            sampling = int(mat[n1,n2]/dl)
            for k in range(sampling):
                node = tuple(p1 + k * (p2 - p1) / sampling)
                cloud.append(node)
            cloud.append(p2)

map = cv2.imread(f'osm/{venue}/ways2.png')

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
    return ret[::-1]

def findPath(p,q):
    pi = localize(p)
    qi = localize(q)
    pathi = dijkstra(pi,qi)
    return [ (lats[i],lons[i]) for i in pathi ]
    
def haversine(point1, point2): # [dg] [dg]
    coord1 = np.array(point1).T
    coord2 = np.array(point2).T
    # Radius of the Earth in meters
    R = 6371000  
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(coord1[0])
    lon1 = np.radians(coord1[1])
    lat2 = np.radians(coord2[0])
    lon2 = np.radians(coord2[1])
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # Distance in meters
    distance = R * c
    return distance # [m]

def point_to_segment_distance(A, P0, P1):
    A = np.array(A)
    P0 = np.array(P0)
    P1 = np.array(P1)
    v = P1 - P0
    w = A - P0
    t = np.dot(w, v) / np.dot(v, v) # Project point A onto the line
    t_clamped = max(0, min(1, t)) # Clamp t to [0, 1] to keep it within the segment
    C = P0 + t_clamped * v # Find the closest point on the segment
    return np.linalg.norm(A - C) # Return the distance from A to the closest point on the segment
    
class OsmAgent(Agent):

    # Kalman filter initialization
    def initialize_kalman(self, gps_point):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurement variables
        # Transition matrix (A)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        # Measurement matrix (H)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        # Initialize state to zero
        self.kalman.statePost = np.array([gps_point[0], gps_point[1], 0.0, 0.0], np.float32)
        self.ticks = cv2.getTickCount()

    # Kalman filtering and speed estimation function
    def update_kalman(self, curr_point):
        ticks = cv2.getTickCount()
        dt = (ticks - self.ticks) / cv2.getTickFrequency()
        self.ticks = ticks
        self.kalman.transitionMatrix = np.array([[1, 0, dt, 0],
                                                 [0, 1, 0, dt],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        # Kalman prediction
        prediction = self.kalman.predict()
        # Measurement (latitude and longitude)
        measurement = np.array([[curr_point[0]],[curr_point[1]]], np.float32)
        # Kalman correction with current measurement
        self.kalman.correct(measurement)
        # Get current estimated state [lat, lon, speed_lat, speed_lon]
        estimated_state = self.kalman.statePost
        # Store speed vector
        estimated_speed = (estimated_state[2], estimated_state[3])  # [speed_lat, speed_lon]
        return estimated_speed

    def findPointOnMap(self,point):
        #print(point)
        if len(self.queue) == 0:
            self.initialize_kalman(point)
        else:
            self.speed_vector = self.update_kalman(point)
        self.queue.append(point)
        if len(self.queue) > 201:
            self.queue = self.queue[-200:]
        if len(self.queue) < 3:
            return point
        pivot = (0,0) if self.last_point is None else (self.last_point[0]-point[0],self.last_point[1]-point[1])
        T = icp(self.queue, cloud, pivot=pivot, angle=0, iterations = 13)
        error = np.linalg.norm(T[:,2])
        if error <= 5.0:
            final_point = tuple(np.array(point)@T[:,:2].T + T[:,2])
        else:
            final_point = point
        map_point,_,nodeinfo = localize_on_way(final_point)
        self.last_point = map_point
        return map_point

    def getHeading(self):
        if self.speed_vector:
            return vector2azimuth(self.speed_vector)
        else:
            return None

    def __init__(self,gpsName,qrName,goalName,simulation=False):
        self.gpsName = gpsName
        self.qrName = qrName
        self.goalName = goalName
        self.simulation = simulation
        self.speed_vector = None
        super().__init__()

    def init(self):
        self.last_point = None
        self.queue = []
        self.simulation_active = self.simulation
        self.position = None
        self.goal = None
        self.path = []
        self.last_distance = 0
        cv2.namedWindow("map")
        cv2.setMouseCallback("map", self.mouseHandler)
        space.attach_trigger(self.qrName,self,Trigger.NAMES)
        space.attach_trigger(self.gpsName,self,Trigger.NAMES)
        if self.simulation:
            self.attach_timer(0.5)

    def mouseHandler(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            pixel = (x, y)
            point = unpin(pixel)
            space[self.qrName] = point
            print('new qr', point)
        elif event == cv2.EVENT_LBUTTONDOWN:
            pixel = (x, y)
            point = unpin(pixel)
            if self.simulation and self.simulation_active:
                space['gps'] = point
            distance = haversine(point,self.path[0]) if len(self.path) > 0 else -1
            print(f'({point[0]:.7f},{point[1]:.7f}) {distance:.2f}m')

    def keyHandler(self, key):
        if key == ord('a'):
            self.simulation_active = not self.simulation_active
            print('simulation:','on' if self.simulation_active else 'off')
            
    def senseSelectAct(self):
        name = self.triggered()
        
        gps = None
        real_absolute_azimuth = 0
        wished_absolute_azimuth = 0

        if name == self.qrName: # a new goal has come
            self.goal = space[self.qrName]
            print('goal',self.goal) # (48.1491242,17.0737278)
            if self.goal is not None and self.position is not None:
                self.path = findPath(self.position,self.goal)
                if len(self.path) > 1:
                    if haversine(self.position,self.path[1]) < haversine(self.path[0],self.path[1]):
                        self.path = self.path[1:]

        elif name == self.gpsName:
            gps = space[self.gpsName]
            self.position = self.findPointOnMap(gps)
            
            if self.goal is not None:
                
                affinities = []
                for i in range(len(self.path) - 1):
                    distance_dg = point_to_segment_distance(self.position, self.path[i], self.path[i+1])
                    affinities.append(distance_dg)
        
                reached = False
                if len(affinities) > 0:
                    i = np.argmin(affinities)
                    distance1 = haversine(self.position,self.path[i+1])
                    distance = haversine(self.path[i],self.path[i+1])
                    reached = False
                    if distance1 < distance:
                        print(f' subgoal {self.path[i]} reached')
                        reached = True
                else:
                    i = 0
                    distance1 = haversine(self.position,self.path[i])
                    if distance1 < 1.5:
                        reached = True

                if reached:
                    i += 1
                elif self.last_distance < distance1:
                    print('replaning')
                    self.path = findPath(self.position,self.goal)
                    if len(self.path) > 1:
                        if haversine(self.position,self.path[1]) < haversine(self.path[0],self.path[1]):
                            self.path = self.path[1:]
                    
                self.path = self.path[i:]
                if len(self.path) == 0:
                    print(f'goal {self.goal} reached')
                    self.goal = None
                    self.last_distance = 0
                else:
                    self.last_distance = distance1
                
                if self.goal is not None:
                    real_absolute_azimuth = space(default=self.getHeading())[self.gpsName+'-azimuth']
                    wished_absolute_azimuth = vector2azimuth(np.array(self.path[0])-np.array(self.position))
                    rel_azimuth = dgdiff(wished_absolute_azimuth,real_absolute_azimuth)
                    # 0..forward <0..left >0..right
                    space(validity=1.0)[self.goalName] = rel_azimuth
                
        disp = np.copy(map)
        lastp = None
        for pi, p in enumerate(self.path):
            if lastp is not None:
                cv2.line(disp,pin(lastp),pin(p),(255,255,0),2)
            color = (255,0,0) if pi == 0 else (0,255,0) if pi == len(self.path)-1 else (255,255,0)
            cv2.circle(disp,pin(p),4,color,cv2.FILLED)
            last = p
        
        if self.position is not None:
            cv2.circle(disp,pin(self.position),3,(0,0,255),cv2.FILLED)
        if gps is not None:
            cv2.circle(disp,pin(gps),2,(0,255,255),cv2.FILLED)
        if len(self.path) > 0:
            cv2.circle(disp,pin(self.path[0]),3,(255,0,255),cv2.FILLED)
        cv2.putText(disp,f'{self.last_distance:.2f}m {real_absolute_azimuth:.0f}dg {wished_absolute_azimuth:.0f}dg',(5,25),0,1.0,(0,0,0),2)
        cv2.imshow("map",disp)
        key = cv2.waitKey(10)
        self.keyHandler(key)
        #if len(path) > 0:
        #    print('distance to subgoal',self.subgoal,'=',self.distance, heading)

if __name__ == '__main__':
    a = OsmAgent('gps','qr','goal',simulation=True)
    time.sleep(2)
    space['gps'] = (48.591787, 17.836143) # start: kino podla OSM
    space['qr'] = (48.5934894, 17.8365257) # ciel: fontanka podla OSM
