import numpy as np

def dgnorm(d):
    if d <= -180:
        d += 360
    elif d > 180:
        d -= 360
    return d

def vector2azimuth(vector):
    return np.degrees(np.arctan2(*list(vector)))

def dgaverage(values):
    if len(values) == 0:
        return None
    return vector2azimuth(np.average(np.array([np.sin(np.radians(values)),np.cos(np.radians(values))]).T,axis=0))

def dgdiff(a,b):
    return dgnorm(b-a)

def dgin(dg,fromdg,todg):
    if todg < fromdg:
        todg += 360
    if fromdg <= dg and dg <= todg:
        return True
    if fromdg <= dg+360 and dg+360 <= todg:
        return True
    if fromdg <= dg-360 and dg-360 <= todg:
        return True
    return False