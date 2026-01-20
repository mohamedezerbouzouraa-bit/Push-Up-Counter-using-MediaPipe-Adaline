import math

def angle(a, b, c):
    bc = math.dist(b, c)
    ba = math.dist(b, a)
    ac = math.dist(a, c)
    return math.degrees(math.acos((bc**2 + ba**2 - ac**2) / (2 * bc * ba)))
#theoreme d'Al Kashi
