import pygame 
import math
import numpy as np
from constants import *


class Point(pygame.math.Vector2):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y 
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def dist(self, P):
        return np.sqrt((self.x - P.x)**2 + (self.y - P.y)**2)


# Code to check if segment AB and CD intersect

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def intersect(A,B,C,D):  # returns True if there is an intersection
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def slope(p1, p2) :
    min_allowed = 1e-5   # guard against overflow
    big_value = 1e10
    if (p2.x - p1.x) < min_allowed:
        return 2
    else: 
        return (p2.y - p1.y) * 1. / (p2.x - p1.x)
   
def y_intercept(slope, p1) :
   return p1.y - 1. * slope * p1.x
   
def intersect_point(line1, line2) :
   min_allowed = 1e-5   # guard against overflow
   big_value = 1e10     # use instead (if overflow would have occurred)
   
   m1 = slope(line1[0], line1[1])
   b1 = y_intercept(m1, line1[0])
   m2 = slope(line2[0], line2[1])
   b2 = y_intercept(m2, line2[0])

   if abs(m1 - m2) < min_allowed :
      x = big_value
   else :
      x = (b2 - b1) / (m1 - m2)
   y = m1 * x + b1
   y2 = m2 * x + b2

   return Point(x,y)     # returns intersection point between 2 lines --> To exectute only if we have the existancy within the segment

# Code for rotations --> angle in radians

def rotate_point_around_center(center,point,angle_):
    angle = math.radians(angle_)
    qx = center.x + math.cos(angle) * (point.x - center.x) + math.sin(angle) * (point.y - center.y)
    qy = center.y + math.sin(angle) * (point.x - center.x) - math.cos(angle) * (point.y - center.y)
    q = Point(qx, qy)
    return q

def rotate_rect(pt1, pt2, pt3, pt4, angle):

    pt_center = Point((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)

    pt1 = rotate_point_around_center(pt_center,pt1,angle)
    pt2 = rotate_point_around_center(pt_center,pt2,angle)
    pt3 = rotate_point_around_center(pt_center,pt3,angle)
    pt4 = rotate_point_around_center(pt_center,pt4,angle)

    return pt1, pt2, pt3, pt4