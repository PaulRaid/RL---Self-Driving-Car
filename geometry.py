import pygame 
import math 
from constants import *


class Point(pygame.math.Vector2):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y 
    
    def to_tuple(self):
        return (self.x, self.y)


# Code to check if segment AB and CD intersect

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def intersect(A,B,C,D):  # returns True if there is an intersection
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


# Code for rotations --> angle in radians

def rotate_point_around_center(center,point,angle):
    qx = center.x + math.cos(angle) * (point.x - center.x) - math.sin(angle) * (point.y - center.y)
    qy = center.y + math.sin(angle) * (point.x - center.x) + math.cos(angle) * (point.y - center.y)
    q = Point(qx, qy)
    return q

def rotate_rect(pt1, pt2, pt3, pt4, angle):

    pt_center = Point((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)

    pt1 = rotate_point_around_center(pt_center,pt1,angle)
    pt2 = rotate_point_around_center(pt_center,pt2,angle)
    pt3 = rotate_point_around_center(pt_center,pt3,angle)
    pt4 = rotate_point_around_center(pt_center,pt4,angle)

    return pt1, pt2, pt3, pt4