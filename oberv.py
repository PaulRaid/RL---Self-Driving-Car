import pygame
from constants import *
from geometry import *
import numpy as np

# The Rays going in all the directions from the car

class Vision():
    def __init__(self, x, y, vec) -> None:
        self.x = x
        self.y = y
        self.orig = Point(x,y)
        self.vec = vec              

    def coord_inf(self):     # Returns infinite point coords, parametric equation
        t=10000
        x_inf = self.x + t * self.v.x
        y_inf = self.x + t * self.v.y 
        return Point(x_inf, y_inf)

    def one_intersects(self, wall):    # returns true, iif ray intersects the wall
        inf = self.coord_inf()
        return intersect(self.orig, inf, wall.get_start(), wall.get_last())
    
    def track_intersection(self, walls):
        distances = [1000 for a in range(len(walls))]
        points = [Point(0,0) for a in distances]
