import pygame
import numpy as np


class Point(pygame.math.Vector2):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y 
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def dist(self, P):
        return np.sqrt((self.x - P.x)**2 + (self.y - P.y)**2)

def intersect_point_fixed(line1, line2):                    # ax + by + c
    # for line1
    vecdir1 = pygame.math.Vector2(line1[1].x-line1[0].x, line1[1].y-line1[0].y).normalize()
    a_1 = vecdir1.x
    b_1 = - vecdir1.y
    c_1 = -1 * ( a_1 * line1[0].x + b_1 * line1[0].y)
    
    print("lin1", a_1, b_1, c_1)
    
    # for line2
    vecdir2 = pygame.math.Vector2(line2[1].x-line2[0].x, line2[1].y-line2[0].y)
    vecdir2.normalize_ip()
    a_2 = vecdir2.x
    b_2 = - vecdir2.y
    c_2 = -1 * ( a_2 * line2[0].x + b_2 * line2[0].y)
    
    print("lin2", a_2, b_2, c_2)

    
    mat_coef = np.array([[a_1, b_1], [a_2, b_2]])
    res_c = np.array([c_1, c_2])
   
    
    mat_inv = np.linalg.inv(mat_coef)
    res = np.array([c_1, c_2])
    
    pt_final = np.matmul(mat_inv, res)
    
    print(mat_inv)
    print(res)
    
    print('pt fin', pt_final)
    
    
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


testp = Point(200, 300)
print(line_intersection([testp, Point(testp.x + 10000, testp.y)], [Point(1000,0), Point(800, 1000)]))