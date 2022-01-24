from cv2 import rotate
import pygame
from constants import * 
import pygame.math
from geometry import *
from oberv import *

class Car():
    def __init__(self, screen, track):
        
        self.screen = screen
        self.track = track

        #self.rect = pygame.Rect((INIT_POS.x, INIT_POS.y), (2*SIZE, SIZE))
        self.image_base = pygame.Surface((2*SIZE, SIZE))
        self.image_base.set_colorkey(BLACK)
        self.image_base.fill(RED)
        self.image = self.image_base.copy()
        self.image.set_colorkey(BLACK)

        self.rect = self.image.get_rect()
        self.rect.x = INIT_POS.x
        self.rect.y = INIT_POS.y
        

        self.p1 = self.rect.topleft
        self.p2 = self.rect.topright
        self.p3 = self.rect.bottomright
        self.p4 = self.rect.bottomleft

        self.angle = 0

        # Position and initial speed
        # Locally
        self.xspeed = 1
        self.yspeed = 0
        self.velvalue = pygame.math.Vector2.length(Point(self.xspeed, self.yspeed))
        self.accel = 1
        self.direction_vector = pygame.math.Vector2(1,0)
        self.as_turned = False

        self.rays = self.set_rays()    # array

    # method to change the class parameters of the car
    def modify(self, event):                          
        if event.key == pygame.K_LEFT:
            self.rect.move_ip(-20, 0)
        if event.key == pygame.K_RIGHT:
            self.rect.move_ip(20, 0)
        if event.key == pygame.K_UP:
            self.rect.move_ip(0, -20)
        if event.key == pygame.K_DOWN:
            self.rect.move_ip(0, 20)
        if event.key == pygame.K_r:
            self.rect.x = INIT_POS.x
            self.rect.y = INIT_POS.y
        if event.key == pygame.K_p:  # Turn Left
            self.turn()
            self.rot()
            #pygame.quit()
        if event.key == pygame.K_o:   # Turn Right
            self.turn(-1)
            self.rot()
            #pygame.quit()


    # method to actually move the position of the car
    def update(self):

        self.rays = self.set_rays()
        self.replace()
    
    def turn(self, dir = 1):

        prev_angle = self.angle

        self.angle = (self.angle + dir * ANGLE ) % 360
        print('angle', self.angle)
        print("dirvect", self.direction_vector)
        self.direction_vector = self.direction_vector.rotate_rad(math.radians(-1*dir*ANGLE))
        print("dirvect", self.direction_vector)
        print(" ")
    
    def rot(self):
        
        angle_ = self.angle
        nouv_vel = rotate_point_around_center(Point(0,0), Point(0, self.velvalue), angle_)
        self.xspeed, self.yspeed = nouv_vel.to_tuple()

        nouv_corner_1 = Point(self.rect.topleft[0] + self.xspeed, self.rect.topleft[1] + self.yspeed)
        nouv_corner_2 = Point(self.rect.topright[0] + self.xspeed, self.rect.topright[1] + self.yspeed)
        nouv_corner_3 = Point(self.rect.bottomright[0] + self.xspeed, self.rect.bottomright[1] + self.yspeed)
        nouv_corner_4 = Point(self.rect.bottomleft[0] + self.xspeed, self.rect.bottomleft[1] + self.yspeed)

        self.p1, self.p2, self.p3, self.p4 = rotate_rect(nouv_corner_1, nouv_corner_2, nouv_corner_3, nouv_corner_4, angle_)

        old_center = self.rect.center

        new_image = pygame.transform.rotate(self.image_base, angle_)
        
        self.rect = new_image.get_rect()

        self.rect.center = old_center

        self.image = new_image.copy()
    
    
    # method to check if the position is authorised and replace it
    def replace(self):                              
        if not self.is_position_valid(): 
            new_image = pygame.transform.rotate(self.image_base, 0)
            self.rect = new_image.get_rect()
            self.rect.x = INIT_POS.x
            self.rect.y = INIT_POS.y
            self.image = new_image.copy()
            self.angle = 0
            self.as_turned = False

    def is_position_valid(self):
        x = self.rect.x
        y = self.rect.y
        dirx = self.direction_vector[0]
        diry = self.direction_vector[1]

        # To remember: the perpendicular clockwith (with our axes config) to vector (x, y) is (-y, x)

        s = SIZE

        #up = [(x,y), (x + 2 * s * dirx, y + s * diry)]
        #down = [(x - diry * s,y + dirx * s), (x - diry * s + 2 * s * dirx, y + dirx * s + s * diry)]
        #left = [(x,y), (x - diry * s,y + dirx * s)]
        #right = [(x + 2 * s * dirx, y + s * diry), (x - diry * s + 2 * s * dirx, y + dirx * s + s * diry)]

        up = [self.rect.topleft, self.rect.topright]
        down = [self.rect.bottomleft, self.rect.bottomright]
        left = [self.rect.topleft, self.rect.bottomleft]
        right = [self.rect.topright, self.rect.bottomright]
        sides = [up, down,left,right]
        res = True
        for a in sides:
            for wall in self.track.get_walls():
                p1 = Point(a[0][0], a[0][1])
                p2 = Point(a[1][0], a[1][1])
                res = res and not intersect(p1, p2, wall.get_start(), wall.get_last())
        return res

    def draw_car(self, screen):
        screen.blit(self.image, self.rect)
        for viz in self.rays:
            point, dist = viz.track_intersection(self.track.get_walls())
            viz.dray_ray(point, screen)
        
    def print_car(self):
        print(self.p1)
        print(self.p2)
        print(self.p3)
        print(self.p4)
        print(" ")
    
    def set_rays(self):
        front = Vision(self.rect.centerx, self.rect.centery, self.direction_vector)
        #back = Vision(self.rect.centerx, self.rect.centery, -self.direction_vector)
        return [front]#, back]