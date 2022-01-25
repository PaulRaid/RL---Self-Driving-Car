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
        

        self.p1 = Point(self.rect.topleft[0], self.rect.topleft[1])
        self.p2 = Point(self.rect.topright[0], self.rect.topright[1])
        self.p3 = Point(self.rect.bottomright[0], self.rect.bottomright[1])
        self.p4 = Point(self.rect.bottomleft[0], self.rect.bottomleft[1])

        self.angle = 0
        self.direction_vector = pygame.math.Vector2(1,0)
        
        self.xspeed = INIT_VEL
        self.yspeed = 0
        self.velvalue = pygame.math.Vector2.length(Point(self.xspeed, self.yspeed))
        self.accel = 1
        
        self.rays = self.set_rays()    # array

    # method to change the class parameters of the car
    def modify(self, event):   
        vec = self.direction_vector.normalize()
        orth = pygame.math.Vector2(vec.y, -vec.x)                       
        if event.key == pygame.K_LEFT:
            self.rect.move_ip(-20*vec.x, -20*vec.y)
        if event.key == pygame.K_RIGHT:
            self.rect.move_ip(20*vec.x, 20*vec.y)
        if event.key == pygame.K_UP:
            self.rect.move_ip(20*orth.x, 20*orth.y)
        if event.key == pygame.K_DOWN:
            self.rect.move_ip(-20*orth.x, -20*orth.y)
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
            vec = self.direction_vector.normalize()
            #pygame.quit()
        if event.key == pygame.K_0:     # action 0 : doing nothing
            pass
        if event.key == pygame.K_1:  # action 1 : accel
            self.accelerate()
            pass
        if event.key == pygame.K_2:  # action 1 : deccelerate
            self.accelerate(-1)
            #pass


    def accelerate(self, dir = 1):
        
        if dir == 1:
            self.velvalue = 2 * self.velvalue
        else:
            self.velvalue = self.velvalue/2
        #pass

    # method to actually move the position of the car
    def update(self):
        vec = self.direction_vector.normalize()
        
        
        vec  = clamp_close_number(vec)
        
        vite = self.velvalue
        decalx = arrondiSup(vite*vec.x)
        decaly = arrondiSup(vite*vec.y)
        
        self.rect.center = (self.rect.centerx + decalx , self.rect.centery + decaly)
        self.rays = self.set_rays()
        self.replace()
    
    def turn(self, dir = 1):
        self.angle = (self.angle + dir * ANGLE ) % 360
        self.direction_vector = self.direction_vector.rotate(-1*dir*ANGLE)
    
    def rot(self):
        
        angle_ = self.angle
        nouv_vel = rotate_point_around_center(Point(0,0), Point(0, self.velvalue), angle_)
        self.xspeed, self.yspeed = nouv_vel.to_tuple()

        '''
        nouv_corner_1 = Point(self.rect.topleft[0] + self.xspeed, self.rect.topleft[1] + self.yspeed)
        nouv_corner_2 = Point(self.rect.topright[0] + self.xspeed, self.rect.topright[1] + self.yspeed)
        nouv_corner_3 = Point(self.rect.bottomright[0] + self.xspeed, self.rect.bottomright[1] + self.yspeed)
        nouv_corner_4 = Point(self.rect.bottomleft[0] + self.xspeed, self.rect.bottomleft[1] + self.yspeed)

        self.p1, self.p2, self.p3, self.p4 = rotate_rect(nouv_corner_1, nouv_corner_2, nouv_corner_3, nouv_corner_4, angle_)
        '''

        old_center = self.rect.center

        new_image = pygame.transform.rotate(self.image_base, angle_)
        
        self.rect = new_image.get_rect()

        self.rect.center = old_center

        self.image = new_image.copy()
    
    
    # method to check if the position is authorised and replace it
    def replace(self):                              
        if not self.is_position_valid(): 
            
            # -- Reset Object
            
            new_image = pygame.transform.rotate(self.image_base, 0)
            self.rect = new_image.get_rect()
            self.rect.x = INIT_POS.x
            self.rect.y = INIT_POS.y
            self.image = new_image.copy()
            
            # -- Reset Position
            self.angle = 0
            self.direction_vector = pygame.math.Vector2(1,0)
            self.p1 = Point(self.rect.topleft[0], self.rect.topleft[1])
            self.p2 = Point(self.rect.topright[0], self.rect.topright[1])
            self.p3 = Point(self.rect.bottomright[0], self.rect.bottomright[1])
            self.p4 = Point(self.rect.bottomleft[0], self.rect.bottomleft[1])
            
            # -- Reset Velocity 
            
            self.xspeed = INIT_VEL
            self.yspeed = 0
            self.velvalue = pygame.math.Vector2.length(Point(self.xspeed, self.yspeed))
            self.accel = 1

    def is_position_valid(self):
        x = self.rect.x
        y = self.rect.y
        dirx = self.direction_vector[0]
        diry = self.direction_vector[1]

        # To remember: the perpendicular clockwith (with our axes config) to vector (x, y) is (-y, x)

        s = SIZE

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
            if DRAW_RAYS:
                viz.dray_ray(point, screen)
        
    def print_car(self):
        print(self.p1)
        print(self.p2)
        print(self.p3)
        print(self.p4)
        print(" ")
    
    def set_rays(self):
        u = self.direction_vector.normalize()
        v = pygame.math.Vector2(u.y, -u.x)    # Anticlock wise
        s = SIZE
        u = s*u
        v = s*v/2
        
        res = []
        
        res.append(Vision(self.rect.centerx + u.x, self.rect.centery + u.y, self.direction_vector)) #front
        res.append(Vision(self.rect.centerx - u.x, self.rect.centery - u.y, -self.direction_vector)) #back
        
        res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector)) # front 0
        res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector.rotate(-15))) # front up 30
        res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector.rotate(-30))) # front up 30
        res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector.rotate(-60))) # front up 60
        res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector.rotate(15))) # front up down 15
        
        
        res.append(Vision(self.rect.centerx + v.x, self.rect.centery + v.y, v)) # side up 90
        res.append(Vision(self.rect.centerx - v.x, self.rect.centery - v.y, -v)) # side down 90
        
        
        res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector)) # down 
        res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector.rotate(15))) # front down  30
        res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector.rotate(30))) # front down  30
        res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector.rotate(60))) # front down  60
        res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector.rotate(-15))) # front down up 60
        
        
        return res