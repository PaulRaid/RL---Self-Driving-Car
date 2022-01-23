import pygame
from sqlalchemy import true
from constants import * 
import pygame.math

# The Rays going in all the directions from the car

class Vision():
    def __init__(self, x1, y1, angle):
        self.x1 = x1
        self.y1 = y1
        self.angle = angle

# code to check if segment AB and CD intersect

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):  # returns True if there is an intersection
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


class Car(pygame.sprite.Sprite):
    def __init__(self, screen, track):
        super().__init__()
        
        self.screen = screen
        self.track = track

        self.rect = pygame.Rect(INIT_POS, (2*SIZE, SIZE))
        self.image = pygame.Surface((2*SIZE, SIZE))
        self.image.fill(RED)


        # Position and initial speed
        self.v = (0,0)
        self.pos = ((340, 240))
        self.vel = (0,0)
        self.acc = (0,0)
        self.direction_vector = (1,0)


    def move(self, event):
        if event.key == pygame.K_LEFT:
            self.rect.move_ip(-20, 0)
            self.replace()
        if event.key == pygame.K_RIGHT:
            self.rect.move_ip(20, 0)
            self.replace()
        if event.key == pygame.K_UP:
            self.rect.move_ip(0, -20)
            self.replace()
        if event.key == pygame.K_DOWN:
            self.rect.move_ip(0, 20)
            self.replace()
        if event.key == pygame.K_r:
            self.rect.x = INIT_POS[0]
            self.rect.y = INIT_POS[1]
            self.replace()
        if event.key == pygame.K_a:
            angle = 45
            orig_rect = self.image.get_rect()
            self.image = pygame.transform.rotate(self.image, angle)
            rot_rect = orig_rect.copy()
            rot_rect.center = self.image.get_rect().center
            self.image = self.image.subsurface(rot_rect).copy()
            self.replace()

    def update(self):
        pass
    
    def attack(self):
        pass
    
    def jump(self):
        pass
    
    def replace(self):
        if not self.is_position_valid():
            self.rect.x = INIT_POS[0]
            self.rect.y = INIT_POS[1]

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
                res = res and not intersect(a[0], a[1], wall.get_start(), wall.get_last())
        return res

    def draw_car(self):
        self.screen.blit(self.image, self.rect)


    