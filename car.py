import pygame
from constants import * 
import pygame.math
from geometry import *

# The Rays going in all the directions from the car

class Vision():
    def __init__(self, x1, y1, angle):
        self.x1 = x1
        self.y1 = y1
        self.angle = angle


class Car(pygame.sprite.Sprite):
    def __init__(self, screen, track):
        super().__init__()
        
        self.screen = screen
        self.track = track

        self.rect = pygame.Rect((INIT_POS.x, INIT_POS.y), (2*SIZE, SIZE))
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
            self.rect.x = INIT_POS.x
            self.rect.y = INIT_POS.y
            self.replace()
        if event.key == pygame.K_a:
            pass

    def update(self):
        pass
    
    def attack(self):
        pass
    
    def jump(self):
        pass
    
    def replace(self):
        if not self.is_position_valid():
            self.rect.x = INIT_POS.x
            self.rect.y = INIT_POS.y

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

    def draw_car(self):
        self.screen.blit(self.image, self.rect)


    