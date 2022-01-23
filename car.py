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
        self.direction = "RIGHT"

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
        print(x,y)
        s = SIZE
        up = [(x,y), (x + 2*s, y)]
        down = [(x,y + s), (x + 2*s, y + s)]
        left = [(x,y), (x, y + s)]
        right = [(x + 2*s, y), (x + 2*s, y + s)]
        sides = [up, down,left,right]
        res = True
        for a in sides:
            for wall in self.track.get_walls():
                res = res and not intersect(a[0], a[1], wall.get_start(), wall.get_last())
        return res

    def draw_car(self):
        self.screen.blit(self.image, self.rect)


    