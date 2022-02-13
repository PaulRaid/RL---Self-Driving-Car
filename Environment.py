import pygame
from pygame.locals import *
from game_utils.utils_track import *
from game_utils.car import *
from game_utils.constants import *
from game_utils.geometry import *


successes, failures = pygame.init()
print("{0} successes and {1} failures".format(successes, failures))

# Game intialisation
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))


track = Track(screen)
driver = Car(screen, track)

# Animation Loop
while True:
    clock.tick(FPS)
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.KEYDOWN:
            driver.modify(event)
    
    driver.update()
    
    track.draw_track()
    driver.draw_car(screen)
    #driver.print_car()
    pygame.display.update()

