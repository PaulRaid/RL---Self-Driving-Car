import pygame
from pygame.locals import *
from utils_track import *
from car import *
from constants import *
from geometry import *
from DeepQN import DQNAgent


successes, failures = pygame.init()
print("{0} successes and {1} failures".format(successes, failures))

# Game intialisation
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))


track = Track(screen)
driver = Car(screen, track)

agent = DQNAgent(state_space = NUM_RAYS+1, 
                 action_space = NUM_ACTIONS, 
                 dropout = 0.2,  
                 hidden_size= 512,
                 pretrained = False, 
                 lr = 0.00025, 
                 gamma=0.99, 
                 max_mem_size = 30000, 
                 exploration_rate = .1, 
                 exploration_decay = .999995, 
                 exploration_min = 0.01,  
                 batch_size = 256)

# Animation Loop
while True:
    clock.tick(FPS)
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.KEYDOWN:
            driver.modify(event)
    
    state = driver.get_observations()
    
    recom_action = agent.act(state)
    
    old_state, issue, reward, action_chosen, terminal = driver.act(recom_action)
    
    agent.remember(state, action_chosen, reward, issue, terminal)
    
    agent.driving_lessons()
    
    track.draw_track()
    driver.draw_car(screen)
    #driver.print_car()
    pygame.display.update()

