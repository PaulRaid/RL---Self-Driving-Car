import pygame
from pygame.locals import *
from game_utils.utils_track import *
from game_utils.car import *
from game_utils.constants import *
from game_utils.geometry import *
from evolution.DeepEvo import *

successes, failures = pygame.init()
print("{0} successes and {1} failures".format(successes, failures))

# Game intialisation
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

track = Track(screen)

population = Evolution(screen=screen,
                       track=track,
                       nb_indiv=NB_INDIV_GEN)

# Animation Loop
game_reward = []
current_reward = 0
counter = 0
counter_pop = -1

while True:
	clock.tick()
	screen.fill(BLACK)

	counter_pop += 1
	print(" > Launching population " + str(counter_pop))

	population.generate_next_pop()

	counter = 0
	state, terminal = population.get_observations()
	current_reward = 0
	issue = state.copy()
 
	while not (sum(terminal) == len(terminal)):

		for event in pygame.event.get():  # handle keybord events
			if event.type == pygame.QUIT:
				population.save("results/evolution")
				quit()

		steer_, throttle_ = population.predict_action(issue)

		old_state, issue, reward, action_chosen, terminal = population.act(steer = steer_, throttle = throttle_)
  
		print(action_chosen)
		track.draw_track()
		population.draw(screen)
		# driver.print_car()
		pygame.display.update()

		

	# The car has crashed
	