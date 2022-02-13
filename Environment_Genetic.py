import pygame
from pygame.locals import *
from utils_track import *
from car import *
from constants import *
from geometry import *
from DeepQN import *
from DeepEvo import *

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

	while not (sum(terminal) == len(terminal)):

		for event in pygame.event.get():  # handle keybord events
			if event.type == pygame.QUIT:
				quit()

		recom_action = population.predict_action(state)

		old_state, issue, reward, action_chosen, terminal = population.act(recom_action)

		'''if reward == 0:  # To kill cars that haven't been rewarded in last 100 games
			counter += 1
			if counter > 100:
				terminal = True
		else:
			counter = 0'''

		'''print("\nold_state",old_state)
		print("issue", issue)
		print("reward", reward)
		print("action", action_chosen)
		print("terminal", terminal )'''

		track.draw_track()
		population.draw(screen)
		# driver.print_car()
		pygame.display.update()

		# current_reward += reward

	# The car has crashed
	# print("kill")
	'''
	game_reward.append(current_reward)
	ind = len(game_reward)
	if ind % REPLACE_TARGET == 0 and ind > REPLACE_TARGET:
			agent.update_params()
	if ind % 10 ==0:                    # print current learning infos every 10 games
		avg = np.mean(game_reward[max(ind-100, 0):ind])
		print("> Game Numer : " + str(ind) + " | Last Game Reward = " + str(current_reward) + " | Average R on 100 last games : " + str(avg) + " | Exploration rate : " + str(agent.get_exploration()))
	'''
