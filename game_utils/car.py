import pygame.math

from game_utils.oberv import *


class Car:
	def __init__(self, screen, track):

		self.screen = screen
		self.track = track
		self.portals = track.get_portals()
		self.portals[FIRST_PORTAL].set_active()
		self.num_portal = FIRST_PORTAL  # the number of the current active portal

		self.image_base = pygame.Surface((2 * SIZE, SIZE))
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
		self.direction_vector = pygame.math.Vector2(1, 0)

		self.xspeed = INIT_VEL
		self.yspeed = 0
		self.velvalue = pygame.math.Vector2.length(Point(self.xspeed, self.yspeed))
		self.velmax = VELMAX
		self.acc = INIT_ACC

		self.rays = self.set_rays()  # array

		self.score = 0
		self.score_obj = Score()
		self.last_reward = 0

		self.old_observations = np.zeros(NUM_RAYS + 1)
		self.current_observations = np.zeros(NUM_RAYS + 1)
		self.last_actions = []

		self.viz_points = [Point(0, 0) for a in range(NUM_RAYS)]

	# -- method to change the class parameters of the car
	def modify(self, event):
		vec = self.direction_vector.normalize()
		orth = pygame.math.Vector2(vec.y, -vec.x)
		if event.key == pygame.K_LEFT:
			self.rect.move_ip(-20 * vec.x, -20 * vec.y)
		if event.key == pygame.K_RIGHT:
			self.rect.move_ip(20 * vec.x, 20 * vec.y)
		if event.key == pygame.K_UP:
			self.rect.move_ip(20 * orth.x, 20 * orth.y)
		if event.key == pygame.K_DOWN:
			self.rect.move_ip(-20 * orth.x, -20 * orth.y)
		if event.key == pygame.K_r:
			self.rect.x = INIT_POS.x
			self.rect.y = INIT_POS.y
		if event.key == pygame.K_p:  # Turn Left
			self.turn()
			self.rot()
		# pygame.quit()
		if event.key == pygame.K_o:  # Turn Right
			self.turn(-1)
			self.rot()
		# pygame.quit()
		if event.key == pygame.K_0:  # action 0 : doing nothing
			pass
		if event.key == pygame.K_1:  # action 1 : accel
			self.accelerate()
			pass
		if event.key == pygame.K_2:  # action 2 : decelerate
			self.accelerate(-1)
			pass
		if event.key == pygame.K_3:  # action 3 : Turn left
			self.turn()
			self.rot()
			pass
		if event.key == pygame.K_4:  # action 4 : Turn right
			self.turn(-1)
			self.rot()
			pass
		if event.key == pygame.K_5:  # action 5 : Acc + Turn left
			self.accelerate()
			self.turn()
			self.rot()
			pass
		if event.key == pygame.K_6:  # action 6 : Acc + Turn right
			self.accelerate()
			self.turn(-1)
			self.rot()
			pass
		if event.key == pygame.K_7:  # action 7 : Dec + Turn left
			self.accelerate(-1)
			self.turn()
			self.rot()
			pass
		if event.key == pygame.K_8:  # action 6 : Dec + Turn right
			self.accelerate(-1)
			self.turn(-1)
			self.rot()
			pass

	# -- method to call after action
	def act(self, action_chosen):
		old_state, t = self.get_observations()
		if action_chosen == 0:  # action 0 : doing nothing
			pass
		if action_chosen == 1:  # action 1 : Turn left
			self.turn()
			self.rot()
			pass
		if action_chosen == 2:  # action 2 : Turn right
			self.turn(-1)
			self.rot()
			pass
		if action_chosen == 3:  # action 3 : accel
			self.accelerate()
			pass
		if action_chosen == 4:  # action 4 : decelerate
			self.accelerate(-1)
			pass
		if action_chosen == 5:  # action 5 : Acc + Turn left
			self.accelerate()
			self.turn()
			self.rot()
			pass
		if action_chosen == 6:  # action 6 : Acc + Turn right
			self.accelerate()
			self.turn(-1)
			self.rot()
			pass
		if action_chosen == 7:  # action 7 : Dec + Turn left
			self.accelerate(-1)
			self.turn()
			self.rot()
			pass
		if action_chosen == 8:  # action 6 : Dec + Turn right
			self.accelerate(-1)
			self.turn(-1)
			self.rot()
			pass
		terminal = self.update()
		self.last_actions.append(action_chosen)
		new_state, t = self.get_observations()
		reward = self.get_reward()
		return old_state, new_state, reward, action_chosen, terminal

	# -- Acceleration of the car
	def accelerate(self, dir=1):
		acc = 2 * self.acc

		if dir == 1:
			self.velvalue = self.velvalue + acc
		else:
			self.velvalue = self.velvalue - acc

		if self.velvalue > self.velmax:
			self.velvalue = self.velmax

		if self.velvalue < -self.velmax:
			self.velvalue = -self.velmax

	# -- method to actually move the position of the car
	def update(self):
		vec = self.direction_vector.normalize()
		vec = clamp_close_number(vec)

		vite = self.velvalue
		decalx = arrondiSup(vite * vec.x)
		decaly = arrondiSup(vite * vec.y)

		self.rect.center = (self.rect.centerx + decalx, self.rect.centery + decaly)
		self.rays = self.set_rays()
		self.set_vision()
		return self.replace()

	# -- method to turn the dir vector by ANGLE
	def turn(self, dir=1):
		self.angle = (self.angle + dir * ANGLE) % 360
		self.direction_vector = self.direction_vector.rotate(-1 * dir * ANGLE)

	# -- method to effectively rotate the car
	def rot(self):

		angle_ = self.angle
		nouv_vel = rotate_point_around_center(Point(0, 0), Point(0, self.velvalue), angle_)
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

	# -- method to check if the position is authorised and replace it # Returns True if it has hurted a wall (for Q learning training)
	def replace(self, reset=False):
		if not self.is_position_valid() or reset:
			# -- Reset Object
			new_image = pygame.transform.rotate(self.image_base, 0)
			self.rect = new_image.get_rect()
			self.rect.x = INIT_POS.x
			self.rect.y = INIT_POS.y
			self.image = new_image.copy()

			# -- Reset Position
			self.angle = 0
			self.direction_vector = pygame.math.Vector2(1, 0)
			self.p1 = Point(self.rect.topleft[0], self.rect.topleft[1])
			self.p2 = Point(self.rect.topright[0], self.rect.topright[1])
			self.p3 = Point(self.rect.bottomright[0], self.rect.bottomright[1])
			self.p4 = Point(self.rect.bottomleft[0], self.rect.bottomleft[1])

			# -- Reset Velocity

			self.xspeed = INIT_VEL
			self.yspeed = 0
			self.velvalue = pygame.math.Vector2.length(Point(self.xspeed, self.yspeed))
			self.accel = 1

			# -- Reset active Portal
			self.portals[self.num_portal].set_inactive()
			self.portals[FIRST_PORTAL].set_active()
			self.num_portal = FIRST_PORTAL  # the number of the current active portal
			
			if not reset:
				# -- Score for IA
				self.score += REWARD_WALL
				self.last_reward = REWARD_WALL
			return 1

		if not self.set_portals():  # updates the portals in place: if nothing has changed --> give life reward
			# -- Score for IA
			self.score += REWARD_BASE
			self.last_reward = REWARD_BASE
		return 0

	# -- Checks intersection with track walls
	def is_position_valid(self):
		x = self.rect.centerx
		y = self.rect.centery

		s = SIZE

		u = self.direction_vector.normalize()
		v = pygame.math.Vector2(u.y, -u.x)  # Anticlock wise
		s = SIZE
		u = s * u
		v = s * v / 2

		topleft = Point(x - u.x + v.x, y - u.y + v.y)
		topright = Point(x + u.x + v.x, y + u.y + v.y)
		bottomleft = Point(x - u.x - v.x, y - u.y - v.y)
		bottomright = Point(x + u.x - v.x, y + u.y - v.y)

		up = [topleft, topright]
		down = [bottomleft, bottomright]
		left = [topleft, bottomleft]
		right = [topright, bottomright]
		sides = [up, down, left, right]
		res = True
		for a in sides:
			for wall in self.track.get_walls():
				res = res and not intersect(a[0], a[1], wall.get_start(), wall.get_last())
		return res

	# -- Actualises the current active portal, returns true if it has done something
	def set_portals(self):
		x = self.rect.centerx
		y = self.rect.centery

		s = SIZE

		u = self.direction_vector.normalize()
		v = pygame.math.Vector2(u.y, -u.x)  # Anticlock wise
		s = SIZE
		u = s * u
		v = s * v / 2

		topleft = Point(x - u.x + v.x, y - u.y + v.y)
		topright = Point(x + u.x + v.x, y + u.y + v.y)
		bottomleft = Point(x - u.x - v.x, y - u.y - v.y)
		bottomright = Point(x + u.x - v.x, y + u.y - v.y)

		up = [topleft, topright]
		down = [bottomleft, bottomright]
		left = [topleft, bottomleft]
		right = [topright, bottomright]
		sides = [up, down, left, right]
		res = False
		for a in sides:
			current_active = self.portals[self.num_portal]
			if intersect(a[0], a[1], current_active.get_start(), current_active.get_last()):
				self.portals[self.num_portal].set_inactive()
				self.num_portal = (self.num_portal + 1) % len(self.portals)
				self.portals[self.num_portal].set_active()
				self.score += REWARD_PORTAL
				self.last_reward = REWARD_PORTAL
				res = True
		return res

	# -- Actualises the current points, and distances for vision
	def set_vision(self):
		for i, viz in enumerate(self.rays):
			point, dist = viz.track_intersection(self.track.get_walls())
			self.viz_points[i] = point
			self.current_observations[i] = (LARG_PISTE - dist) / LARG_PISTE  # -> to normalise the current observation between 0, 1
			if self.current_observations[i] == -np.inf:
				self.current_observations[i] = 1
		self.current_observations[-1] = 0.5 + self.velvalue / (2 * VELMAX)

	# -- To draw the car, the rays, and the portals
	def draw_car(self, screen):
		screen.blit(self.image, self.rect)

		for i, viz in enumerate(self.rays):
			point = self.viz_points[i]
			if DRAW_RAYS:
				viz.dray_ray(point, screen)
		if DRAW_PORTALS:
			for portal in self.portals:
				portal.draw_portal(screen)
		if DRAW_SCORE:
			self.score_obj.draw_score(screen, self.score)

	# -- getter for AI
	def get_observations(self):
		return self.current_observations.copy(), 0

	# -- getter for AI
	def get_reward(self):
		return self.last_reward

	def get_distance_from_begin(self):
		x = self.rect.centerx
		y = self.rect.centery
		return False

	# -- prints 4 coords of the car
	def print_car(self):
		print("\nLast Reward", self.get_reward())
		print("score", self.score)
		print("Observation", self.get_observations())

	# -- To set up the rays leaving the car
	def set_rays(self):
		u = self.direction_vector.normalize()
		v = pygame.math.Vector2(u.y, -u.x)  # Anticlock wise
		s = SIZE
		u = s * u
		v = s * v / 2

		res = []

		res.append(Vision(self.rect.centerx + u.x, self.rect.centery + u.y, self.direction_vector))  # front
		res.append(Vision(self.rect.centerx - u.x, self.rect.centery - u.y, -self.direction_vector))  # back

		res.append(
			Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector))  # front 0
		res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y,
		                  self.direction_vector.rotate(-15)))  # front up 30
		res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y,
		                  self.direction_vector.rotate(-30)))  # front up 30
		res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y,
		                  self.direction_vector.rotate(-60)))  # front up 60
		# res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, self.direction_vector.rotate(15))) # front up down 15

		res.append(Vision(self.rect.centerx + v.x, self.rect.centery + v.y, v))  # side up 90
		res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y, v))

		res.append(Vision(self.rect.centerx - v.x, self.rect.centery - v.y, -v))  # side down 90
		res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, -v))

		res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector))  # down
		res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y,
		                  self.direction_vector.rotate(15)))  # front down  30
		res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y,
		                  self.direction_vector.rotate(30)))  # front down  30
		res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y,
		                  self.direction_vector.rotate(60)))  # front down  60
		# res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y, self.direction_vector.rotate(-15))) # front down up 60

		return res


class Car_evo(Car):
	def __init__(self, screen, track):
		super().__init__(screen, track)
		self.rays = self.set_rays()
		self.current_observations = np.zeros(NUM_RAYS_GENETIC + 1 )

	# @Override
	def act(self, action_chosen):
		old_state, t = self.get_observations()
		if action_chosen == 0:  # action 0 : doing nothing
			pass
		if action_chosen == 1:  # action 1 : Turn left
			self.turn()
			self.rot()
			pass
		if action_chosen == 2:  # action 2 : Turn right
			self.turn(-1)
			self.rot()
			pass
		if action_chosen == 3:  # action 3 : accel
			self.accelerate()
			pass
		if action_chosen == 4:  # action 4 : decelerate
			self.accelerate(-1)
			pass
		if action_chosen == 5:  # action 5 : Acc + Turn left
			self.accelerate()
			self.turn()
			self.rot()
			pass
		if action_chosen == 6:  # action 6 : Acc + Turn right
			self.accelerate()
			self.turn(-1)
			self.rot()
			pass
		if action_chosen == 7:  # action 7 : Dec + Turn left
			self.accelerate(-1)
			self.turn()
			self.rot()
			pass
		if action_chosen == 8:  # action 6 : Dec + Turn right
			self.accelerate(-1)
			self.turn(-1)
			self.rot()
			pass
		terminal = self.update()
		self.last_actions.append(action_chosen)
		if terminal:
			new_state, t = self.get_observations()
		else:
			new_state, t = self.get_observations()
		reward = self.get_reward()
		return old_state, new_state, reward, action_chosen, terminal

	# @Override
	def replace(self, reset=False):
		if not self.is_position_valid() or reset:
			# -- Score for IA
			self.score += REWARD_WALL
			self.last_reward = REWARD_WALL
			return 1

		if not self.set_portals():  # updates the portals in place: if nothing has changed --> give life reward
			# -- Score for IA
			self.score += REWARD_BASE
			self.last_reward = REWARD_BASE
		return 0

	# @Override --> New version
	def set_rays(self):
		u = self.direction_vector.normalize()
		v = pygame.math.Vector2(u.y, -u.x)  # Anticlock wise
		s = SIZE
		u = s * u
		v = s * v / 2

		res = []

		res.append(Vision(self.rect.centerx + u.x, self.rect.centery + u.y, self.direction_vector))  # front

		res.append(Vision(self.rect.centerx + u.x + v.x, self.rect.centery + u.y + v.y,
		                  self.direction_vector.rotate(-15)))  # front up 30
		
		res.append(Vision(self.rect.centerx + u.x - v.x, self.rect.centery + u.y - v.y,
		                  self.direction_vector.rotate(15)))  # front down  30
		

		return res

class CarNN(Car):
	def __init__(self, screen, track):
		super().__init__(screen, track)

	def get_distance_from_beginning(self):
		cosangle = self.rect.centerx * self.portals[self.num_portal].x1 + self.rect.centery * self.portals[
			self.num_portal].y1
		return
