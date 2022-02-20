from game_utils.geometry import *
# colors constants

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (127, 127, 127)
GREY_LIGHT = (190, 190, 190)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Window parameters

HEIGHT = 480*2
WIDTH = 720*2
ACC = 0.5
FRIC = -0.12
FPS = 120

# Track parameters 

OFFSET = 0.1
DECAL = 2
NUM_WALLS = 18 #14 , 18 or 45
LARG_PISTE = max(HEIGHT, WIDTH)
EASY_MAP = False

# Car parameters

SIZE = 16

#INIT_POS = Point(0.85*DECAL*OFFSET*WIDTH , 0.8*DECAL*OFFSET*HEIGHT)
INIT_POS = Point(OFFSET * WIDTH + 3 * (1-2*OFFSET) * WIDTH / 10 -100*(1-EASY_MAP) , 3 * DECAL* OFFSET * HEIGHT/4 -10*(1-EASY_MAP))
FIRST_PORTAL = 6

ANGLE = 15
INIT_VEL = 2
VELMAX = 6
INIT_ACC = 1

NUM_ACTIONS = 5
NUM_RAYS = 14
NUM_RAYS_GENETIC = 5


# Drawing conditions
DRAW_RAYS = True
DRAW_PORTALS = True
DRAW_SCORE = False

#AI 

REWARD_PORTAL = 1
REWARD_WALL = -1
REWARD_BASE = 0

PRINT_INFOS = True
REPLACE_TARGET = 50

NB_INDIV_GEN = 5