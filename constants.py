from geometry import *
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
FPS = 60

# Track parameters 

OFFSET = 0.1
DECAL = 2
NUM_WALLS = 14

# Car parameters

SIZE = 16
INIT_POS = Point(0.85*DECAL*OFFSET*WIDTH , 0.8*DECAL*OFFSET*HEIGHT)

ANGLE = 15
INIT_VEL = 2

# Drawing conditions
DRAW_RAYS = True
DRAW_PORTALS = True
DRAW_SCORE = True