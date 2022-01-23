from geometry import *
# colors constants

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (127, 127, 127)
GREY_LIGHT = (190, 190, 190)
RED = (255, 0, 0)


# Window parameters

HEIGHT = 480*2
WIDTH = 720*2
ACC = 0.5
FRIC = -0.12
FPS = 60

# Track parameters 

OFFSET = 0.1
DECAL = 1.7

# Car parameters

SIZE = 16
INIT_POS = Point(0.85*DECAL*OFFSET*WIDTH , 0.8*DECAL*OFFSET*HEIGHT)
#INIT_POS = Point(OFFSET*WIDTH , OFFSET*HEIGHT)
#INIT_POS = Point(0,0)