import pygame
import torch
import torch.nn as nn
import numpy as np
from car import * 
from constants import *
import random
import pickle 
from DeepEvo import *
import numpy as np
import matplotlib.pyplot as plt
  
# Using exponential() method
gfg = np.random.exponential(500, 10000)
gfg1 = np.random.exponential(gfg, 10000)
  
count, bins, ignored = plt.hist(gfg, 14, density = True)
for a in gfg:
    if a ==0:
        print(a)

print(np.max(gfg))
plt.show()


