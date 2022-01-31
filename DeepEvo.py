from matplotlib.pyplot import get
import pygame
import torch
import torch.nn as nn
import numpy as np
from car import * 
from constants import *
import random
import pickle 


class IndividualBrain(nn.Module):
    def __init__(self, screen, track, input_size = NUM_RAYS+1, n_actions = NUM_ACTIONS , hidden_s1 = 10, hidden_s2 = 10) -> None:
        super().__init__()
        self.car = Car(screen, track)
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_s1),
            nn.ReLU(),
            nn.Linear(hidden_s1, hidden_s2),
            nn.ReLU(),
            nn.Linear(hidden_s2, n_actions),
            nn.Softmax()
        )
        self.nn.apply(self.init_weights)
        
    def init_weights(self, m):
         if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            #m.bias.data.fill_(0.01)
    
        
    def forward(self, x):
        return self.nn(x)
    
    def mutate(self, x):
        pass
    
    def get_car(self):
        return self.car


    
class Evolution():
    def __init__(self,screen, track, nb_indiv = 250) -> None:
        self.nb_indiv = nb_indiv
        self.list_indiv = [IndividualBrain(screen, track) for i in range(nb_indiv)]
        self.scores = np.zeros(nb_indiv)
        
    def act(self):
        for i,a in enumerate(self.list_indiv):
            car = a.get_car()
            state, t = car.get_observations()
            
            # PyTorch part
            state_tensor = torch.from_numpy(state).float()
            action = a(state_tensor.to(self.device)).argmax().unsqueeze(0).unsqueeze(0).cpu()
            
            #car part
            car.act(action_chosen= action)

        
            
            