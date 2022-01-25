from turtle import forward
import pygame
import torch
import torch.nn as nn
import numpy as np
from car import * 
from constants import *
import random


class DQNSolver(nn.Module):
    def __init__(self,ins = 2, n_actions = NUM_ACTIONS  ) -> None:
        super(DQNSolver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ins, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        pass

class DQNAgent: 
    def __init__(self, max_mem_size = 30000) -> None:
         self.state_space = 0
         self.action_space = NUM_ACTIONS
         self.pretrained = False
         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         
         self.dqn = DQNSolver().to(self.device)
         
         self.optimizer = torch.optim.Adam(self.dqn.parameters())
         self.loss = nn.SmoothL1Loss().to(self.device)
         self.gamma = 1
         
         self.memory_size = max_mem_size
         self.exploration_rate = 0.5      # To preserve from getting stuck
         
         self.rem_states = torch.zeros(max_mem_size, NUM_RAYS + 1)
         self.rem_actions = torch.zeros(max_mem_size, 1)
         self.rem_rewards = torch.zeros(max_mem_size, 1)
         self.rem_issues = torch.zeros(max_mem_size, NUM_RAYS + 1)
         
         self.current_position = 0
         
    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, 8)
        else: 
            return self.dqn(state.to(self.device))
        
    def remember(self, state, action, reward, issue):
        self.rem_states[self.current_position] =  state.float()
        self.rem_actions[self.current_position] =  action.float()
        self.rem_rewards[self.current_position]=  reward.float()
        self.rem_issues[self.current_position] =  issue.float()
        self.current_position = (self.current_position + 1) % self.memory_size
        
    def driving_lessons(self):
        
        # compute a random batch from the memory before and pass it, then retrop
        
        self.optimizer.zero_grad()
        
        
        #target = reward + torch.mul ... --> Formula for the DQN impl r + gamma * max (on actions) ( issue, action)
        current = self.dqn(state)
        
        loss = self.loss(current, target)
        loss.backward()
        self.optimizer.step()
        
    