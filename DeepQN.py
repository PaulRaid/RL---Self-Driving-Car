from turtle import forward
import pygame
import torch
import torch.nn as nn
import numpy as np
from car import * 
from constants import *
import random


class DQNSolver(nn.Module):
    def __init__(self, input_size = NUM_RAYS+1, n_actions = NUM_ACTIONS , dropout = 0.2) -> None:
        super(DQNSolver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_actions),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent: 
    def __init__(self, max_mem_size = 30000, batch_size = 32) -> None:
         self.state_space = 0
         self.action_space = NUM_ACTIONS
         self.pretrained = False
         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         
         self.dqn = DQNSolver().to(self.device)
         
         self.lr = 0.00025
         
         self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr = self.lr)
         self.loss = nn.SmoothL1Loss().to(self.device)
         self.gamma = 0.9
         
         self.memory_size = max_mem_size
         self.exploration_rate = 1.0      # To preserve from getting stuck
         self.exploration_decay = 0.99
         self.exploration_min = 0.1
         
         self.rem_states = torch.zeros(max_mem_size, NUM_RAYS + 1)
         self.rem_actions = torch.zeros(max_mem_size, 1)
         self.rem_rewards = torch.zeros(max_mem_size, 1)
         self.rem_issues = torch.zeros(max_mem_size, NUM_RAYS + 1)
         self.rem_terminals = torch.zeros(max_mem_size, 1)
         
         self.current_position = 0
         
         self.batch_size = batch_size
         
    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, NUM_ACTIONS-1)
        else: 
            state = torch.from_numpy(state).float()
            action = self.dqn(state.to(self.device)).argmax()
            return action  
        
    def remember(self, state, action, reward, issue, terminal):
        self.rem_states[self.current_position] =  torch.from_numpy(state).float()
        self.rem_actions[self.current_position] =  torch.tensor(action).float()
        self.rem_rewards[self.current_position] = torch.tensor(reward).float()
        self.rem_issues[self.current_position] = torch.from_numpy(issue).float()
        self.rem_terminals[self.current_position] = torch.tensor(terminal).float()
        self.current_position = (self.current_position + 1) % self.memory_size
    
    def compute_batch(self):
        indices = random.choices(range(self.current_position), k = self.batch_size)
        
        state_batch = self.rem_states[indices]
        action_batch = self.rem_actions[indices]
        reward_batch = self.rem_rewards[indices]
        issue_batch = self.rem_issues[indices]
        terminal_batch = self.rem_terminals[indices]
        
        return state_batch,action_batch,reward_batch,issue_batch, terminal_batch
       
    def driving_lessons(self):
        
        # compute a random batch from the memory before and pass it, then retrop
        
        state,action,reward,issue,term = self.compute_batch()
        
        self.optimizer.zero_grad()
        
        #Q - learning :  target = r + gam * max_a Q(S', a)
        
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        issue = issue.to(self.device)
        term = term.to(self.device)
        
        target = reward + torch.mul(self.gamma * self.dqn(issue).max(1).values , 1-term)
        current = self.dqn(state).gather(1, action.long())
        
        loss = self.loss(current, target)
        loss.backward()
        self.optimizer.step()
        
        # Eventually reduce the exploration rate
        
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
        
    