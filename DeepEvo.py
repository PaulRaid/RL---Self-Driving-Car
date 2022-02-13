from matplotlib.pyplot import get
import pygame
import torch
import torch.nn as nn
import numpy as np
from car import * 
from constants import *
import random
import pickle 
import copy
from collections import *



class IndividualBrain(nn.Module):
    def __init__(self, screen, track, input_size = NUM_RAYS+1, n_actions = NUM_ACTIONS , hidden_s1 = 10, hidden_s2 = 10) -> None:
        super().__init__()
        self.car = Car_evo(screen, track)
        self.input_size =input_size
        self.n_actions = n_actions
        self.hidden_s1 =hidden_s1
        self.hidden_s2 =hidden_s2
        self.screen = screen
        self.track = track
        
        self.nn = nn.Sequential(
             OrderedDict([
                ('input' , nn.Linear(input_size, hidden_s1)),
                ('relu1' , nn.ReLU()),
                ('hidden' , nn.Linear(hidden_s1, hidden_s2)),
                ('relu2' , nn.ReLU()),
                ('output' , nn.Linear(hidden_s2, n_actions)),
                ('sigm' , nn.Softmax())]
            )
        )
        self.nn.apply(self.init_weights)
        
    def init_weights(self, m):
         if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a = -3, b =3)
            torch.nn.init.uniform_(m.bias,a = -3, b =3)
        
    def forward(self, x):
        return self.nn(x)
    
    def cross(self, partner):
        child = IndividualBrain(self.screen, self.track, self.input_size, self.n_actions, self.hidden_s1, self.hidden_s2)
        mother_dic = self.get_dict()
        father_dic = self.get_dict()
        
        child_dic = mother_dic.copy()
        
        key_tab = mother_dic.keys()
        key_tab = [(key.split(".")[0], key.split(".")[1] ) for key in key_tab]
        
        if random.randint(0, 8) == 1:
            mutation_rate = 10
        else:
            mutation_rate = 100000000 # Infinity -> no mutation
        
        for i, (level, type_) in enumerate(key_tab):
            if type_ == "weight":
                tens_ = child_dic[level + "." + type_].clone()
                for i,elem in enumerate(tens_):
                    for j, num in enumerate(elem):
                        p = random.random()
                        if p > 0.5:
                            tens_[i][j] = mother_dic[level + "." + type_][i][j]
                        else:
                            tens_[i][j] = father_dic[level + "." + type_][i][j]

                    # Mutate
                        if random.randint(0, mutation_rate) == 1:
                            tens_[i][j] += 4*random.random() - 2
        
                child_dic[level + "." + type_] = tens_.clone()
                                
            if type_ == "bias":
                tens_ = child_dic[level + "." + type_].clone()
                for i,elem in enumerate(tens_):
                    p = random.random()
                    tens_[i] = p * mother_dic[level + "." + type_][i] + (1 - p) * father_dic[level + "." + type_][i]
                    if random.randint(0, mutation_rate) == 1:
                        tens_[i] += 4*random.random() - 2
                child_dic[level + "." + type_] = tens_.clone()
        
        for old_key in child_dic.copy().keys():
            child_dic["nn."+old_key] = child_dic.pop(old_key)
        
        child.load_state_dict(child_dic)
        return child
    
    def get_car(self):
        return self.car
    
    def get_dict(self):
        return self.nn.state_dict().copy()

    
class Evolution():
    def __init__(self,screen, track, nb_indiv = 250) -> None:
        self.nb_indiv = nb_indiv
        self.list_indiv = [IndividualBrain(screen, track) for i in range(nb_indiv)]
        self.scores = np.zeros(nb_indiv)
        self.dead = np.zeros(nb_indiv)
        self.decision = np.zeros(nb_indiv)  # To check if an agent has scored moved in the last 150 ticks 
        self.last_rew = np.zeros(nb_indiv)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def act(self):
        for i,a in enumerate(self.list_indiv):
            car = a.get_car()
            state, t = car.get_observations()
            
            # PyTorch part
            state_tensor = torch.from_numpy(state).float()
            action = a(state_tensor.to(self.device)).argmax().unsqueeze(0).unsqueeze(0).cpu()
            
            #car part
            car.act(action_chosen = action)
            
    # After having selected the 10% best of the previous gen
    def cross(self, best):
        new_indiv = []
        nb_best = len(best)
        for i in range(self.nb_indiv):
            idx1 = random.randint(0,nb_best-1)
            idx2 = random.randint(0,nb_best-1)
            new_indiv.append(best[idx1].cross(best[idx2]))
        return new_indiv
            
    
    def generate_next_pop(self):
        argm = np.argmax(self.scores)
        valm = self.scores[argm]
        print("     > Best score for this gen is " + str(valm))
        best_people =[]
        coef = 0.9
        while len(best_people)<2:
            best_people = [elem for (i,elem) in enumerate(self.list_indiv) if self.scores[i] >= coef * valm]
            coef *=0.9
        self.list_indiv = self.cross(best_people)
        self.scores = np.zeros(self.nb_indiv)
        
        self.dead = np.zeros(self.nb_indiv)
        self.decision = np.zeros(self.nb_indiv)
        self.last_rew = np.zeros(self.nb_indiv)
         
    def get_observations(self):
        state_ = []
        terminal_ = []
        for i,elem in enumerate(self.list_indiv):
            if self.dead[i] == 0:
                state, terminal = elem.car.get_observations()
                state_.append(state)
                terminal_.append(terminal)
            else:
                state, terminal = None, 1
                state_.append(state)
                terminal_.append(terminal)
                
        return state_.copy(), terminal_.copy()
    
    def predict_action(self, state):
        recom = []
        for i,elem in enumerate(self.list_indiv):
            if self.dead[i] == 0:
                state_ = torch.from_numpy(state[i]).float()
                action_ = elem(state_.to(self.device)).argmax().unsqueeze(0).unsqueeze(0).cpu()
                recom.append(action_)
            else:
                recom.append(None)
        return recom
    
    def act(self, actions):
        state_ = []
        issue_ = []
        reward_ = []
        action_chosen_ = []
        terminal_ = []
        #print("dec",self.decision)
        for i,elem in enumerate(self.list_indiv):
            
            if self.decision[i] >= 150: 
                state_.append(None)
                issue_.append(None)
                reward_.append(0)
                action_chosen_.append(None)
                terminal_.append(1)
            
            elif self.dead[i] == 0:
                state, issue, reward, action_chosen, terminal = elem.car.act(actions[i])
                state_.append(state)
                issue_.append(issue)
                reward_.append(reward)
                action_chosen_.append(action_chosen)
                terminal_.append(terminal)
                self.scores[i] += reward
                self.dead[i] = terminal
                self.last_rew[i] = reward
                if reward == 0:
                    self.decision[i] +=1
                else:
                    self.decision[i] =0
                
            else:
                state_.append(None)
                issue_.append(None)
                reward_.append(0)
                action_chosen_.append(None)
                terminal_.append(1)
        return state_, issue_, reward_, action_chosen_, terminal_

    def draw(self, screen):
        best_people = [(i,elem) for (i,elem) in enumerate(self.list_indiv)]# if self.scores[i] >= 0.9 * np.max(self.scores)]
        for j, (i,elem) in enumerate(best_people):
            if self.dead[i] == 0:
                elem.car.draw_car(screen)

        
            
            