import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transcribe.constants.general import *


class PolicyNet(nn.Module):
    def __init__(self, state_space_size, action_space_size, hidden_size=128,
                 hidden_layer_num = 3,
                 uniform_prior=0.0, take_log_state=True):
        super().__init__()
        
        self.take_log_state = take_log_state
        self.action_space = np.arange(action_space_size)
        self.action_space_size = action_space_size
        self.uniform_prior = uniform_prior
        
        layers = [nn.Linear(state_space_size, hidden_size), nn.LeakyReLU()]
        layers += [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()] * (hidden_layer_num - 1)
        layers += [nn.Linear(hidden_size, action_space_size), nn.Softmax(dim=-1)]
        
        self.net = nn.Sequential(*layers)
    
    def predict(self, state):
        state = torch.FloatTensor(state)
        if self.take_log_state:
            state = torch.log(state)
        
        action_probs = self.net(state)
        return action_probs
    
    def sample_action(self, state):
        action_probs = self.predict(state)
        action_probs_prior = action_probs * (1-self.uniform_prior) + self.uniform_prior/self.action_space_size
        sampled_action = np.random.choice(self.action_space,
                                          p=action_probs_prior.detach().numpy())
        
        return sampled_action, action_probs
    

class DeepQNetwork(nn.Module):
    
    def __init__(self, input_size=MODEL_INPUT_SIZE, 
                 output_size=MODEL_OUTPUT_SIZE,
                 architecture=MODEL_ARCHITECTURE):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, architecture[0])
        self.fc2 = nn.Linear(architecture[0], architecture[1])
        self.fc3 = nn.Linear(architecture[1], architecture[2])
        self.fc4 = nn.Linear(architecture[2], output_size)
        
    
    def forward(self, x):
        
        # input should be mel spectrogram
        x = torch.log(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        
        return x