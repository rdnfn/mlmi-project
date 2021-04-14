""" Utils for deep Q-learning.

Including replay memory for saving and randomly sampling 
transitions, strongly influenced by [1].

[1] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch
import random
import math
import mido
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:
    """Replay memory to store and sample (s, a, r, s')
    transitions during training."""
    
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.memory = []
        self.position = 0
        
    def add(self, *args):
        transition = Transition(*args)
        
        if len(self.memory) < self.max_capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
            
        self.position = (self.position + 1) % self.max_capacity 
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    

class Scheduler:
    """Scheduler for events during training:
    epsilon, no_note_epsilon, target_net update,
    model_save
    """
    
    def __init__(self, epsilon_start=0.9, 
                 epsilon_end=0.05,
                 epsilon_decay_rate=200,
                 target_update_frequency=10, 
                 model_save_frequency=10):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.target_update_frequency = target_update_frequency
        self.model_save_frequency = model_save_frequency
    
    def get_epsilon(self, steps_done):
        
        epsilon_start = self.epsilon_start
        epsilon_end = self.epsilon_end
        
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1. * steps_done / self.epsilon_decay_rate)
        return epsilon
    
    def should_update_targets(self, steps_done):
        if steps_done % self.target_update_frequency == 0:
            return True
        else:
            return False
    
    def should_save_model(self, steps_done):
        if steps_done % self.model_save_frequency == 0:
            return True
        else:
            return False
        
        
def select_action(q_net, state, epsilon, no_note_epsilon, num_actions):
    """Returns action according to epsilon-greedy policy
    w.r.t. the given q_net. Additionally possible to set
    a fixed proportion of no note events via `no_note_epsilon`.
    `epsilon` + `no_note_epsilon` must be less than 1.
    """
    
    action_type = None
    
    sample = random.random()
    
    if sample > epsilon + no_note_epsilon:
        with torch.no_grad():
            action = q_net(state).argmax().view(1,1)
        action_type = "greedy"
    elif no_note_epsilon and sample < epsilon + no_note_epsilon and sample > epsilon:
        # no note played == highest action index
        action = torch.LongTensor([[num_actions-1]])
        action_type = "fixed"
    else:
        action = torch.LongTensor([[random.randrange(0,num_actions)]])
        action_type = "random"
        
    return action, action_type


def check_midi_empty(midi_path):
    """Checks if MIDI file is empty. Returns True if it is empty.
    """
    
    mid = mido.MidiFile(midi_path)
    midi_is_empty = not bool(list(mid.play()))
    return midi_is_empty
            
        
