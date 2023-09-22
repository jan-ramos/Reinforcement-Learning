import random
from collections import namedtuple
import numpy as np
import torch



class ReplayBuffer:
    def __init__(self, batch_size,device, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
        self.memory = []*100000
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
      
    def sample_memory(self):
        sequences =  random.sample(self.memory,self.batch_size)
        states = torch.from_numpy(self.stack_torch(sequences,"state")).float().to(self.device)
        actions = torch.from_numpy(self.stack_torch(sequences,"action")).long().to(self.device) 
        rewards = torch.from_numpy(self.stack_torch(sequences,"next_reward")).float().to(self.device)       
        next_states = torch.from_numpy(self.stack_torch(sequences,"next_state")).float().to(self.device)
        is_dones = torch.from_numpy(self.stack_torch(sequences,"is_done").astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, is_dones)
    
    def upload_memory(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))
        
    def stack_torch(self,sequences,observation):
        observations = ['state','action','next_reward','next_state','is_done']
        index = observations.index(observation)
        observation_list = []
        for sequence in sequences:
            observation_list.append(sequence[index])
            
        return np.vstack(observation_list)
    
    def storage(self):
        if not self.memory:
            print("Empty Memory")
        else:
            print("Memory:", len(self.memory))

    def reset_memory(self):
        self.memory = []  
        
    def __len__(self):
        
        return len(self.memory)