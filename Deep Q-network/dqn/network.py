import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(NeuralNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)     
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 