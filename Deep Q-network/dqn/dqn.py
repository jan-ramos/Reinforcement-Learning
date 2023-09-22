
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dqn.replaybuffer import ReplayBuffer
from dqn.network import NeuralNetwork

class DQN:
    def __init__(self, state_size, action_size, device, batch_size, gamma, seed,):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        # Initialize Q and Fixed Q networks
        self.online_Qnet = NeuralNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.target_Qnet = NeuralNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.optimizer = optim.Adam(self.online_Qnet.parameters())
        
        # Initiliase memory 
        self.memory = ReplayBuffer(256,self.device, seed)
       
        
    
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.upload_memory(state, action, reward, next_state, done)
        
    def update(self, state, action, reward, next_state, done):
        if len(self.memory) > self.batch_size:
            samples = self.memory.sample_memory()
            states, actions, rewards, next_states, is_dones = samples
            max_actions = self.target_Qnet(next_states).detach().max(1)[0].unsqueeze(1)
            target = rewards+(max_actions*(1-is_dones)*self.gamma)
            expectation = self.online_Qnet(states).gather(1, actions)

            loss = F.mse_loss(expectation, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            parameters = zip(self.online_Qnet.parameters(), self.target_Qnet.parameters())
            for source, target in parameters:
                target.data.copy_(0.001 * source.data + (0.999) * target.data)


    def greedy_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else: 
            actions = self.online_Qnet(torch.from_numpy(state).float().unsqueeze(0).to(self.device))
            
            return np.argmax(actions.cpu().data.numpy())
              