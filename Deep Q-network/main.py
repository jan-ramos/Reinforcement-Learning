from time import time
from collections import deque
import numpy as np
from dqn.dqn import DQN
import gym
import torch

import matplotlib.pyplot as plt


def update_epsilon():
        if epsilon < 0.01:
            return 0.01
        else:
            return epsilon * 0.999

start = time()
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
device = torch.device('cuda:0')
batch_size = 256   
gamma = 0.99  

agent = DQN(state_size, action_size,device, batch_size,gamma, seed=0, )
scores_list = []

batch_size = 256         

n_episodes = 1000
epsilon = 1 
update_step = 0
max_length = 1000

scores_window = deque(maxlen=100)
for episode in range(n_episodes):
    state = env.reset()
    score = 0
    for i in range(max_length):
        update_step +=1 
        action = agent.greedy_action(state,epsilon)
        next_state, reward, is_done,_ = env.step(action)
        
        agent.add_memory(state, action, reward, next_state, is_done)
        
        if update_step % 4 == 0:
            agent.update(state, action, reward, next_state, is_done)
        
        state = next_state        
        score += reward        
        if is_done:
            break
            
#------------------------------------------------------------------------------------------------------------------------------
        if episode % 1 == 0:
            mean_score = np.mean(scores_window)
            print('\r Progress {}/{}, average score:{:.2f}'.format(episode, n_episodes, mean_score), end="")
        
#------------------------------------------------------------------------------------------------------------------------------
       
        epsilon = update_epsilon() 
    scores_list.append(score)
    scores_window.append(score)
end = time()