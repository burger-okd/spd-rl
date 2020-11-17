# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import gym
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from collections import deque


# %%
TRAINING  = 30
GAMMA = 0.99
LR    = 0.001

MAX_LEN  = 200
memory   = deque(maxlen = MAX_LEN)

env = gym.make('FrozenLake-v0').env

state_size  = env.observation_space.n
action_size = env.action_space.n
hidden_size = 16

SA = state_size * action_size


# %%
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VNet(nn.Module):
    """
    V_net = nn.Sequential(nn.Linear(state_size,hidden_size), 
                          nn.ReLU(),
                          nn.Linear(hidden_size,1)
                         )
    """
    def __init__(self):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MuNet(nn.Module):
    """
    mu_net = nn.Sequential(nn.Linear(state_size,hidden_size), 
                           nn.ReLU(),
                           nn.Linear(hidden_size,action_size),
                           nn.Softmax(dim=0)            
                          )

    """
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        ### makes mu greater than 0
        x = F.softmax(self.fc2(x), dim = 0)
        return x

v_net  = VNet()
mu_net = MuNet()

optimizer_v  = optim.Adam(v_net.parameters(), lr = 1e-3)
optimizer_mu = optim.Adam(mu_net.parameters(), lr = 1e-3)


# %%
def one_hot(state):
    temp = torch.zeros(state_size)
    temp[state] = 1
            
    return temp


# %%
to_be_popped = 0
for i in range(TRAINING):
    ### State sampled uniformly
    state = np.random.choice(state_size - 1)
    
    ##
    v = v_net(one_hot(state))
    mu = mu_net(one_hot(state))
    
    action = torch.argmax(mu)
    next_state, reward, done, info = env.step(action.item())

    memory.append([state, action, reward, next_state, done])
    
    if len(memory) is MAX_LEN:
        lagr_v  = 0
        lagr_mu = 0
        for state, action, reward, next_state, done in memory:
            with torch.no_grad():
                fixed_mu     = mu_net(one_hot(state))[action.item()]
                fixed_v      = v_net(one_hot(state)) 
                fixed_v_next = v_net(one_hot(next_state))
            ##lagrangian mu-fixed
            lagr_v += (1-GAMMA) * v_net(one_hot(state)) + SA * fixed_mu * (reward + GAMMA * v_net(one_hot(next_state)) - v_net(one_hot(state)))
            #lagrangian v-fixed
            lagr_mu -= (1-GAMMA) * fixed_v + SA * mu_net(one_hot(state))[action.item()] * (reward + GAMMA * fixed_v_next - fixed_v)
        
        optimizer_v.zero_grad()
        optimizer_mu.zero_grad()
        
        lagr_v.backward()
        lagr_mu.backward()
        
        optimizer_v.step()
        optimizer_mu.step()


# %%
###Test
TEST  = 100
success = 0
for e in range(TEST):
    done  = False
    state = env.reset()
    while not done:
        #env.render()
        
        mu = MuNet()
        action = torch.argmax(mu(one_hot(state)))
        
        next_state, reward, done, info = env.step(action.item())
        next_state = state
    
    if reward == 1:
        success = success + 1
        
print(f"Total success: {success}/{TEST}")


# %%