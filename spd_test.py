# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import gym
import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from collections import deque


# %%
TRAINING  = 3000
GAMMA = 0.99

MAX_LEN  = 300
memory   = deque(maxlen = MAX_LEN)

env = gym.make('FrozenLake-v0', is_slippery = False).env

state_size  = env.observation_space.n
action_size = env.action_space.n
hidden_size = 16

SA = state_size * action_size


# %%
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        ### makes mu greater than 0
        #x = torch.softmax(self.fc2(x),dim=0)
        x = torch.exp(self.fc2(x))
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
for state in range(state_size -1):
    for action in range(action_size):
        next_state, reward, done, info = env.step(action)
        memory.append([state, action, reward, next_state, done])

for i in range(TRAINING):
    lagr_v  = 0
    lagr_mu = 0
    for state, action, reward, next_state, done in random.sample(memory, 4):
        with torch.no_grad():
            fixed_mu     = mu_net(one_hot(state))[action]
            fixed_v      = v_net(one_hot(state)) 
            fixed_v_next = v_net(one_hot(next_state))
        if done:
            ##lagrangian mu-fixed
            lagr_v += (1-GAMMA) * v_net(one_hot(state)) + SA * fixed_mu * (reward - v_net(one_hot(state)))
            #lagrangian v-fixed
            lagr_mu -= (1-GAMMA) * fixed_v + SA * mu_net(one_hot(state))[action] * (reward - fixed_v)
        else:
            ##lagrangian mu-fixed
            lagr_v += (1-GAMMA) * v_net(one_hot(state)) + SA * fixed_mu * (reward + GAMMA * v_net(one_hot(next_state)) - v_net(one_hot(state)))
            #lagrangian v-fixed
            lagr_mu -= (1-GAMMA) * fixed_v + SA * mu_net(one_hot(state))[action] * (reward + GAMMA * fixed_v_next - fixed_v)

    optimizer_v.zero_grad()
    optimizer_mu.zero_grad()

    lagr_v.backward()
    lagr_mu.backward()

    optimizer_v.step()
    optimizer_mu.step()


# %%
###Test
# env = gym.make('FrozenLake-v0', is_slippery=False)

TEST  = 100
success = 0
for e in range(TEST):
    done  = False
    state = env.reset()
    while not done:
        #env.render()
        mu = mu_net(one_hot(state))
        action_prob = mu.detach().numpy()/mu.sum().item()
        action = np.random.choice(action_size, p = action_prob)
        #action = torch.argmax(mu)
        
        next_state, reward, done, info = env.step(action)
        state = next_state
    
    if reward == 1:
        success = success + 1
        
print(f"Total success: {success}/{TEST}")

# %%
action_prob

# %%
# %%
