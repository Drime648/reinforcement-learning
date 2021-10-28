import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

import gym
import numpy as np

from collections import deque
from tqdm import trange


class DQN(nn.Module):
    """Some Information about DQN"""
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.lr = lr

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.out = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size,
n_actions, max_mem_size = 100000, eps_end = 0.01, eps_decrement = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_decrement = eps_decrement
        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        self.dqn = DQN(self.lr, input_dims[0], 256, 256, n_actions)

        self.state_mem = np.zeros((self.mem_size, *input_dims), np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), np.float32)

        self.action_mem = np.zeros(self.mem_size, np.int32)

        self.reward_mem = np.zeros(self.mem_size, np.float32)
        self.terminal_mem = np.zeros(self.mem_size, np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.reward_mem[index] = reward
        self.action_mem[index] = action
        self.terminal_mem[index] = done

        self.mem_cntr += 1


    def get_action(self, obs):
        if np.random.random() > self.epsilon:
            # take best known action
            state = torch.tensor(np.array(obs)).to(self.dqn.device)
            actions = self.dqn.forward(state)
            action = torch.argmax(actions).item()
        else :
            #explore actions
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        #start learning after batch size is filled up
        if self.mem_cntr < self.batch_size:
            return
        
        self.dqn.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = torch.tensor(self.state_mem[batch]).to(self.dqn.device)
        new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.dqn.device)
        reward_batch = torch.tensor(self.reward_mem[batch]).to(self.dqn.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.dqn.device)

        action_batch = self.action_mem[batch]

        q_eval = self.dqn.forward(state_batch)[batch_index, action_batch]

        q_next = self.dqn.forward(new_state_batch)

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim = 1)[0]

        loss = self.dqn.loss(q_target, q_eval).to(self.dqn.device)
        loss.backward()
        self.dqn.optimizer.step()

        self.epsilon = self.epsilon - self.eps_decrement if self.epsilon > self.eps_end \
            else self.eps_end
        








        
        
