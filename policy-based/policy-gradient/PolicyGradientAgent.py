from nn import Network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Agent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions = 4):
        self.gamma = gamma
        self.lr = lr
        self.reward_mem = []
        self.action_mem = []

        self.policy = Network(lr, input_dims, n_actions)

    def choose_action(self, obs):
        obs = torch.tensor(np.array([obs])).to(self.policy.device)

        probs = F.softmax(self.policy.forward(obs), dim = 0)

        action_probs = torch.distributions.Categorical(probs)

        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_mem.append(log_probs)

        return action.item()#open ai doesnt take pytorch tensor as input
    
    def store_transition(self, reward):
        self.reward_mem.append(reward)
    
    def learn(self):
        #G_t = R_(t+1) + R_(t_2)*gamma +  R_(t_3)*gamma^2...

        G = np.zeros_like(self.reward_mem, np.float64)
        for t in range(len(self.reward_mem)):
            sum = 0
            discount = 1
            for k in range(t, len(self.reward_mem)):
                sum += self.reward_mem[k] * discount
                discount *= self.gamma
            G[t] = sum
        G = torch.tensor(G, dtype=torch.float).to(self.policy.device)

        loss = 0

        for g, log_prob in zip(G, self.action_mem):
            loss += -g * log_prob
        loss.backward()
        self.policy.optimizer.step()

        self.action_mem = []
        self.reward_mem = []
    






