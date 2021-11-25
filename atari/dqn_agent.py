import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from memory import ReplayBuffer
from dqn import DQN


class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
    mem_size, batch_size, eps_min = 0.1, eps_dec=1e-5, replace=1000,
    algo=None,env_name=None,dir='tmp/dqn'):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.epsilon_min = eps_min
        self.epsilon_decrement = eps_dec
        self.replace = replace
        self.algo = algo
        self.env_name = env_name
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_cntr = 0

        self.mem = ReplayBuffer(mem_size, input_dims, n_actions)

        self.dir = dir

        self.q_eval = DQN(lr, n_actions,
        self.env_name + "_"+ algo +"_q_eval", input_dims, dir)

        self.q_next = DQN(lr, n_actions,
        self.env_name + "_"+ algo +"_q_next", input_dims, dir)

    def choose_action(self, obs):
        if(np.random.random() > self.epsilon):
            state = torch.tensor(np.array([obs]),dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            return torch.argmax(actions).item()#make it not a tensor
        else:
            return np.random.choice(self.action_space)
    
    def store_transition(self, state, action, reward, new_state, done):
        self.mem.store_transition(state, action, reward, new_state, done)
    
    def sample_mem(self):
        state, action, reward, new_state, done = self.mem.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        new_states = torch.tensor(new_state).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, new_states, dones
    
    def replace_target_network(self):
        if self.learn_step_cntr % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_min
    
    def save(self):
        self.q_eval.save()
        self.q_next.save()
    
    def load(self):
        self.q_eval.load()
        self.q_next.load()

    def learn(self):
        if self.mem.mem_cntr < self.batch_size:
            return 
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, new_states, dones = self.sample_mem()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]#takes action every row

        q_next = self.q_next.forward(new_states).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_cntr += 1

        self.decrement_epsilon()










        
