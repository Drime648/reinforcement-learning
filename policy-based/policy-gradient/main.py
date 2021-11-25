import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from plot import plotLearning
from PolicyGradientAgent import Agent


import gym
import matplotlib.pyplot
import numpy as np

env = gym.make('LunarLander-v2')
n_games = 3000
agent = Agent(0.0001, [8])

fname = 'REINFORCE_lunar_lr_'+ str(agent.lr) + '_' + str(n_games) + '_games'
dir = 'plots/' + fname + '.png'

scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(obs)
        new_obs,reward,done,info = env.step(action)
        score += reward
        agent.store_transition(reward)
        obs = new_obs
    agent.learn()
    scores.append(score)

    avg_score = np.mean(scores[-100:])
    print('epsiode, ', i, 'score %.2f' % score, 'avg_score %.2f' % avg_score)

x = [i+1 for i in range(len(scores))]

plotLearning(scores, dir, x)


