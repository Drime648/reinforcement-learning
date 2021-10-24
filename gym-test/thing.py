import gym
import numpy as np
import random
import math

env = gym.make('CartPole-v0')

no_buckets = (1, 1, 6, 3)
no_actions = env.action_space.n

state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_value_bounds[1] = [-0.5, 0.5]
state_value_bounds[3] = [-math.radians(50), math.radians(50)]


state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_value_bounds[1] = [-0.5, 0.5]
state_value_bounds[3] = [-math.radians(50), math.radians(50)]

action_index = len(no_buckets)

q_table = np.zeros(no_buckets + (no_actions,))

min_explore_rate = 0.01
min_learning_rate = 0.1

max_episodes = 1000
max_time_steps = 250
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0


def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state_value])
    return action