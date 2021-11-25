import gym

import numpy as np

from dqn_agent import DQNAgent
from utils import make_env, plot_learning_curve


env = make_env('PongNoFrameskip-v4')
best_score = -np.inf
load_checkpoint = False
n_games = 500
agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

if load_checkpoint:
    agent.load()


fname = 'DQN_PONG'

fig_file = 'graphs/' + fname + '.png'

n_steps = 0

scores, eps_history, steps_arr = [],[],[]

for i in range(n_games):
    done = False
    score = 0
    obs = env.reset()

    while not done:
        action = agent.choose_action(obs)
        new_obs, reward, done, info = env.step(action)
        score += reward

        if not load_checkpoint:
            agent.store_transition(obs, action, reward, new_obs, int(done))
            agent.learn()
        obs = new_obs
        n_steps += 1
    
    scores.append(score)
    steps_arr.append(n_steps)

    avg_score = np.mean(scores[-100:])

    print('episode: ', i, ' score: ', score,
     ' average score: %.2f best score: %.2f epsilon: %.2f' %
     (avg_score, best_score, agent.epsilon), 'steps: ', n_steps)
    
    if avg_score > best_score:
        if not load_checkpoint:
            agent.save()
        best_score = avg_score
    
    eps_history.append(agent.epsilon)

plot_learning_curve(steps_arr, scores, eps_history, fig_file)

        


