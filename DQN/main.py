import gym
from dqn import Agent
from utils import plotLearning
import numpy as np

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
    eps_end=0.01, input_dims=[8], lr=0.001)

    scores, epsilon_history = [], []

    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.get_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, new_obs, done)
            agent.learn()
            obs = new_obs
        scores.append(score)
        epsilon_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode: ', i, 'score %.2f' % score,
                'average score: %.2f' % avg_score,
                'epsilon: %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    f_name = 'lunar_lander.png'
    plotLearning(x, scores, epsilon_history, f_name)


