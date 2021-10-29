import numpy as np
from dqn import Agent
from utils import make_env, plotLearning


if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    num_games = 250
    load_checkpoint = False
    best_score = -21

    agent = Agent(gamma = 0.99, epsilon =1.0, alpha=0.0001,
    input_dims = (4,80,80), n_actions=6, mem_size=25000, eps_min = 0.02,
    batch_size=32, replace=1000, eps_dec=1e-5)

    if load_checkpoint:
        agent.load_models()

    fname = 'Pong_results.png'

    scores, epsilon_history = [], []

    n_steps =0

    for i in range(num_games):
        score = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            new_obs, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            if not load_checkpoint:
                agent.store_transition(obs, action, reward, new_obs, done)
                agent.learn()
            else:
                env.render()

            obs = new_obs
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("episode ", i, "score ", score,
        "avg score %.2f" % avg_score, "epsilon %.2f" % agent.epsilon,
        'steps ', n_steps)

        if avg_score > best_score:
            agent.save_models()
            print('avg score of %.2f is better than the best score of %.2f' % (avg_score, best_score))

            best_score = avg_score
        epsilon_history.append(agent.epsilon)
    x = [i + 1 for i in range(num_games)]
    plotLearning(x, scores, epsilon_history, fname)
    

