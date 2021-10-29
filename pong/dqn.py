import keras
from keras.layers import Activation, Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras import backend as K
import numpy as np
import gym


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        super().__init__()
        self.mem_size = max_size
        self.mem_counter = 0

        self.state_mem = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_shape), dtype = np.float32)

        self.action_mem = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_mem = np.zeros(self.mem_size, np.float32)
        self.terminal_mem = np.zeros(self.mem_size, np.uint8)

    def store_transition(self, state, action, reward, new_state, done):
        i = self.mem_counter % self.mem_size

        self.state_mem[i] = state
        self.new_state_mem[i] = new_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.terminal_mem[i] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace = False)

        states = self.state_mem[batch]
        new_states = self.new_state_mem[batch]

        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]

        return states, actions, rewards, new_states, dones
    
def make_dqn(lr, n_actions, input_dims, fc1_dims):
    model = Sequential([
        Conv2D(32, kernel_size = 8,strides= 4, activation="relu", 
        input_shape = (*input_dims,), data_format='channels_first'),

        Conv2D(64, kernel_size = 4, strides=2, activation="relu", 
        data_format='channels_first'),
        
        Conv2D(64, kernel_size = 3, strides=1, activation="relu", 
        data_format='channels_first'),

        Flatten(),

        Dense(fc1_dims, activation="relu"),

        Dense(n_actions)

    ])
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
    replace, input_dims, eps_dec = 1e-5, eps_min = 0.01, mem_size = 1000000,
    eval_f_name = 'eval.h5', target_f_name = 'target.h5'):

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = eps_dec
        self.epsilon_min = eps_min
        self.batch_size = batch_size
        self.replace = replace

        self.target_file = target_f_name
        self.eval_file = eval_f_name

        self.learn_step = 0
        self.mem = ReplayBuffer(mem_size, input_dims)

        self.q_eval = make_dqn(alpha, n_actions, input_dims, 512)
        self.q_next = make_dqn(alpha, n_actions, input_dims, 512)

    def replace_target_network(self):
        if self.replace != 0 and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
    
    def store_transition(self, state, action, reward, new_state, done):
        self.mem.store_transition(state, action, reward, new_state, done)
    
    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([obs], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if self.mem.mem_counter > self.batch_size:
            state, action, reward, new_state, done = self.mem.sample_buffer(self.batch_size)

            self.replace_target_network()

            q_eval = self.q_eval.predict(state)
            q_next = self.q_next.predict(new_state)

            q_next[done] = 0.0

            indices = np.arange(self.batch_size)

            q_target = q_eval[:]
            q_target[indices, action] = reward + self.gamma * np.max(q_next, axis=1)

            self.q_eval.train_on_batch(state, q_target)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_dec
            
            self.learn_step += 1

    def save_models(self):
        self.q_eval.save(self.eval_file)
        self.q_next.save(self.target_file)
        print('saving models')

    def load_models(self):
        self.q_eval = load_model(self.eval_file)
        self.q_next = load_model(self.target_file)




class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip =4):
        super(SkipEnv, self).__init__(env)

        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (80,80,1), dtype = np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        n_frame = np.reshape(frame, frame.shape).astype(np.float32)
        #grayscale
        n_frame = 0.299 * n_frame[:,:,0] + 0.587*n_frame[:,:,1] + 0.114*n_frame[:,:,2]

        #cropping
        n_frame = n_frame[35:195:2, ::2].reshape(80,80,1)
        return n_frame.astype(np.uint8)
    
class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0,
         shape = (self.observation_space.shape[-1], 
                    self.observation_space.shape[0],
                    self.observation_space.shape[1]), dtype = np.float32)
    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype = np.float32
        )
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype = np.float32)
        return self.observation(self.env.reset())
    
    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer

def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)


