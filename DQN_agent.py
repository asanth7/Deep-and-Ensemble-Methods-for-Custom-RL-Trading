import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Reshape, LSTM, Bidirectional, MaxPooling2D
from keras.optimizers import Adam
from random import random
import numpy as np

class DQNAgent:
  def __init__(self, state_space_size, action_space_size, replay_memory_obj,
               min_exploration_rate, exploration_decay_rate,
               target_net_update_freq, learning_rate, num_training_episodes, batch_size):
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size
    self.replay_memory = replay_memory_obj
    self.gamma = 0.999
    self.start_epsilon = 1
    self.current_epsilon = self.start_epsilon
    self.min_epsilon = min_exploration_rate
    self.epsilon_decay_rate = exploration_decay_rate
    self.adam_optimizer = Adam(learning_rate=learning_rate)
    self.policy_network = self._build_policy_net()
    self.policy_network.compile(optimizer = self.adam_optimizer, loss="mse")
    self.target_network = self._build_target_net()
    self.target_update_freq = target_net_update_freq
    self.learning_rate = learning_rate
    self.num_training_episodes = num_training_episodes
    self.batch_size = batch_size

  def _build_policy_net(self, building_target=None):
    network = Sequential([
        Dense(256, input_shape=(48, self.state_space_size + 1)),
        Dense(128, activation="relu"), #KEEP without relu FOR CURRENT BEST (1.07 testing profit)
        #Dense(64, activation="relu"),
        Dense(self.state_space_size, activation="relu"),
        #Reshape((self.state_space_size, 1)),
        Conv1D(48, 3, strides=1, padding="same", activation="relu"),
        #Reshape((1, 48, 48)),
        #MaxPooling2D(pool_size=(2, 2), padding="same"),
        Conv1D(96, 3, strides=1, padding="same", activation="relu"),
        #Reshape((1, 24, 96)),
        #MaxPooling2D(pool_size=(2, 2), padding="same"),
        Conv1D(48, 3, strides=1, padding="same", activation="relu"),
        #Reshape((1, 12, 48)),
        #MaxPooling2D(pool_size=(2, 2), padding="same"),
        Flatten(),
        Reshape((1, 2304)), # WAS (1, 2304) FOR BEST (testing 1.07)
        #Dense(64, activation="relu"),
        Bidirectional(LSTM(64, return_sequences=False)),
        #Bidirectional(LSTM(32, return_sequences=False)),
        Dense(self.action_space_size)
    ])

    if building_target != None:
      print(network.summary())

    return network

  def _build_target_net(self):
    target_net = self._build_policy_net(building_target=True)
    target_net.set_weights(self.policy_network.get_weights())

    return target_net

  def epsilon_decay(self, episode_num):
    self.current_epsilon = self.min_epsilon + (self.start_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate*episode_num)

  def log_experience(self, experience):
    self.replay_memory.add_experience(experience)

  def take_action(self, state):
    if self.current_epsilon > random.random():
      return random.randrange(self.action_space_size)
    return self.policy_network.predict(state, verbose=0).argmax(axis=1).item()