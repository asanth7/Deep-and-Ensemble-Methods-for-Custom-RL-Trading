import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
import dateutil

from finta import TA

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import yfinance as yf
import os

import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error

# Suppress Tensorflow info messages (substitute for verbose=0 given custom training loop)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_preprocessing import preprocess_data
from custom_environment import customIndicatorEnv
from train_models import train_model
from test_models_and_metrics import test_models_and_metrics
from ensemble_helpers import run_ensemble_strategy
from replay_memory import replayMemory
from DQN_agent import DQNAgent
from profit_based_early_stopping import ProfitEarlyStopping

ticker = "GOOG"
data, train_df, test_df, ticker_in_snp500 = preprocess_data(ticker)

window_size = 48
frame_bound = (window_size, 2*len(train_df)//3 - 1)
custom_env = customIndicatorEnv(df=train_df, window_size=window_size, frame_bound=frame_bound)
print("Frame bound: ", frame_bound)

print("env information:")
print("> shape:", custom_env.unwrapped.shape)
print("> df.shape:", custom_env.unwrapped.df.shape)
print("> prices.shape:", custom_env.unwrapped.prices.shape)
print("> signal_features.shape:", custom_env.unwrapped.signal_features.shape)
print("> max_possible_profit:", custom_env.unwrapped.max_possible_profit())

# Baseline analysis

custom_env.reset()
custom_env.render()

observation = custom_env.reset()
while True:
  action = custom_env.action_space.sample()
  observation, reward, terminated, truncated, info = custom_env.step(action)
  done = terminated or truncated

  if done:
    print("info: ", info)
    break

# Wrap custom environment instance with wrapper
env2 = DummyVecEnv([lambda: custom_env])

timesteps = 150000
use_callbacks = ticker_in_snp500
a2c_completely_trained = train_model(model_name="A2C", train_env=env2, timesteps=timesteps,
                                     use_callbacks=use_callbacks)
a2c_sharpe = A2C.load("./models/A2C_sharpe")
a2c_calmar = A2C.load("./models/A2C_calmar")

ppo_completely_trained = train_model(model_name="PPO", train_env=env2, timesteps=timesteps,
                                     use_callbacks=use_callbacks)
ppo_sharpe = PPO.load("./models/PPO_sharpe")
ppo_calmar = PPO.load("./models/PPO_calmar")

dqn_completely_trained = train_model(model_name="DQN", train_env=env2, timesteps=timesteps,
                                     use_callbacks=use_callbacks)
dqn_sharpe = DQN.load("./models/DQN_sharpe")
dqn_calmar = DQN.load("./models/DQN_calmar")

all_models = [a2c_completely_trained, a2c_sharpe, a2c_calmar,
              ppo_completely_trained, ppo_sharpe, ppo_calmar,
              dqn_completely_trained, dqn_sharpe, dqn_calmar]


metrics, best_3 = test_models_and_metrics(all_models=all_models, group_test=True)
print(best_3)
print(metrics)

# Runs ensemble strategy with test_models_and_metrics function
run_ensemble_strategy(all_models, metrics, best_3)

print("Train dataframe shape: ", train_df.shape)

# Instantiates and wraps a training environment as an instance of the custom trading environment
train_env = customIndicatorEnv(df=train_df, window_size=window_size, frame_bound=frame_bound)
train_env = DummyVecEnv([lambda: train_env])

replay_memory_inst = replayMemory(125)
dqn_agent = DQNAgent(state_space_size=train_df.shape[1], action_space_size=2,
                     replay_memory_obj=replay_memory_inst, min_exploration_rate=0.01,
                     exploration_decay_rate=0.999, target_net_update_freq=10, learning_rate=0.0001,
                     num_training_episodes=100, batch_size=32)

profit_early_stopper = ProfitEarlyStopping(patience=5)


for episode in range(dqn_agent.num_training_episodes):
    obs = train_env.reset()
    done = False
    episode_reward = 0

    episode_loss = 0
    timesteps = 0

    info = 0

    while not done:
        current_state = obs
        action = (np.array(dqn_agent.take_action(np.array(current_state))), None)
        next_state, reward, done, info = train_env.step(action)
        current_state = np.squeeze(current_state)
        next_state = np.squeeze(next_state)
        dqn_agent.log_experience((current_state, action[0], next_state, reward, done))
        current_state = next_state

        sample_batch = dqn_agent.replay_memory.sample_batch(dqn_agent.batch_size)
        if sample_batch is not None:
            states, actions, next_states, rewards, dones = zip(*sample_batch)
            states_reshaped = np.reshape(states, (len(states), 48, 24))
            next_states_reshaped = np.reshape(next_states, (len(next_states), 48, 24))

            with tf.GradientTape() as tape:
                q_vals_current = dqn_agent.policy_network(states_reshaped)
                q_vals_next = dqn_agent.target_network(next_states_reshaped)
                q_vals_target = (q_vals_next * dqn_agent.gamma) + np.array(rewards)

                mse_loss = mean_squared_error(q_vals_target, q_vals_current)
                loss = tf.reduce_mean(mse_loss)
                episode_loss += loss.numpy()[()]
                timesteps += 1

            gradients = tape.gradient(loss, dqn_agent.policy_network.trainable_variables)

            dqn_agent.adam_optimizer.apply_gradients(zip(gradients, dqn_agent.policy_network.trainable_variables))

            if episode % dqn_agent.target_update_freq == 0:
                dqn_agent.target_network.set_weights(dqn_agent.policy_network.get_weights())

            episode_reward += reward
            current_state = next_state

    profit_early_stopper._log_profit(dqn_agent.policy_network, info[0]['total_profit'])
    dqn_agent.epsilon_decay(episode)

    print(f"Episode {episode + 1}/{dqn_agent.num_training_episodes} --- Episode Reward: {episode_reward} --- Average Loss: {episode_loss/timesteps}")
    print("Patience: ", profit_early_stopper.patience)
    print("No improvement count: ", profit_early_stopper.no_profit_improvement)
    print("-"*30)

    if profit_early_stopper._continue_training() == False:
      print("""Training performance is not improving.
      ----------ENDING TRAINING----------""")
      break

# Loading best model from training, saved by the profit early stopper
best_dqn_trained = tf.keras.models.load_model("best_deepnn_training.h5")

# Test model with same testing function and optional testing_dqn parameter for slight modification in data structure in model.predict()
test_models_and_metrics(model=best_dqn_trained, testing_dqn=True)