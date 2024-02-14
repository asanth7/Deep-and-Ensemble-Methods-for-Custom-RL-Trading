import pandas as pd
import numpy as np
from main_program import train_df, test_df, env2
from custom_environment import customIndicatorEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from ensemble_helpers import ensemble_model_action

def test_models_and_metrics(model=None, model_name=None,
                num_episodes=15, group_test=False,
                all_models=None, best_sharpe=None,
                best_calmar = None, skip_desc=False,
                ensemble_testing = None, testing_dqn=False):

    if group_test == True:

      models_list = ["A2C_Training_Complete", "A2C_Sharpe", "A2C_Calmar",
                "PPO_Training_Complete", "PPO_Sharpe", "PPO_Calmar",
                "DQN_Training_Complete", "DQN_Sharpe", "DQN_Calmar"]
      # if ensemble_testing != False:
      #   models_list = ensemble_testing

      models = {model: idx for idx, model in enumerate(models_list)}
      print(models)

      results = {"Model": [], "Average Reward ($)": [],
                 "Average Profit (%)": [], "Average Rate of Return (%)": [],
                 "Buy-and-Hold Return (%)": []}

      # if ensemble_testing != False:
      #   average_reward, average_profit, average_ror, buy_and_hold_performance = test_models_and_metrics(group_test=False, ensemble_testing=ensemble_testing, skip_desc=True)
      for model in models.keys():
        average_reward, average_profit, average_ror, buy_and_hold_performance = test_models_and_metrics(model=all_models[models[model]], group_test=False, skip_desc=True)
        results['Model'].append(model)
        results['Average Reward ($)'].append(float(average_reward))
        results['Average Profit (%)'].append(float(average_profit))
        results['Average Rate of Return (%)'].append(float(average_ror))
        results['Buy-and-Hold Return (%)'].append(float(buy_and_hold_performance))

      df = pd.DataFrame(results)
      sorted = df.sort_values(by=['Average Profit (%)'], axis=0, ascending=False)
      #sorted = df.iloc[df['Average Profit (%)'].abs().argsort()]
      best_3 = sorted.index[0:3]
      sorted.reset_index(drop=True, inplace=True)

      return sorted, best_3

    window_size = env2.get_attr("_return_windowsize")[0]()
    test_env_df = test_df


    # Using first 18 days immediately after training set for testing
    train_env_upper_bound = env2.get_attr("_return_framebound")[0]()[1]
    if train_env_upper_bound < len(train_df) - window_size//2:
      frame_bound = (train_env_upper_bound + 1, train_env_upper_bound + window_size//2)
      test_env_df = train_df
    elif train_env_upper_bound == len(train_df) - 1:
      frame_bound = (window_size, 3*window_size//2 - 1)

    # Using last 18 days for testing
    # frame_bound = (len(test_df) - 18, len(test_df) - 1)

    print(f"Using frame_bound {frame_bound}")

    test_env = customIndicatorEnv(df=test_env_df, window_size=window_size, frame_bound=frame_bound)
    test_env = DummyVecEnv([lambda: test_env])

    total_rewards = []
    total_profits = []
    total_ror = []

    reset_portfolio_method = test_env.get_attr("_reset_portfolio_on_episode")[0]

    if ensemble_testing != None:
      for model in ensemble_testing:
        model.set_env(test_env)

    for _ in range(num_episodes):

      # Reward resets and profit resets to 1 (in TradingEnv)
      obs = test_env.reset()
      episode_reward = 0

      reset_portfolio_method()

      done = False
      while not done:
        if ensemble_testing != None:
          action = ensemble_model_action(ensemble_testing, obs)
        else:
          if testing_dqn:
            action = model.predict(np.array(obs), verbose=0)
            action = [np.argmax(action[0], axis=0)]
          else:
            action, _states = model.predict(np.array(obs))
        obs, rewards, done, info = test_env.step(action)
        episode_reward += rewards

      episode_profit = (info[0]['total_profit'] - 1) * 100
      # print(f"TESTING EPISODE {_} PROFIT: ", episode_profit)
      episode_ror = info[0]['rate_of_return'] * 100
      # print(f"TESTING EPISODE {_} ROR: ", episode_ror)
      # print(f"TESTING EPISODE {_} REWARD: ", episode_reward)

      total_rewards.append(episode_reward)
      total_profits.append(episode_profit)
      total_ror.append(episode_ror)

    average_reward = np.mean(total_rewards)
    average_profit = str(round(np.mean(total_profits), 4))
    average_ror = str(round(np.mean(total_ror), 4))

    return_df_method = test_env.get_attr("_return_dataframe")[0]
    start_date_close = return_df_method().iloc[frame_bound[0]]['Close']
    end_date_close = return_df_method().iloc[frame_bound[1]]['Close']

    buy_and_hold_performance = str(round(((end_date_close - start_date_close) / start_date_close) * 100, 4))

    if not skip_desc:

      print(f"==========={model_name} Results============")
      print("Average Total Reward: ", average_reward)
      print("Average Total Profit: ", average_profit + "%")
      print("Average Rate of Return: ", average_ror + "%")
      print("Buy-and-Hold Performance: ", buy_and_hold_performance + "%")
      print("\n")
      print("Best Sharpe Ratio During Training: ", best_sharpe)
      print("Best Calmar Ratio During Training: ", best_calmar)

    return average_reward, average_profit, average_ror, buy_and_hold_performance