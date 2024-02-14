from gym import spaces
import gym_anytrading
from gym_anytrading.envs import StocksEnv
import numpy as np

class customIndicatorEnv(StocksEnv):
    def _process_data(self):
        temp_df = self.df.reset_index(drop=True, inplace=False)
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        prices = temp_df.loc[:, 'Close'].to_numpy()[start:end]

        signal_columns = self.df.columns.to_list()
        signal_features = temp_df.loc[start:end-1, signal_columns].to_numpy()
        final = np.column_stack((prices, signal_features))

        return prices.astype(np.float32), final.astype(np.float32)



    def step(self, action, return_profit=None):
      obs, reward, terminated, truncated, info = super().step(action)

      self.current_portfolio_value = self.previous_portfolio_value + reward

      if terminated or truncated:
        info['rate_of_return'] = (self.current_portfolio_value - self.starting_portfolio_value) / self.starting_portfolio_value
        info['current_portfolio_value'] = self.current_portfolio_value
        print("Episode rate of return: ", info['rate_of_return'])
        print("Total profit: ", info['total_profit'])
        print("Custom env function total profit: ", self._return_total_profit_and_reward()[0])
        print("\n")

        self.previous_portfolio_value = self.starting_portfolio_value
        self.current_portfolio_value = self.starting_portfolio_value

      else:
        self.previous_portfolio_value = self.current_portfolio_value

      return obs, reward, terminated, truncated, info

    def _return_dataframe(self):
      return self.df

    def _return_windowsize(self):
      return self.window_size

    def _reset_portfolio_on_episode(self):
      self.current_portfolio_value = self.previous_portfolio_value = self.starting_portfolio_value

    def _return_framebound(self):
      return self.frame_bound

    def _return_total_profit_and_reward(self):
      return self._total_profit, self._total_reward


    def __init__(self, df, window_size, frame_bound):
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.starting_portfolio_value = 10000
        self.current_portfolio_value = 10000
        self.previous_portfolio_value = self.starting_portfolio_value
