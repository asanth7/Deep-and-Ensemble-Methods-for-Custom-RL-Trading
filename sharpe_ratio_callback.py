from stable_baselines3.common.callbacks import BaseCallback
import datetime
import warnings
import numpy as np
import yfinance as yf

class sharpeRatioCallback(BaseCallback):
  def __init__(self, env, file_path, eval_freq=5000, verbose=1):
    super().__init__(verbose=verbose)
    self.env = env
    self.file_path = file_path
    self.eval_freq = eval_freq
    self.verbose=verbose
    self.best_sharpe_ratio = float('-inf')
    self.treasury_rates = []

  def _on_step(self):
    if self.n_calls % self.eval_freq == 0:
      sharpe_ratio = self._evaluate()
      if sharpe_ratio > self.best_sharpe_ratio:
        self.best_sharpe_ratio = sharpe_ratio
        if self.verbose > 0:
          print(f"BEST MODEL FOUND: Sharpe ratio = {sharpe_ratio:.4f}")
          print(f"Saving model to {self.file_path}")
        self.model.save(self.file_path)

    return True

  def _evaluate(self):
    total_returns = []

    return_window_size_method = self.env.get_attr("_return_windowsize")[0]
    episode_duration = datetime.timedelta(days=return_window_size_method())
    return_df_method = self.env.get_attr("_return_dataframe")[0]
    reset_portfolio_method = self.env.get_attr("_reset_portfolio_on_episode")[0]

    # REMEMBER that 1 episode is 18 days (timesteps)

    for episode in range(50):

      # Reward = 0, _total_profit = 0
      obs = self.env.reset()
      annualized_return = 0
      done = False
      # episode_return = 0

      # Current and previous portfolio values (instance variables) are reset to starting value of 10000
      reset_portfolio_method()

      start_date = return_df_method().index[episode]
      end_date = start_date + episode_duration

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        treasury_data = yf.Ticker("^IRX").history(start=start_date.strftime('%Y-%m-%d'),
                                                  end=end_date.strftime("%Y-%m-%d"))

        if not treasury_data.empty:
          # print("Number of treasury rates found: ", len(treasury_data))
          # Taking average of treasury rates from 18 days as an approximation
          # rates = treasury_data['Close']
          avg_return_rate = np.mean(treasury_data['Close']) / 100
          # Annualizing treasury rates from 18 day rate
          # rate_geometric_mean = (np.prod([1 + rate for rate in rates])**(1 / 18)) - 1
          annualized_episode_treasury_rate = (1 + avg_return_rate)**(252 / return_window_size_method()) - 1
          self.treasury_rates.append(annualized_episode_treasury_rate)

      while not done:
        action, _state = self.model.predict(np.array(obs))
        obs, reward, done, info = self.env.step(action)

        if done:
          # Annualizing rate of return from 18-day rate using geometric compounding
          annualized_return = ((1 + info[0]['rate_of_return'])**(252 / return_window_size_method())) - 1
          #without_prod = (1 + ([1 + info[0]['rate_of_return']] * (365 / 18)))**(18/365) - 1
          #print("WITHOUT np.prod: ", without_prod)
          # episode_return = info[0]['rate_of_return']

      total_returns.append(annualized_return)

    # episode_daily_returns = [(1 + episode_return)**(1/365)-1 for episode_return in total_returns]
    # annualized_treasury_rates = [(1 + rate)**(365)-1 for rate in self.treasury_rates]

    mean_return_across_episodes = np.mean(total_returns)
    print("Mean episode rate of return: ", mean_return_across_episodes)
    risk_free_return = np.mean(self.treasury_rates)
    print("Mean treasury rate across episodes: ", risk_free_return)
    excess_return = [return_rate - treasury_rate for return_rate, treasury_rate in zip(total_returns, self.treasury_rates)]
    print("Excess return: ", excess_return)
    std_excess_return = np.std(excess_return)
    print("STD excess return: ", std_excess_return)
    sharpe_ratio = (mean_return_across_episodes - risk_free_return)/std_excess_return
    print("SHARPE ratio found: ", sharpe_ratio)

    return sharpe_ratio

  def _return_best_sharpe(self):
    return self.best_sharpe_ratio