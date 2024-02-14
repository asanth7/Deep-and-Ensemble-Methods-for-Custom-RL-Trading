from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class calmarRatioCallback(BaseCallback):
  def __init__(self, env, file_path, eval_freq=5000, verbose=1):
    super().__init__(verbose=verbose)
    self.env = env
    self.file_path = file_path
    self.eval_freq = eval_freq
    self.verbose = verbose
    self.max_drawdown = float('-inf')
    self.best_calmar_ratio = float('-inf')

  def _on_step(self):
    if self.n_calls % self.eval_freq == 0:
      calmar_ratio = self._evaluate()
      if calmar_ratio > self.best_calmar_ratio:
        self.best_calmar_ratio = calmar_ratio
        if self.verbose > 0:
          print(f"BEST MODEL FOUND: Calmar ratio = {calmar_ratio:.4f}")
          print(f"Saving model to {self.file_path}")
        self.model.save(self.file_path)

    return True

  def _evaluate(self):
    total_returns = []
    all_max_drawdowns = []

    return_df_method = self.env.get_attr("_return_dataframe")[0]
    return_window_size_method = self.env.get_attr("_return_windowsize")[0]
    episode_duration = return_window_size_method() # datetime.timedelta(days=return_window_size_method())
    reset_portfolio_method = self.env.get_attr("_reset_portfolio_on_episode")[0]

    for episode in range(16):
      obs = self.env.reset()
      done = False

      reset_portfolio_method()

      # print("Computing close prices")
      close_prices = return_df_method().iloc[episode:episode + episode_duration]['Close']

      while not done:
        action, _state = self.model.predict(np.array(obs))
        obs, reward, done, info = self.env.step(action)

      # Annualized return from a single episode (18 days/timesteps) with geometric compounding
      #annualized_return = ((1 + info[0]['rate_of_return'])**(365 / 18)) - 1
      total_returns.append(info[0]['rate_of_return'])

      max_drdwn = self._max_drawdown(close_prices)
      if max_drdwn != 0:
        all_max_drawdowns.append(max_drdwn)

    avg_rate_of_return = np.mean(total_returns)
    max_drawdown = np.mean(all_max_drawdowns)

    calmar_ratio = avg_rate_of_return / max_drawdown
    return calmar_ratio

  def _max_drawdown(self, time_series):
    # print("\n")

    # return_window_size_method = self.env.get_attr("_return_windowsize")[0]
    # assert len(time_series) == return_window_size_method()

    max_drawdown = 0
    peak = time_series[0]

    for value in time_series[1:]:
      # print(value)
      if value > peak:
        peak = value
      else:
        drawdown = (peak - value)/peak
        if drawdown > max_drawdown:
          max_drawdown = drawdown

    # print("Max Drawdown: ", max_drawdown)

    return max_drawdown

  def _return_best_calmar(self):
    return self.best_calmar_ratio
