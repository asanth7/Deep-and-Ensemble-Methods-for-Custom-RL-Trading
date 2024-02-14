import numpy as np

class ProfitEarlyStopping:
  def __init__(self, patience):
    self.patience = patience
    self.no_profit_improvement = 0
    self.best_profit = -np.inf

  def _log_profit(self, model, profit):
    if profit > self.best_profit:
      self.best_profit = profit
      self.no_profit_improvement = 0
      model.save("best_deepnn_training.h5")
      print("------FOUND NEW BEST MODEL: SAVING------")
    else:
      self.no_profit_improvement += 1

  def _continue_training(self):
    return self.no_profit_improvement <= self.patience