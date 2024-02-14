from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import A2C, PPO, DQN
from sharpe_ratio_callback import sharpeRatioCallback
from calmar_ratio_callback import calmarRatioCallback
from test_models_and_metrics import test_models_and_metrics

def train_model(model_name, train_env, timesteps, learning_rate=0.0001,
                use_callbacks=True):

  if use_callbacks:
    sharpe_callback = sharpeRatioCallback(train_env, file_path=f'./models/{model_name}_sharpe')
    calmar_callback = calmarRatioCallback(train_env, file_path=f'./models/{model_name}_calmar')
    callbacks = [sharpe_callback, calmar_callback]

  if model_name == "A2C":

    model = A2C('MlpPolicy', train_env, verbose=1)
    if use_callbacks:
      model.learn(total_timesteps=timesteps, callback=callbacks)
    else:
      model.learn(total_timesteps=timesteps)
    best_model = A2C.load(f'./models/{model_name}_sharpe')
    model.save(f"./models/{model_name}_no_callbacks")

  if model_name == "PPO":
    model = PPO('MlpPolicy', train_env, verbose=1)
    if use_callbacks:
      model.learn(total_timesteps=timesteps, callback=callbacks)
    else:
      model.learn(total_timesteps=timesteps)
    best_model = PPO.load(f'./models/{model_name}_sharpe')
    model.save(f"./models/{model_name}_no_callbacks")


  if model_name == "DQN":
    model = DQN("MlpPolicy", train_env, verbose=1)
    if use_callbacks:
      model.learn(total_timesteps=timesteps, callback=callbacks)
    else:
      model.learn(total_timesteps=timesteps)
    best_model = DQN.load(f"./models/{model_name}_sharpe")
    model.save(f"./models/{model_name}_no_callbacks")


  best_sharpe = sharpe_callback._return_best_sharpe()
  best_calmar = calmar_callback._return_best_calmar()

  _, _, _, _ = test_models_and_metrics(model=best_model, model_name=model_name,
                                    best_sharpe=best_sharpe, best_calmar=best_calmar)

  return model
