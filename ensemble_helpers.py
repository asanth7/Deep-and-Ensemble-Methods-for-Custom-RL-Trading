from test_models_and_metrics import test_models_and_metrics
import numpy as np

def run_ensemble_strategy(all_models, metrics_df, best_3):
  best_model_names = metrics_df['Model'].to_list()[0:3]
  best_models = [all_models[idx] for idx in best_3.to_list()]

  #print(best_models)
  #print(best_models != False)
  test_models_and_metrics(group_test=False, ensemble_testing=best_models, model_name="Ensemble Strategy")

  # Used for determining ensemble model action through majority voting during testing
def ensemble_model_action(models, obs):
  actions_list = [model.predict(obs)[0][0] for model in models]
  majority_action = max(set(actions_list), key=actions_list.count)
  # print("Actions: ", actions_list)
  # print("Majority action: ", majority_action)

  return (np.array(majority_action), None)