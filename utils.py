import json
import os

import pandas as pd
import torch

from config import CONFIG


def get_trials_results() -> pd.DataFrame:
    """
    Returns the results of the trials for model selection.
    
    Results
    -------
    trials: pd.DataFrame
        Dataframe containing trials results.
    """
    try:
        trials = pd.DataFrame([], columns=["trial", "val loss", "val f1 score"])
        basepath = CONFIG.get("OPTIMIZATION", "trials_dir")
        for t in os.listdir(basepath):
            try:
                result_path = os.path.join(basepath, t)
                with open(os.path.join(result_path, "result.json")) as f:
                    result = json.load(f)
                trials = pd.concat([trials, pd.DataFrame([
                    [t, result["val_loss"], result["val_f1_score"]]], columns=trials.columns)
                ])
            except:
                pass
        trials = trials.sort_values(by="val f1 score", ascending=False)
        return trials

    except Exception as ex:
        print(ex)
        pass
    

def load_class_weights() -> torch.Tensor:
    """
    Loads previously estimated class weights to mitigate class imbalance problems.

    Returns
    -------
    class_weights: dict
        Dict containing class labels as keys and the corresponding weights as values.
    """
    try:
        class_weights_path = CONFIG.get("TRAINING", "class_weights_linux_abspath")
        class_weights_path = os.path.join(class_weights_path, "class_weights.pt")
        return torch.load(class_weights_path)

    except Exception as ex:
        print(ex)
