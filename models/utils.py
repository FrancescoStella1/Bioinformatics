import os

from BorutaShap import BorutaShap
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import  KFold
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tqdm.auto import tqdm
from xgboost import XGBRFRegressor

import config as cfg


def perform_wilcoxon_test(all_performance: pd.DataFrame):
    """
    Performs the signed rank Wilcoxon Test on the history of the models.
    The history contains training, validation and evaluation data.
    N.B.: test data is used to perform the Wilcoxon Test and it should be a sample
    large enough to guarantee reliable results.

    :param all_performance: the training history of the models.
    
    """
    window_size = cfg.data_retrieval["window_size"]
    task = all_performance["task"].iloc[0]
    models = all_performance["model_name"].unique()
    # all_performance = all_performance[all_performance["run_type"]=="test"]
    savepath = os.path.join(cfg.models["performance_savepath"], f"wilcoxon_win_{window_size}.csv")
    df = pd.DataFrame(columns=["Task", "Window Size", "First Model", "Second Model", "p_value", "Statistically significant difference?"])
    for first in models:
        for second in models:
            if np.where(models==second)<=np.where(models==first):
                continue
            first_performance = all_performance[all_performance["model_name"]==first]["mse"]
            second_performance = all_performance[all_performance["model_name"]==second]["mse"]
            # print(first_performance.values, second_performance.values)
            _, p_value = wilcoxon(first_performance, second_performance)
            if p_value < 5e-2:
                if first_performance.mean() < second_performance.mean():
                    df = pd.concat([df, pd.DataFrame([[task, window_size, first, second, p_value, "Yes"]], columns=df.columns)])
                else:
                    df = pd.concat([df, pd.DataFrame([[task, window_size, second, first, p_value, "Yes"]], columns=df.columns)])
            else:
                df = pd.concat([df, pd.DataFrame([[task, window_size, first, second, p_value, "No"]], columns=df.columns)])
    df.to_csv(savepath, index=False, mode="a")
    print("Wilcoxon signed-rank Test completed.")


def plot(filename: str, model: Model) -> None:
    """
    Plots the graph of the model.

    :param filename: The name of the image to save (without extension).
    :param model: The keras Model to plot.
    """
    imgs_path = cfg.images["path"]
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    img_path = os.path.join(imgs_path, f"{filename}.png")
    plot_model(model, dpi=128,  # show_layer_activations=True,
               show_layer_names=True, show_shapes=True, to_file=img_path)

def run_boruta_shap(X: pd.DataFrame, y: pd.DataFrame, n_splits: int, task: str, n_trials: int = 100) -> list:
    """
    Performs BorutaSHAP algorithm for feature selection, using Random Forest as model.

    Parameters
    ----------
    X: pd.DataFrame
        Dataframe containing the data and the features from which to extract a subset.
    y: pd.DataFrame
        Dataframe containing the labels.
    n_splits: int
        Number of splits to perform KFold.
    task: str
        Name of the task, i.e. 'enhancers' or 'promoters'.
    n_trials: int
        Number of trials for Random Forest fitting (default: 100).    
    """
    window_size = cfg.data_retrieval["window_size"]
    filepath = os.path.join(cfg.feature_selection["features_dir"], f"{task}_selected_features_win_{window_size}.txt")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            features = f.read().splitlines()
        return features
    model = XGBRFRegressor(random_state=42, objective="reg:squarederror", tree_method="gpu_hist")
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    selected_features = set()
    if not os.path.exists(cfg.feature_selection["features_dir"]):
        os.mkdir(cfg.feature_selection["features_dir"])
    for k, (train_idxs, test_idxs) in tqdm(enumerate(k_fold.split(X, y)), leave=True, desc="Boruta SHAP K-Fold CV"):
        print(type(y.iloc[train_idxs]))
        feature_selector = BorutaShap(model=model, importance_measure="shap", classification=False)
        feature_selector.fit(X=X.iloc[train_idxs].squeeze(), y=y.iloc[train_idxs].squeeze(), n_trials=n_trials, random_state=42)
        feature_selector.plot(which_features="all")
        selected_features = selected_features.union(set(feature_selector.Subset().columns))
        # selected_features.append(sorted(feature_selector.Subset().columns))
        print(f"Selected features at Fold {k+1}: {feature_selector.Subset().columns}")
    features_str = "\n".join([f for f in selected_features])
    with open(filepath, "w") as f:
        f.write(features_str)
    return selected_features


def save_training_results(results: pd.DataFrame, task: str):
    """
    Saves the training history of the models.
    
    :param results: dataframe containing the history to save.
    :param task: task (enhancers or promoters) related to the history.
    """
    savepath = cfg.models["performance_savepath"]
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    results.to_csv(os.path.join(savepath, f"results_{task}.csv"), index=False, mode="a")
