import os
from typing import Callable, Tuple

from tensorflow.keras import Model
import keras_tuner as kt
from keras_tuner.tuners import BayesianOptimization
import numpy as np

import config as cfg
from data_retrieval.data_retriever import get_genome
from models import convolutional as conv
from models import feed_forward as ff
from models.trainer import get_cnn_sequence, get_ffnn_sequence


def _get_tuner(build_model: Callable, max_trials: int, n_features: int, model_name: str) -> kt.tuners.bayesian.BayesianOptimization:
    """
    Returns tuner for bayesian optimization.
    
    Parameters
    ----------
    build_model: function
        Function used to build the model.
    max_trials: int
        Number of maximum trials to perform during the search.
    model_name: str
        Name of the model used for the hyperparameters search.
    
    Returns
    -------
    bayesian_optimization: Tuner
    """
    hp = kt.HyperParameters()
    hp.values = {"input_shape": n_features}
    return BayesianOptimization(build_model, "val_loss", directory=f"optimizers_checkpoints/{model_name}",
                                max_trials=max_trials, hyperparameters=hp, project_name="bayesian_opt")


def _get_sequences(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, model_name: str) -> Tuple[np.ndarray]:
    """
    Returns the sequences for the feed-forward or cnn models.
    
    Parameters
    ----------
    train_X: np.ndarray
        Datapoints train set.
    train_y: np.ndarray
        Labels train set.
    test_X: np.ndarray
        Datapoints test set.
    test_y: np.ndarray
        Labels test set.
    model_name: str
        Name of the model for which to retrieve sequences.
    
    Returns
    -------
    train_seq, test_seq: Tuple[np.ndarray]
        Train and test sequences.
    """
    train_seq = None
    test_seq = None
    if model_name == "ffnn":
        train_X = train_X
        test_X = test_X
        train_seq = get_ffnn_sequence(train_X, train_y)
        test_seq = get_ffnn_sequence(test_X, test_y)
    elif model_name == "cnn":
        genome = get_genome()
        train_seq = get_cnn_sequence(genome, train_X, train_y)
        test_seq = get_cnn_sequence(genome, test_X, test_y)
    return train_seq, test_seq


def load_hyperparameters(model_name: str, n_features: int) -> kt.HyperParameters:
    """
    Loads the hyperparameters for a previously optimized model.

    Parameters
    ----------
    model_name: str
        Name of the model for which to retrieve the hyperparameters.
    
    Returns
    -------
    hyperparams: kt.HyperParameters
        Best hyperparameters found for the given model.
    """
    print("Loading hyperparameters found by optimizer...")
    workdir = os.path.join(cfg.optimizers["checkpoints_dir"], model_name)
    obj = "val_loss"
    if model_name == "ffnn":
        hp = kt.HyperParameters()
        hp.values = {"input_shape": n_features}
        tuner = BayesianOptimization(ff.build_model, objective=kt.Objective(obj, direction="min"),
                                     overwrite=True, directory=workdir, hyperparameters=hp, project_name="bayesian_opt")
    else:
        tuner = BayesianOptimization(conv.build_model, objective=kt.Objective(obj, direction="min"),
                                     overwrite=False, directory=workdir, project_name="bayesian_opt")

    return tuner.get_best_hyperparameters(1)[0]


def search(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, n_features: int, build_model: Callable, model_name: str) -> Model:
    """
    Performs Bayesian Optimization on a specific model.
    
    Parameters
    ----------
    train_X: np.ndarray
        Datapoints train set.
    train_y: np.ndarray
        Labels train set.
    test_X: np.ndarray
        Datapoints test set.
    test_y: np.ndarray
        Labels test set.
    n_features: int
        Number of retained features after feature selection.
    build_model: function
        Function used to build the model.
    model_name: str
        Name of the model for which to perform the optimization.
    
    Returns
    -------
    model: Model
        Best model found with Bayesian Optimization.
    """
    if os.path.exists(os.path.join(cfg.optimizers["checkpoints_dir"], model_name)):
        max_trials = cfg.optimizers["cnn_max_trials"]
        if model_name == "ffnn":
            max_trials = cfg.optimizers["ffnn_max_trials"]
        tuner = _get_tuner(build_model, max_trials, n_features, model_name)
        best_hyperparams = load_hyperparameters(model_name, n_features)
        model = tuner.hypermodel.build(best_hyperparams)
        return model

    epochs = cfg.optimizers["cnn_epochs"]
    max_trials = cfg.optimizers["cnn_max_trials"]
    if model_name == "ffnn":
        epochs = cfg.optimizers["ffnn_epochs"]
        max_trials = cfg.optimizers["ffnn_max_trials"]

    train_seq, test_seq = _get_sequences(train_X, train_y, test_X, test_y, model_name)
    print(len(train_seq), len(test_seq))
    tuner = _get_tuner(build_model, max_trials, n_features, model_name)
    tuner.search(train_seq, epochs=epochs, validation_data=test_seq)
    best_hyperparams = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hyperparams)
    return model
