
data_retrieval = {
    "assembly": "hg38",
    "cell_line": "GM12878",
    "dataset": "fantom",
    "root": "datasets",
    "window_size": 256
}

feature_selection = {
    "colsample_bynode": .8,
    "features_dir": "./feature_selection",
    "early_stopping_rounds": 5,
    "eval_metric": "rmse",
    "max_depth": 10,
    "num_parallel_tree": 100,
    "objective": "reg:squaredlogerror",
    "predictor": "gpu_predictor",
    "tree_method": "gpu_hist",
    "subsample": .6,
    "verbosity": 1
}

images = {
    "path": "images"
}

models = {
    "checkpoints_dir": "model_checkpoints",
    "epochs": 100,
    "learning_rate": 1e-3,
    "loss": "mse",
    "metrics": ["mse"],
    "min_delta": 1e-3,
    "optimizer": "nadam",
    "performance_savepath": "results"
}

optimizers = {
    "checkpoints_dir": "optimizers_checkpoints",
    "cnn_epochs": 10,
    "cnn_max_trials": 15,
    "ffnn_epochs": 15,
    "ffnn_max_trials": 20
}

preprocessing = {
    "corr_threshold": .05,
    "p_value_threshold": .01
}

