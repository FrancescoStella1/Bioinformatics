import os
from typing import Tuple

from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence, VectorSequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm
from ucsc_genomes_downloader import Genome

import config as cfg
from data_retrieval.data_retriever import get_genome
from data_visualization.visualization import plot_training_stats
from models import convolutional as conv
from models import feed_forward as ff
from models import multimodal as mm
from models import optimization as opt
from models import utils
from data_preprocessing import preprocessing as proc


def get_cnn_sequence(genome: Genome, bed: pd.DataFrame, y: np.ndarray, bacth_size: int = 512) -> MixedSequence:
    return MixedSequence(x={"sequence_data": BedSequence(genome, bed, batch_size=bacth_size)},
                         y=VectorSequence(y, batch_size=bacth_size))


def get_ffnn_sequence(X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> MixedSequence:
    return MixedSequence(x={"epigenomic_data": VectorSequence(X, batch_size=batch_size)},
                         y=VectorSequence(y, batch_size=batch_size))


def get_mmnn_sequence(genome: Genome, bed:pd.DataFrame, X: np.ndarray,
                      y: np.ndarray, batch_size: int = 512) -> MixedSequence:
    return MixedSequence(x={"sequence_data": BedSequence(genome, bed, batch_size=batch_size),
                            "epigenomic_data": VectorSequence(X, batch_size=batch_size)},
                         y=VectorSequence(y, batch_size=batch_size))


def train_models(X: pd.DataFrame, y: pd.DataFrame, n_repeats: int, n_splits: int, task: str):
    genome = get_genome()

    all_performance = []
    models = {}
    performance = {}
    n_features = len(X.columns)

    X_bed = proc.to_bed(X)

    # Apply scaling
    X = proc.apply_scaling(X)

    rkf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
    train_idxs, test_idxs = next(rkf.split(X=X, y=y))
    train_X, test_X = X.iloc[train_idxs], X.iloc[test_idxs]
    train_y, test_y = y.iloc[train_idxs].values.flatten(), y.iloc[test_idxs].values.flatten()

    train_X_bed = X_bed.iloc[train_idxs]
    test_X_bed = X_bed.iloc[test_idxs]

    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()

    ffnn_model = opt.search(train_X, train_y, test_X, test_y, n_features, ff.build_model, "ffnn")
    models["ffnn"] = ffnn_model
    cnn_model = opt.search(train_X_bed, train_y, test_X_bed, test_y, n_features, conv.build_model, "cnn")
    models["cnn"] = cnn_model
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    for fold_num, (train_idxs, test_idxs) in tqdm(enumerate(rkf.split(X, y)), leave=True,
                                                  desc="Performing K-Fold CV"):
        train_X, val_X, train_X_bed, val_X_bed, train_y, val_y = train_test_split(
            X.iloc[train_idxs],
            X_bed.iloc[train_idxs],
            y.iloc[train_idxs],
            train_size=.9
        )

        train_X = train_X.to_numpy()
        val_X = val_X.to_numpy()

        test_X = X.iloc[test_idxs].to_numpy()
        test_X_bed = X_bed.iloc[test_idxs]
        test_y = y.iloc[test_idxs]

        train_y = train_y.values.flatten()
        val_y = val_y.values.flatten()
        test_y = test_y.values.flatten()

        for model_name, train_seq, val_seq, test_seq in tqdm(
            (
                ("ffnn",
                 get_ffnn_sequence(train_X, train_y),
                 get_ffnn_sequence(val_X, val_y),
                 get_ffnn_sequence(test_X, test_y)),
                ("cnn",
                 get_cnn_sequence(genome, train_X_bed, train_y),
                 get_cnn_sequence(genome, val_X_bed, val_y),
                 get_cnn_sequence(genome, test_X_bed, test_y)),
                ("mmnn",
                 get_mmnn_sequence(genome, train_X_bed, train_X, train_y),
                 get_mmnn_sequence(genome, val_X_bed, val_X, val_y),
                 get_mmnn_sequence(genome, test_X_bed, test_X, test_y)),
            ),
            desc="Training models", leave=True
        ):

            if model_name == "mmnn":
                input_ffnn = models["ffnn"].get_layer("epigenomic_data")
                input_cnn = models["cnn"].get_layer("sequence_data")
                last_hidden_ffnn = models["ffnn"].get_layer("ffnn_dp_3")
                last_hidden_cnn = models["cnn"].get_layer("cnn_dp_final")
                
                assert(input_ffnn is not None)
                assert(input_cnn is not None)
                assert(last_hidden_ffnn is not None)
                assert(last_hidden_cnn is not None)
                
                model = mm.initialize_mmnn_model(input_epigenomic_data=input_ffnn,
                                                 input_sequence_data=input_cnn,
                                                 last_hidden_ffnn=last_hidden_ffnn,
                                                 last_hidden_cnn=last_hidden_cnn
                                                 )
                
            else:
                model = models[model_name]
                
            utils.plot(f"{model_name}_architecture_{task}", model)
            _, performance = _train_model(model, train_seq, val_seq, test_seq, fold_num, task)
            all_performance.append(performance)
    all_performance = pd.concat(all_performance)
    print(all_performance.drop(columns=["loss"]))
    utils.save_training_results(all_performance, task)
    plot_training_stats(all_performance, task)
    utils.perform_wilcoxon_test(all_performance)


def _train_model(model: Model, train_seq: MixedSequence, val_seq: MixedSequence, test_seq: MixedSequence,
                 fold_num: int, task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    checkpoints_dir = os.path.join(cfg.models["checkpoints_dir"], model.name)
    history = pd.DataFrame(model.fit(train_seq, validation_data=val_seq, epochs=cfg.models["epochs"],
                                     verbose=True, callbacks=[EarlyStopping("val_loss",
                                                                            min_delta=cfg.models["min_delta"],
                                                                            mode="min", patience=10,
                                                                            restore_best_weights=True),
                                                              ModelCheckpoint(checkpoints_dir, save_weights_only=False,
                                                                              monitor="val_loss", mode="min",
                                                                              save_best_only=True),
                                                              ]).history)
    train_eval = dict(zip(model.metrics_names, model.evaluate(train_seq, verbose=False)))
    validation_eval = dict(zip(model.metrics_names, model.evaluate(val_seq, verbose=False)))
    test_eval = dict(zip(model.metrics_names, model.evaluate(test_seq, verbose=False)))
    train_eval["run_type"] = "train"
    validation_eval["run_type"] = "validation"
    test_eval["run_type"] = "test"

    for evaluation in (train_eval, validation_eval, test_eval):
        evaluation["fold_number"] = fold_num
        evaluation["model_name"] = model.name
        evaluation["task"] = task

    evaluations = pd.DataFrame([train_eval, validation_eval, test_eval])
    return history, evaluations
