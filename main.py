import argparse
from itertools import product

import numpy as np
from tqdm.auto import tqdm

from data_preprocessing import preprocessing as proc
from data_retrieval import data_retriever
from data_visualization import visualization as vis
from models import trainer, utils

import config as cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-plotting', action='count', default=0, help="Enable or disable plotting")
    args = parser.parse_args()

    cell_line = cfg.data_retrieval["cell_line"]
    window_size = cfg.data_retrieval["window_size"]
    # Data retrieval and constant features check
    x_enhancers, y_enhancers = data_retriever.get_enhancers("mean", window_size)
    x_promoters, y_promoters = data_retriever.get_promoters("mean", window_size)

    assert(not proc.check_constant_features(x_enhancers))
    assert(not proc.check_constant_features(x_promoters))

    # Data imputation
    print("\nEnhancers samples imputation:")
    x_enhancers = proc.imputation(x_enhancers, "knn")
    print("\n--\nEnhancers labels imputation:")
    y_enhancers = proc.imputation(y_enhancers, "knn")
    print("\n----\nPromoters samples imputation:")
    x_promoters = proc.imputation(x_promoters, "knn")
    print("\n--\nPromoters labels imputation:")
    y_promoters = proc.imputation(y_promoters, "knn")

    print("\nData imputation performed.")

    assert(not x_enhancers.isna().values.any())
    assert(not y_enhancers.isna().values.any())
    assert(not x_promoters.isna().values.any())
    assert(not y_promoters.isna().values.any())

    # Labels clipping
    if args.enable_plotting > 0:
        vis.labels_clipping(y_enhancers, "Enhancers")
        vis.labels_clipping(y_promoters, "Promoters")

    y_enhancers_arr = np.array(y_enhancers[cell_line].values.tolist())
    y_enhancers[cell_line] = np.where(y_enhancers_arr > 3., 3., y_enhancers_arr).tolist()
    y_promoters_arr = np.array(y_promoters[cell_line].values.tolist())
    y_promoters[cell_line] = np.where(y_promoters_arr > 50., 50., y_promoters_arr).tolist()

    x_enhancers_scaled = proc.apply_scaling(x_enhancers)
    x_promoters_scaled = proc.apply_scaling(x_promoters)
    print("Scaling performed.")

    # Train and test splits before any correlation test
    X_train_enhancers, X_test_enhancers, y_train_enhancers, y_test_enhancers = proc.train_test_splits(
        x_enhancers_scaled, y_enhancers, .8)

    X_train_promoters, X_test_promoters, y_train_promoters, y_test_promoters = proc.train_test_splits(
        x_promoters_scaled, y_promoters, .8)

    regions = ["enhancers", "promoters"]
    tests = ["Pearson", "Spearman"]

    for region, test in tqdm(list(product(regions, tests)), desc="Performing tests", leave=False):
        X_train = None
        y_train = None
        if region == "enhancers":
            X_train = X_train_enhancers
            y_train = y_train_enhancers
        else:
            X_train = X_train_promoters
            y_train = y_train_promoters
        highly_correlated_features, correlated_features = proc.find_ff_correlations(X_train, test)
        if highly_correlated_features.empty:
            print(f"No highly correlated features found in {region} using {test} test.")
        else:
            print(f"Highly correlated features found in {region} using {test} test:")
            print(highly_correlated_features)
        if test == "Pearson" and not correlated_features.empty and args.enable_plotting > 0:
            vis.visualize_feature_pairplot(X_train, y_train, correlated_features)

        highly_correlated = proc.find_fo_correlations(X_train, y_train, test)
        if highly_correlated.empty:
            print(f"No highly correlated features-outputs found in {region} with {test} test.")
        else:
            print(f"Highly correlated features and outputs found in {region} using {test} test:")
            print(highly_correlated)

    del X_train_enhancers
    del X_test_enhancers
    del X_train_promoters
    del X_test_promoters
    del y_train_enhancers
    del y_train_promoters
    del y_test_enhancers
    del y_test_promoters
    del x_enhancers_scaled
    del x_promoters_scaled


    # Feature selection with Random Forest and BorutaSHAP
    enhancers_selected_features = utils.run_boruta_shap(x_enhancers, y_enhancers, n_splits=3, task="enhancers", n_trials=80)
    promoters_selected_features = utils.run_boruta_shap(x_promoters, y_promoters, n_splits=3, task="promoters", n_trials=80)

    vis.apply_dimensionality_reduction(x_enhancers, y_enhancers,
                                       x_promoters, y_promoters,
                                       cfg.data_retrieval["assembly"],
                                       cell_line=cell_line)
    
    trainer.train_models(x_enhancers[enhancers_selected_features], y_enhancers, n_repeats=1, n_splits=3, task="Enhancers")
    trainer.train_models(x_promoters[promoters_selected_features], y_promoters, n_repeats=1, n_splits=3, task="Promoters")