from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tqdm.auto import tqdm

import config as cfg



def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature scaling using robust scaling if the density of the features
    is greater than or equal to 0.85, otherwise uses maximum absolute value scaling.

    :param df: dataframe containing the features data.
    """
    density = _compute_density(df)
    if density >= 0.85:
        print("Performing Robust Scaling...")
        return pd.DataFrame(
            RobustScaler(copy=False).fit_transform(df.values),
            columns=df.columns,
            index=df.index
        )
    print("Performing Max Absolute Scaling...")
    return pd.DataFrame(
        MaxAbsScaler(copy=False).fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )


def check_constant_features(df: pd.DataFrame) -> bool:
    """
    Checks if there is any constant feature in the given dataframe.

    :param df: Pandas dataframe to check.
    :return: Returns true if there is at least a constant feature, false otherwise.
    """
    shape1 = df.shape
    return df.loc[:, (df != df.iloc[0]).any()].shape != shape1


def _compute_density(df: pd.DataFrame) -> float:
    """
    Computes the ratio of non-sparse points to total (dense) data points.

    :param df: Pandas dataframe on which to compute the sparsity.
    :return: Returns the density of the dataframe.
    """
    return df.astype(pd.SparseDtype("float", 0)).sparse.density


def find_ff_correlations(df: pd.DataFrame, corr_test: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finds correlations among features. This method can use both Pearson and Spearman correlations for the evaluation of
    correlated features. Typically, low correlations are desirable among features.

    :param df: Pandas dataframe on which to perform correlation tests.
    :param corr_test: Test to perform in order to evaluate feature correlations (it should be 'pearson' or 'spearman').
    :return: Returns the dataframe containing the correlated features.
    """
    if corr_test.lower() != "pearson" and corr_test.lower() != "spearman":
        raise ValueError("[ find_ff_correlations() ] - parameter 'corr_test' should be equal "
                         "to 'pearson' or 'spearman'.")
    columns = ['first_feature', 'second_feature', 'correlation']
    highly_correlated_features = pd.DataFrame(columns=columns)
    correlated_features = pd.DataFrame(columns=columns)
    if corr_test.lower() == "pearson":
        for first_feature in tqdm(df.columns, desc="Executing Pearson test between features", dynamic_ncols=True,
                                  leave=False):
            for second_feature in df.columns:
                if first_feature >= second_feature:
                    continue
                pearson = _perform_pearson_test(df, first_feature, second_feature)
                abs_corr = np.abs(pearson["correlation"])
                if pearson["p_value"] < cfg.preprocessing["p_value_threshold"]:
                    if abs_corr > (1 - cfg.preprocessing["corr_threshold"]):
                        highly_correlated_features = pd.concat(objs=(highly_correlated_features, pd.DataFrame(
                            [[first_feature, second_feature, pearson["correlation"]]], columns=columns
                        )))
                    if 0.8 <= abs_corr <= (1 - cfg.preprocessing["corr_threshold"]):
                        correlated_features = pd.concat(objs=(correlated_features, pd.DataFrame(
                            [[first_feature, second_feature, pearson["correlation"]]], columns=columns
                        )))
        return highly_correlated_features, correlated_features

    for first_feature in tqdm(df.columns, desc=f"Executing Spearman test between features", dynamic_ncols=True,
                              leave=False):
        for second_feature in df.columns:
            if first_feature >= second_feature:
                continue
            spearman = _perform_spearman_test(df, first_feature, second_feature)
            abs_corr = np.abs(spearman["correlation"])
            if spearman["p_value"] < cfg.preprocessing["p_value_threshold"]:
                if abs_corr > (1 - cfg.preprocessing["corr_threshold"]):
                    highly_correlated_features = pd.concat(objs=(highly_correlated_features, pd.DataFrame(
                        [[first_feature, second_feature, spearman["correlation"]]], columns=columns
                    )))
                if 0.8 <= abs_corr <= (1 - cfg.preprocessing["corr_threshold"]):
                    correlated_features = pd.concat(objs=(correlated_features, pd.DataFrame(
                        [[first_feature, second_feature, spearman["correlation"]]], columns=columns
                    )))
    return highly_correlated_features, correlated_features


def find_fo_correlations(df_data: pd.DataFrame, df_outputs: pd.DataFrame, corr_test: str) -> pd.DataFrame:
    """
    Finds correlations between features and outputs. This method uses both Pearson and Spearman correlations for the
    evaluation of correlated features and outputs. In this case, high correlation is desired.

    :param df_data: Pandas dataframe containing the features.
    :param df_outputs: Pandas dataframe containing the outputs.
    :param corr_test: Correlation test to perform (it can be "Pearson" or "Spearman").
    :return: Returns the dataframe containing features highly correlated to outputs.
    """
    columns = ['feature', 'correlation']
    features = pd.DataFrame(columns=columns)
    if corr_test.lower() == "pearson":
        for feature in tqdm(df_data.columns, desc="Executing Pearson test between features and outputs",
                            dynamic_ncols=True, leave=False):
            corr, p_value = pearsonr(df_data[feature].values.flatten(), df_outputs.values.flatten())
            if p_value > cfg.preprocessing["p_value_threshold"]:
                pd.concat(objs=(features, pd.DataFrame(
                    [[feature, corr]], columns=columns
                )))
        return features
    elif corr_test.lower() == "spearman":
        for feature in tqdm(df_data.columns, desc="Executing Pearson test between features and outputs",
                            dynamic_ncols=True, leave=False):
            corr, p_value = spearmanr(df_data[feature].values.flatten(), df_outputs.values.flatten())
            if p_value > cfg.preprocessing["p_value_threshold"]:
                pd.concat(objs=(features, pd.DataFrame(
                    [[feature, corr]], columns=columns
                )))
        return features
    else:
        raise ValueError("[ find_fo_correlations ] - Correlation test to perform should be 'Pearson' or 'Spearman'.")


def imputation(df: pd.DataFrame, imp_type: str, n_neighbors=5) -> pd.DataFrame:
    """
    Performs the data imputation process on the given dataframe.

    :param df: Pandas DataFrame on which to perform the imputation.
    :param imp_type: Type of the imputation (only 'median' and 'knn' are implemented).
    :param n_neighbors: Number of neighbors to consider for the knn imputation (default: 5).
    :return: Returns the imputed dataframe.
    """
    print("NaN values: ", df.isna().values.sum())
    if imp_type.lower() == "median":
        return df.fillna(df.median())
    elif imp_type.lower() == "knn":
        return pd.DataFrame(
            KNNImputer(n_neighbors=n_neighbors).fit_transform(df.values),
            columns=df.columns,
            index=df.index
        )


def _perform_pearson_test(df: pd.DataFrame,
                          first_feature: str,
                          second_feature: str) -> dict:
    """
    Performs Pearson correlation test on a pair of features.

    :param df: Pandas dataframe containing the original data.
    :param first_feature: first feature of the pair to check.
    :param second_feature: second feature of the pair to check.
    :return: Returns a dictionary containing correlation and p_value for the given pair of features.
    """
    corr, p_value = pearsonr(df[first_feature].values.flatten(), df[second_feature].values.flatten())
    return dict(correlation=corr, p_value=p_value)


def _perform_spearman_test(df: pd.DataFrame,
                           first_feature: str,
                           second_feature: str) -> dict:
    """
    Performs Spearman test on a pair of features.

    :param df: Pandas dataframe containing the original data.
    :param first_feature: first feature of the pair to check.
    :param second_feature: second feature of the pair to check.
    :return: Returns a dictionary containing correlation and p_value for the given pair of features.
    """
    corr, p_value = spearmanr(df[first_feature].values.flatten(), df[second_feature].values.flatten())
    return dict(correlation=corr, p_value=p_value)


def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the BED coordinates from the given dataframe.

    :param data: Pandas dataframe from which to get the coordinates.
    :return: BED coordinates.
    """
    return data.reset_index()[data.index.names]


def train_test_splits(X: pd.DataFrame, y:pd.DataFrame, train_size: float):
    """
    Utility function that returns a split of the given dataset.

    :param X: dataframe containing the samples.
    :param y: dataframe containing ground truth values.
    :param train_size: size of the training split expressed as a fraction of the entire dataset (float).
    """
    return train_test_split(X, y, train_size=train_size, random_state=42)