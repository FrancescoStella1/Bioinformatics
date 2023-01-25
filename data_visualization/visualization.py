import os

from barplots import barplots
from keras_bed_sequence import BedSequence
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import table
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import KBinsDiscretizer


from data_preprocessing.preprocessing import to_bed
from data_retrieval.data_retriever import get_genome

import config as cfg


def apply_dimensionality_reduction(enhancers_data: pd.DataFrame, enhancers_labels: pd.DataFrame,
                                   promoters_data: pd.DataFrame, promoters_labels: pd.DataFrame,
                                   assembly: str, cell_line: str) -> None:
    """
    Applies PCA and T-SNE algorithms to enhancer and promoter regions.

    :param enhancers_data: dataframe containing the enhancers data.
    :param enhancers_labels: dataframe containing the enhancers labels.
    :param promoters_data: dataframe containing the promoters data.
    :param promoters_labels: dataframe containing the promoters labels.
    :param assembly: assembly used for retrieving the genome.
    :param cell_line: the cell line used for the analysis.
    """
    fig, axes = plt.subplots(nrows=4, ncols=2, squeeze=False)
    row = 0
    for decomposition_method in (_apply_pca, _apply_tsne):
        for ((epigenomic, scores), region) in [((enhancers_data, enhancers_labels), "Enhancers"),
                                               ((promoters_data, promoters_labels), "Promoters")]:
            decomposed_epigenomic = decomposition_method(epigenomic)
            _scatter_with_heatmap(decomp=decomposed_epigenomic,
                                  scores=scores.values,
                                  figure=fig,
                                  ax=axes[row][0])
            axes[row][0].set_title(f"{cell_line} - {region} {assembly}, epigenomic data (PCA)")
            axes[row][0].set_axis_off()

            sequence = pd.DataFrame(
                np.array(BedSequence(get_genome(),
                                     bed=to_bed(scores),
                                     batch_size=1)
                         ).reshape(-1, 4*256),
                index=scores.index)
            decomposed_sequence = decomposition_method(sequence)
            _scatter_with_heatmap(decomp=decomposed_sequence,
                                  scores=scores.values,
                                  figure=fig,
                                  ax=axes[row][1])
            axes[row][1].set_title(f"{cell_line} - {region} {assembly}, sequence data (T-SNE)")
            axes[row][1].set_axis_off()
            row += 1
    fig.tight_layout()
    plt.show()


def _apply_pca(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Applies PCA algorithm on the given DataFrame.

    :param df: Pandas dataframe on which to perform the PCA decomposition.
    :return: Returns the DataFrame containing the 2 principal components.
    """
    return pd.DataFrame(PCA(n_components=2, random_state=42).fit_transform(df.values), index=df.index)


def _apply_tsne(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Applies TSNE on the given DataFrame.

    :param df: Pandas dataframe on which to perform the TSNE algorithm.
    :return: Returns the DataFrame containing the 2D components found by TSNE.
    """
    # Cosine distance currently not supported, only euclidean distance or inner product (default: euclidean)
    return pd.DataFrame(TSNE(init='pca', learning_rate='auto', n_iter=300,
                             random_state=42, verbose=True).fit_transform(df),
                        index=df.index)


def labels_clipping(labels: pd.DataFrame, region: str) -> None:
    """
    Performs labels clipping on the ground truth values for different values of the clip
    and plots the related amount of samples.

    :param labels: Pandas dataframe containing the ground truth values.
    :param region: The region (enhancer or promoters) for which to perform labels clipping.
    """
    plt.figure()
    samples_per_clip = []
    clips = [0, 1, 3, 5, 7, 10, 50, 100, 250, 500, 1000, 1500, 2000]
    for clip in clips:
        samples_per_clip.append(np.sum(np.where(labels > clip)))
    print(samples_per_clip)
    plt.plot(samples_per_clip)
    plt.xticks(range(len(clips)), clips)
    plt.xlabel("clip value")
    plt.ylabel("# of samples")
    plt.title(f"Samples vs clip value ({region})")
    plt.show()


def plot_training_stats(performance: pd.DataFrame, task: str) -> None:
    """
    Plots some statistics related to the training and evaluation of the models.

    :param performance: Pandas dataframe containing relevant statistics to plot.
    """
    performance = performance.drop(columns=["loss"])
    barplots(performance.drop(columns=["fold_number"]),
             groupby=["task", "model_name", "run_type"],
             height=8,
             orientation="horizontal",
             subplots=True
             )
    plt.tight_layout()
    plt.show()

    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    try:
        for model_name in ["FFNN_regression_model", "CNN_regression_model",
                           "MMNN_boosted_regression_model"]:
            filtered_performance = performance[performance['model_name'] == model_name].reset_index(drop=True)
            table(ax, filtered_performance)
            img_path = os.path.join(cfg.images["path"], f"{model_name}_{task}_table.png")
            plt.savefig(img_path, bbox_inches="tight", dpi=200)
    except:
        pass


def _scatter_with_heatmap(decomp: pd.DataFrame, scores: np.ndarray, figure: plt.Figure, ax: plt.Axes) -> None:
    """
    Visualizes a scatterplot for a given decomposition.

    :param decomp: decomposition to visualize.
    :param scores: ground truth values.
    :param figure: Matplotlib Figure.
    :param ax: Matplotlib Axes.
    """
    scatter = ax.scatter(*decomp.values.T, c=scores, cmap=plt.cm.get_cmap('RdYlBu'), norm=LogNorm(), s=3)
    color_bar = figure.colorbar(scatter, ax=ax)


def visualize_feature_pairplot(df_data: pd.DataFrame, df_outputs: pd.DataFrame, df_corr: pd.DataFrame) -> None:
    """
    Visualizes a pairplot of a few correlated features.

    :param df_data: Pandas dataframe containing all the features.
    :param df_outputs: Pandas dataframe containing the outputs.
    :param df_corr: Pandas dataframe containing the correlation values for a few correlated features.
    :param region: Region of interest (should be 'enhancers' or 'promoters').
    """
    print("Visualizing pairplot...")
    df_corr = df_corr.sort_values('correlation', ascending=False).head(3)
    features = set(df_corr['first_feature'].values).union(set(df_corr['second_feature'].values))
    df_data = df_data[list(features)]
    discretized = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit_transform(df_outputs)
    discretized = pd.DataFrame(discretized, columns=['value'], index=df_outputs.index)
    sns.pairplot(pd.concat(objs=(df_data, discretized), axis=1).sample(5000, random_state=42),
                 hue=discretized.columns[0],
                 palette=sns.color_palette("ch:s=.25,rot=-.25", np.unique(discretized).shape[0]),
                 plot_kws={'alpha': .7})
    plt.show()
