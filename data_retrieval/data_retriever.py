from typing import Tuple

from epigenomic_dataset import active_enhancers_vs_inactive_enhancers, active_promoters_vs_inactive_promoters
import pandas as pd
from ucsc_genomes_downloader import Genome

import config as cfg


def get_enhancers(metric: str, window_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves the enhancers data.
    
    :param metric: The metric to use in order to aggregate data.
    :param window_size: The window to use in order to process the data.
    :return: Returns two dataframes for the samples and the ground truth values.
    """
    x_enhancers, y_enhancers = active_enhancers_vs_inactive_enhancers(
        assembly=cfg.data_retrieval["assembly"],
        cell_line=cfg.data_retrieval["cell_line"],
        dataset=cfg.data_retrieval["dataset"],
        metric=metric,
        root=cfg.data_retrieval["root"],
        window_size=window_size
    )
    print("Ratio between enhancers samples and features: ",
          x_enhancers.shape[0]/x_enhancers.shape[1])
    return x_enhancers, y_enhancers


def get_genome():
    """
    Retrieves the entire genome from hg38 assembly.

    :return: Returns the genome for hg38 assembly.
    """
    return Genome(cfg.data_retrieval["assembly"])


def get_promoters(metric: str, window_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves the promoters data.
    
    :param metric: The metric to use in order to aggregate data.
    :param window_size: The window to use in order to process the data.
    :return: Returns two dataframes for the samples and the ground truth values.
    """
    x_promoters, y_promoters = active_promoters_vs_inactive_promoters(
        assembly=cfg.data_retrieval["assembly"],
        cell_line=cfg.data_retrieval["cell_line"],
        dataset=cfg.data_retrieval["dataset"],
        metric=metric,
        root=cfg.data_retrieval["root"],
        window_size=window_size
    )
    print("Ratio between promoters samples and features: ",
          x_promoters.shape[0]/x_promoters.shape[1])
    return x_promoters, y_promoters
