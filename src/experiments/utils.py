"""Utility functions"""

import os

import numpy as np
import pandas as pd
from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
from aeon.datasets.tsc_datasets import univariate_equal_length
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
)

# Define the RNG seeds for all the experiments
seeds = list(range(10))

# Define all the datasets
univariate_equal_length = sorted(univariate_equal_length)

# Define the development datasets
# Same as R-Clustering:  https://github.com/jorgemarcoes/R-Clustering/blob/main/R_Clustering_on_UCR_Archive.ipynb
development_datasets = [
    'Beef', 'BirdChicken', 'Car', 'CricketX', 'CricketY', 'CricketZ', 'DistalPhalanxTW', 'ECG200', 'ECG5000',
    'FiftyWords', 'Fish', 'FordA', 'FordB', 'Haptics', 'Herring', 'InsectWingbeatSound', 'ItalyPowerDemand',
    'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
    'OSULeaf', 'OliveOil', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'ScreenType',
    'ShapeletSim', 'Strawberry', 'SwedishLeaf', 'SyntheticControl', 'ToeSegmentation1', 'Trace',
    'UWaveGestureLibraryY', 'Wafer', 'WordSynonyms', 'Worms', 'Yoga'
]

# Define the evaluation datasets
evaluation_datasets = np.setdiff1d(univariate_equal_length, development_datasets).tolist()

# Define the metrics
metric_names_abbreviations = {
    "Rand index": "ri",
    "Adjusted Rand index": "ari",
    "Mutual information": "mi",
    "Adjusted mutual information": "ami",
    "Normalized mutual information": "nmi",
    "Clustering accuracy": "clacc",
    "Fowlkes-Mallows index": "fmi",
    "Homogeneity": "homogeneity",
    "Completeness": "completeness",
}

# Hyperparameter values for PCA
pca_total_explained_variance_ratios = [1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.1, 0.05]

# Hyperparameter optimization results
total_explained_variance_ratios = {
    "WEASELTransformerV2": 0.2,
    "MultiRocket": 0.9,
    "HydraTransformer": 0.7,
    "RandomDilatedShapeletTransform": 0.7,
    "QUANTTransformer": 0.7,
    "TSFresh": 1.0,
}


def scores(labels_true, labels_pred):
    """Compute several metrics to evaluate the quality of predicted clusters.

    Parameters
    ----------
    labels_true : np.ndarray of shape (n_samples,)
        True labels.

    labels_pred : np.ndarray of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    dict
        Dictionary with several metrics computed.
    """
    homogeneity, completeness, _ = homogeneity_completeness_v_measure(labels_true, labels_pred)

    dictionary = {
        "Rand index": rand_score(labels_true, labels_pred),
        "Adjusted Rand index": adjusted_rand_score(labels_true, labels_pred),
        "Mutual information": mutual_info_score(labels_true, labels_pred),
        "Adjusted mutual information": adjusted_mutual_info_score(labels_true, labels_pred),
        "Normalized mutual information": normalized_mutual_info_score(labels_true, labels_pred),
        "Clustering accuracy": clustering_accuracy_score(labels_true, labels_pred),
        "Fowlkes-Mallows index": fowlkes_mallows_score(labels_true, labels_pred),
        "Homogeneity": homogeneity,
        "Completeness": completeness,
    }

    return dictionary


def save_results(results, seeds, directory="."):
    """Save the results in a directory.

    Parameterss
    ----------
    results : dict
        Dictionary with the results

    seeds : list
        List of seeds

    directory : str, default = "."
        Directory where the results will be saved.

    clacc_rank : bool, default = False
        If True, compute the mean rank based on clustering accuracy scores and print the results.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)

    df = pd.DataFrame(results).T

    for metric_name, metric_abbreviation in metric_names_abbreviations.items():
        df_metric = df.map(lambda x: np.mean([x[f"seed={seed}"][metric_name] for seed in seeds]))
        df_metric.to_csv(os.path.join(directory, f"{metric_abbreviation}_mean.csv"))
