"""Run CLUES-WEASEL on all the UCR univariate equal-length datasets."""

import os
import time
import warnings

import numpy as np
import pandas as pd
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.transformations.collection import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.experiments.utils import seeds

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    total_explained_variance_ratio = 0.2

    # Save the results in a dictionary
    results = {}

    # Get the number of development datasets
    n_datasets = len(univariate_equal_length)

    # For each data set
    for i, dataset_name in enumerate(sorted(univariate_equal_length)):

        # Print the current dataset
        print(f"[{i + 1:>{len(str(n_datasets))}}/{n_datasets}] {dataset_name}")

        results[dataset_name] = {}

        # Load and normalize the dataset(s)
        X_train, _ = load_classification(dataset_name, split="train", return_metadata=False)
        X_test, labels_true = load_classification(dataset_name, split="test", return_metadata=False)
        X_train_normalized = Normalizer().transform(X_train)
        X_test_normalized = Normalizer().transform(X_test)

        # Encode the labels and get the true number of clusters
        labels_true = LabelEncoder().fit_transform(labels_true)
        n_clusters = np.unique(labels_true).size

        # For each RNG seed
        for seed in seeds:

            print(f"seed = {seed}")

            transformer = WEASELTransformerV2(feature_selection="random", random_state=seed, n_jobs=1)

            pipeline = make_pipeline(transformer, VarianceThreshold(), StandardScaler(), PCA(random_state=seed))

            # We have to add fake labels in WEASELTransformerV2.fit_transform() in aeon==1.3.0
            fake_labels = np.zeros(X_train_normalized.shape[0])

            # Start (runtime)
            start_time = time.time()

            X_train_pca = pipeline.fit_transform(X_train_normalized, fake_labels)
            X_test_pca = pipeline.transform(X_test_normalized)
            cumulative_explained_variance_ratio = np.cumsum(pipeline[-1].explained_variance_ratio_)
            n_components = max(
                2, np.nonzero(cumulative_explained_variance_ratio >= total_explained_variance_ratio)[0][0]
            )
            kmeans = KMeans(n_clusters, random_state=seed)
            _ = kmeans.fit(X_train_pca[:, :n_components]).predict(X_test_pca[:, :n_components])

            # End (runtime)
            end_time = time.time()

            results[dataset_name][f"seed={seed}"] = end_time - start_time

    # Create the split subdirectory if it does not exist
    if not os.path.isdir(os.path.join("runtimes", "train-test", "CLUES-WEASEL")):
        os.makedirs(os.path.join("runtimes", "train-test", "CLUES-WEASEL"))

    df = pd.DataFrame(results).T
    df.to_csv(os.path.join("runtimes", "train-test", "CLUES-WEASEL", "runtimes.csv"))
