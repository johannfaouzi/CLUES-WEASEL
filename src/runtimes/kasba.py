"""Run KASBA on all the UCR univariate equal-length datasets."""

import os
import time
import warnings

import numpy as np
import pandas as pd
from aeon.clustering import KASBA
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.transformations.collection import Normalizer
from sklearn.preprocessing import LabelEncoder

from src.experiments.utils import seeds

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

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

            clusterer = KASBA(n_clusters=n_clusters, random_state=seed)

            # Start (runtime)
            start_time = time.time()

            _ = clusterer.fit(X_train_normalized).predict(X_test_normalized)

            # End (runtime)
            end_time = time.time()

            results[dataset_name][f"seed={seed}"] = end_time - start_time

    # Create the split subdirectory if it does not exist
    if not os.path.isdir(os.path.join("runtimes", "train-test", "KASBA")):
        os.makedirs(os.path.join("runtimes", "train-test", "KASBA"))

    df = pd.DataFrame(results).T
    df.to_csv(os.path.join("runtimes", "train-test", "KASBA", "runtimes.csv"))
