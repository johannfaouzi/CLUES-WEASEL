"""Perform hyperparameter optimization."""

import os
import warnings

import numpy as np
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.datasets import load_classification
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.feature_based import TSFresh
from aeon.transformations.collection.interval_based._quant import QUANTTransformer
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.experiments.utils import (
    development_datasets,
    pca_total_explained_variance_ratios,
    save_results,
    scores,
    seeds,
)

# Ignore warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    transformer_names = [
        "WEASELTransformerV2",
        "MultiRocket",
        "HydraTransformer",
        "RandomDilatedShapeletTransform",
        "QUANTTransformer",
        "TSFresh",
    ]

    # Save the results in a dictionary
    results = {}

    for transformer_name in transformer_names:
        results[transformer_name] = {}

    # Get the number of development datasets
    n_datasets = len(development_datasets)

    # For each data set
    for i, dataset_name in enumerate(development_datasets):

        # Print the current dataset
        print(f"[{i + 1:>{len(str(n_datasets))}}/{n_datasets}] {dataset_name}")

        # Create a new (key, value) pair for the dataset for each transformer
        for transformer_name in transformer_names:
            results[transformer_name][dataset_name] = {}

        # Load the training and test sets
        X_train, _ = load_classification(dataset_name, split="train", return_metadata=False)
        X_test, labels_true = load_classification(dataset_name, split="test", return_metadata=False)

        # Normalize both data sets
        X_train_normalized = Normalizer().transform(X_train)
        X_test_normalized = Normalizer().transform(X_test)

        # Encode the labels and get the number of classes
        le = LabelEncoder()
        labels_true = le.fit_transform(labels_true)
        n_classes = le.classes_.size

        # For each RNG seed
        for seed in seeds:

            print(f"seed = {seed}")

            transformers = [
                WEASELTransformerV2(feature_selection="random", random_state=seed, n_jobs=-1),
                MultiRocket(random_state=seed, n_jobs=-1),
                HydraTransformer(output_type="numpy", random_state=seed, n_jobs=-1),
                RandomDilatedShapeletTransform(random_state=seed, n_jobs=-1),
                QUANTTransformer(),
                TSFresh(n_jobs=-1),
            ]

            # For each transformer
            for transformer_name, transformer in zip(transformer_names, transformers):

                print(transformer_name)

                # Initialize the pipeline
                pipeline = make_pipeline(transformer, VarianceThreshold(), StandardScaler())

                # Fit and transform on the training set
                if transformer_name == "WEASELTransformerV2":
                    # We have to add fake labels in WEASELTransformerV2.fit_transform() in aeon==1.3.0
                    fake_labels = np.zeros(X_train_normalized.shape[0])
                    X_train_new = pipeline.fit_transform(X_train_normalized, fake_labels)
                else:
                    X_train_new = pipeline.fit_transform(X_train_normalized)

                # Transform on the test set
                X_test_new = pipeline.transform(X_test_normalized)

                # PCA
                pca = PCA(random_state=seed)
                X_train_pca = pca.fit_transform(X_train_new)
                X_test_pca = pca.transform(X_test_new)

                del X_train_new, X_test_new

                cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                for total_explained_variance_ratio in pca_total_explained_variance_ratios:
                    if total_explained_variance_ratio == 1.0:
                        n_components = X_train_pca.shape[1]
                    else:
                        n_components = max(
                            2, np.nonzero(cumulative_explained_variance_ratio >= total_explained_variance_ratio)[0][0]
                        )
                    kmeans_pca = KMeans(n_clusters=n_classes, n_init=1, random_state=seed)
                    labels_pred = kmeans_pca.fit(X_train_pca[:, :n_components]).predict(X_test_pca[:, :n_components])

                    if f"PCA({total_explained_variance_ratio})" not in results[transformer_name][dataset_name]:
                        results[transformer_name][dataset_name][f"PCA({total_explained_variance_ratio})"] = {}

                    results[transformer_name][dataset_name][f"PCA({total_explained_variance_ratio})"][
                        f"seed={seed}"
                    ] = scores(labels_true, labels_pred)

                del X_train_pca, X_test_pca

    for transformer_name in transformer_names:
        save_results(
            results[transformer_name], seeds, directory=os.path.join("hyperparameter_optimization", transformer_name)
        )
